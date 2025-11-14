import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterable, List, Dict


# ------------------------------
# Simple protocol-like interfaces
# ------------------------------


class STTEngine:
    """
    Very simple interface for STT.
    Internally you might stream audio into it and then call finalize().
    """

    async def accept_audio(self, frame: Any) -> None:
        ...

    async def finalize(self) -> str:
        """
        Return the final transcript for the last chunk of speech.
        """
        return ""


class LLMEngine:
    async def complete(self, history: List[Dict[str, str]]) -> str:
        """
        Given chat history, return assistant reply text.
        history: [{ "role": "user"|"assistant", "content": "..." }, ...]
        """
        return "Hello from the LLM"


class TTSEngine:
    async def stream_tts(self, text: str) -> AsyncIterable[Any]:
        """
        Yield audio frames for the reply text.
        """
        if False:
            yield None  # pragma: no cover


class AudioInput:
    """
    Wraps your incoming audio. For LiveKit you would adapt this around the audio track.
    """

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class AudioOutput:
    """
    Wraps your outgoing audio. For LiveKit this would publish to a LocalAudioTrack.
    """

    async def play(self, frame: Any) -> None:
        ...


# ------------------------------
# Config and Agent definition
# ------------------------------


@dataclass
class MinimalVoiceOptions:
    silence_timeout: float = 0.7  # seconds of silence to treat as end-of-turn


class MinimalAgent:
    """
    LiveKit-style agent with hooks.
    You can override these in your own subclass.
    """

    def __init__(self, label: str = "minimal-agent") -> None:
        self.label = label

    async def on_enter(self) -> None:
        """
        Called when the session starts.
        """
        pass

    async def on_user_turn_completed(
        self,
        chat_history: List[Dict[str, str]],
        new_message: Dict[str, str],
    ) -> None:
        """
        Called when we have a final user message, before LLM is called.
        You can modify chat_history if you want.
        """
        pass

    async def on_exit(self) -> None:
        """
        Called when the session is shutting down.
        """
        pass


# ------------------------------
# MinimalAgentSession (LiveKit-ish)
# ------------------------------


class MinimalAgentSession:
    """
    Inspired by livekit.agents.AgentSession but much simpler.
    Manages chat history and the main loop.
    """

    def __init__(
        self,
        *,
        stt: STTEngine,
        llm: LLMEngine,
        tts: TTSEngine,
        audio_in: AudioInput,
        audio_out: AudioOutput,
        options: MinimalVoiceOptions | None = None,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.options = options or MinimalVoiceOptions()

        # global chat history, like ChatContext
        self.history: List[Dict[str, str]] = []

        self._activity: MinimalAgentActivity | None = None
        self._running = False

    async def start(self, agent: MinimalAgent) -> None:
        """
        LiveKit-style start. Creates an activity and starts forwarding audio.
        """
        if self._running:
            return

        self._running = True
        self._activity = MinimalAgentActivity(
            agent=agent,
            session=self,
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            audio_in=self.audio_in,
            audio_out=self.audio_out,
            options=self.options,
        )

        await agent.on_enter()
        await self._activity.run()
        await agent.on_exit()
        self._running = False

    async def say(self, text: str) -> None:
        """
        Like AgentSession.say: speak some text but do not change history.
        """
        async for frame in self.tts.stream_tts(text):
            await self.audio_out.play(frame)

    async def generate_reply(self, user_input: str) -> None:
        """
        Like AgentSession.generate_reply: update history and respond via TTS.
        """
        if not user_input.strip():
            return

        # 1. Add user message to history
        user_msg = {"role": "user", "content": user_input}
        self.history.append(user_msg)

        # 2. Call LLM
        reply = await self.llm.complete(self.history)

        # 3. Add assistant message to history
        asst_msg = {"role": "assistant", "content": reply}
        self.history.append(asst_msg)

        # 4. Speak reply via TTS
        async for frame in self.tts.stream_tts(reply):
            await self.audio_out.play(frame)


# ------------------------------
# MinimalAgentActivity (LiveKit-ish)
# ------------------------------


class MinimalAgentActivity:
    """
    Inspired by AgentActivity: owns the running pipeline for one agent.
    It:
      - receives audio frames via audio_in
      - feeds them to STT
      - detects end of turn using a simple silence timeout
      - calls session.generate_reply(user_text)
    """

    def __init__(
        self,
        agent: MinimalAgent,
        session: MinimalAgentSession,
        stt: STTEngine,
        llm: LLMEngine,
        tts: TTSEngine,
        audio_in: AudioInput,
        audio_out: AudioOutput,
        options: MinimalVoiceOptions,
    ) -> None:
        self.agent = agent
        self.session = session
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.options = options

        self._last_speech_ts: float | None = None
        self._speaking = False

    async def run(self) -> None:
        """
        Main loop: consume audio frames, detect end-of-turn, respond.
        """

        silence_timeout = self.options.silence_timeout

        async for frame in self.audio_in:
            now = time.time()

            # 1. Feed audio to STT engine
            await self.stt.accept_audio(frame)
            self._speaking = True
            self._last_speech_ts = now

            # 2. Small sleep to avoid tight loop
            await asyncio.sleep(0.001)

            # 3. Check for silence-based end-of-turn
            #    In a real setup you would use VAD or STT events instead.
            if self._speaking and self._last_speech_ts is not None:
                if (time.time() - self._last_speech_ts) > silence_timeout:
                    self._speaking = False
                    await self._handle_user_turn()

        # When audio_in ends, flush one last time
        if self._speaking:
            await self._handle_user_turn()

    async def _handle_user_turn(self) -> None:
        """
        Called when we detect end-of-turn.
        Finalize STT, run LLM, run TTS.
        """

        # 1. Get final transcript
        user_text = (await self.stt.finalize()).strip()
        if not user_text:
            return

        # 2. Let agent modify chat history or inspect the message
        new_message = {"role": "user", "content": user_text}
        await self.agent.on_user_turn_completed(self.session.history, new_message)

        # 3. Use session-level generate_reply, which handles history + TTS
        await self.session.generate_reply(user_text)
