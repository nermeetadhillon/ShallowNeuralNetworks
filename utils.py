# Utility functions used for HA4

import numpy as np
import os
import sys


def load_data():
    data_files = ['X_train.npy', 'Y_train.npy', 'X_test.npy', 'Y_test.npy']
    for df in data_files:
        if not os.path.exists(df):
            sys.stderr.write('Make sure that {} is in the current directory'.format(df))
            sys.stderr.flush()
            sys.exit(1)

    X_train = np.load(open('X_train.npy', 'rb'))
    Y_train = np.load(open('Y_train.npy', 'rb'))
    X_test = np.load(open('X_test.npy', 'rb'))
    Y_test = np.load(open('Y_test.npy', 'rb'))

    return X_train, Y_train, X_test, Y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forwardprop_testcase():
    np.random.seed(2)

    X = np.random.randn(3, 5)
    W1 = np.array([[-1.99538791, -1.06742819, -0.75526442],
       [ 0.29453881, -0.98914596, -1.22217581],
       [-0.2728255 , -1.60988139,  0.10709325],
       [-0.14254936,  1.07290263,  0.42090126],
       [ 0.59674457,  0.86014672, -1.60764156]])
    b1 = np.array([[0.],
       [0.],
       [0.],
       [0.],
       [0.]])
    W2 = np.array([[ 1.08267975, -0.25012972,  1.10267451,  0.19690611,  1.46066089]])
    b2 = np.array([[0.]])

    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return X, params


def backprop_testcase():
    np.random.seed(2)

    Y = np.random.randint(2, size=(1, 5))
    X, params = forwardprop_testcase()

    Z1 = np.array([[ 1.31360482, -2.15573881,  5.56042225, -1.29935921,  4.14176847],
       [ 0.03588585, -3.31547707,  0.55181058,  2.89589401, -0.28791921],
       [ 1.52787259, -0.5487484 ,  2.59202347,  1.13594762,  2.01041891],
       [-0.6115967 ,  1.51235684, -1.01407549, -1.8394355 , -0.49273324],
       [-1.85926466, -3.28607399, -2.41267434,  1.86605197, -2.71871542]])
    A1 = np.array([[ 0.8651847 , -0.97352766,  0.9999704 , -0.86155811,  0.99949484],
       [ 0.03587046, -0.99736567,  0.50187598,  0.99391354, -0.28021853],
       [ 0.91005983, -0.49958158,  0.98885194,  0.81304502,  0.96475634],
       [-0.54525008,  0.90735641, -0.76744242, -0.95074093, -0.45638308],
       [-0.95261085, -0.99720633, -0.98408023,  0.95323485, -0.9913364 ]])
    Z2 = np.array([[ 0.43244098, -2.63334003,  0.45897383,  0.92027136,  0.67816511]])
    A2 = np.array([[0.6064564 , 0.06702329, 0.61277071, 0.71509739, 0.66332905]])

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    return X, Y, params, cache


def update_params_testcase():
    _, params = forwardprop_testcase()

    dW2 = np.array([[ 0.21853072,  0.25655992,  0.37129089, -0.37252822,  0.15155831]])
    db2 = np.array([[0.13293537]])
    dW1 = np.array([[ 0.05203932, -0.07544746, -0.0505175 ],
       [ 0.03578307,  0.03583598, -0.03154186],
       [ 0.0726706 , -0.16038869, -0.39562689],
       [-0.02589183, -0.03124887,  0.00223595],
       [ 0.02619004, -0.03334111, -0.01412392]])
    db1 = np.array([[ 0.06251278],
       [-0.04657385],
       [-0.06973409],
       [ 0.02740995],
       [ 0.03372477]])

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    return params, grads


def nn_model_testcase():
    X, Y, _, _ = backprop_testcase()
    return X, Y


def linear_forward_testcase():
    np.random.seed(2)
    
    A = np.random.randn(3, 2)
    W = np.random.randn(5, 3)*0.01
    b = np.zeros((5, 1))

    return A, W, b


def activation_forwad_testcase():
    Z = np.array([1,-1,0])
    return Z


def forward_testcase():
    A, W, b = linear_forward_testcase()
    return A, W, b


def forward_deep_testcase():
    np.random.seed(2)

    X = np.random.randn(10, 100)
    W1 = np.random.randn(200, 10)*0.01
    b1 = np.zeros((200, 1))
    W2 = np.random.randn(100, 200)*0.01
    b2 = np.zeros((100, 1)) 
    W3 = np.random.randn(50, 100)*0.01
    b3 = np.zeros((50, 1)) 
    W4 = np.random.randn(10, 50)*0.01
    b4 = np.zeros((10, 1))
    W5 = np.random.randn(1, 10)*0.01
    b5 = np.zeros((1, 1))

    params = {
        'W1': W1, 'b1': b1,\
        'W2': W2, 'b2': b2,\
        'W3': W3, 'b3': b3,\
        'W4': W4, 'b4': b4,\
        'W5': W5, 'b5': b5,
    }

    return X, params


def activation_backward_testcase():
    np.random.seed(2)

    dA = np.random.randn(4,3)
    activation_cache = np.random.randn(4,3)

    return dA, activation_cache


def linear_backward_testcase():
    np.random.seed(2)

    dZ = np.random.randn(4,3)
    A_prev = np.random.randn(2,3)
    W = np.random.randn(4,2)
    b = np.random.randn(4,1)
    linear_cache = (A_prev, W, b)

    return dZ, linear_cache


def backward_testcase():
    dA, activation_cache = activation_backward_testcase()
    dZ, linear_cache = linear_backward_testcase()
    cache = (linear_cache, activation_cache)

    return dA, cache


def backward_deep_testcase():
    """
    X = np.random.rand(3,2)
    Y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}
    aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
   np.array([[ 0.41791293,  1.91720367]]))])
   """
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A0 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    cache1 = ((A0, W1, b1), Z1)

    A1 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    cache2 = ( (A1, W2, b2), Z2)

    caches = (cache1, cache2)

    return AL, Y, caches