import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    """

    A = 1/(1+np.exp(-Z))
    return A

def softmax(Z):
    """
    Compute softmax values for each sets of scores in x.

    Z -- numpy array of any shape
    """
    exps = np.exp(Z)
    return exps / np.sum(exps, axis=0, keepdims=True)

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    return A

def softmax_backward(Y_hat, Y):
    """
    Compute softmax derivative.

    A -- numpy array of any shape
    """
    return Y_hat - Y


def relu_backward(Z):
    return 1. * (Z > 0)


def sigmoid_backward(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ
