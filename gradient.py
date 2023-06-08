import functools
from inspect import Parameter
import json
import math
# import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(params):
    """The quantum circuit that you will differentiate!

    Args:
        params (list(float)): The parameters for gates in the circuit

    Returns:
        (numpy.array): An expectation value.
    """
    qml.RY(params[0], 0)
    qml.RX(params[1], 1)
    return qml.expval(qml.PauliZ(0) + qml.PauliZ(1))


def my_parameter_shift_grad(params, shift):
    """Your homemade parameter-shift rule function.
    
    NOTE: you cannot use qml.grad within this function

    Args:
        params (list(float)): The parameters for gates in the circuit

    Returns:
        gradient (numpy.array): The gradient of the circuit with respect to the given parameters.
    """

    gradient = np.zeros_like(params)
    for i in range(len(params)):
        # Put your code here #
        params[i] += shift
        a = circuit(params)
        params[i] -= 2*shift
        b = circuit(params)
        params[i] += shift
        gradient[i] = (a-b)/(2*np.sin(shift))
    return np.round_(gradient, decimals=5).tolist()

print(my_parameter_shift_grad([0.75, 1.0], 1.23))