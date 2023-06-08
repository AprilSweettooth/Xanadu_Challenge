import functools
import json
import math
# import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

np.random.seed(1967)


def get_matrix(params):
    """
    Args:
        - params (array): The four parameters of the model.
    Returns:
        - (matrix): The associated matrix to these parameters.
    """
    alpha, beta, gamma, phi = params
    Rz_gamma = np.array([[ math.e**(-1j*gamma/2),  0], [ 0, math.e**(1j*gamma/2)]])
    Rz_alpha = np.array([[ math.e**(-1j*alpha/2),  0], [ 0, math.e**(1j*alpha/2)]])
    Rx_beta = np.array([[  (math.e**(1j*beta/2)+math.e**(-1j*beta/2))/2,  (math.e**(-1j*beta/2)-math.e**(1j*beta/2))/2], [ (math.e**(-1j*beta/2)-math.e**(1j*beta/2))/2, (math.e**(1j*beta/2)+math.e**(-1j*beta/2))/2]])
    temp = np.matmul(Rx_beta, Rz_alpha)
    return  (math.e**(1j*phi))*np.matmul(Rz_gamma, temp)
    # Put your code here #

def error(U, params):
    """
    This function determines the similarity between your generated matrix and the target unitary.

    Args:
        - U (matrix): Goal matrix that we want to approach.
        - params (array): The four parameters of the model.

    Returns:
        - (float): Error associated with the quality of the solution.
    """
    matrix = get_matrix(params)
    print(matrix)
    e = 0
    e_mat = U-matrix
    for i in range(2):
        for j in range(2):
            e += abs(e_mat[i][j])
    # Put your code here #
    # print(e)
    return e

# print(error([[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]],[ 0.70710678,  0.70710678, 0.70710678, -0.70710678]))
def train_parameters(U):

    epochs = 1
    lr = 0.01

    grad = qml.grad(error, argnum=1)
    params = np.random.rand(4) * np.pi

    for epoch in range(epochs):
        params -= lr * grad(U, params)
        # print(params)
        # print(params)

    return params

print(train_parameters([[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]]))
# These functions are responsible for testing the solution.

# def run(test_case_input: str) -> str:
#     matrix = json.loads(test_case_input)
#     params = [float(p) for p in train_parameters(matrix)]
#     return json.dumps(params)

# def check(solution_output: str, expected_output: str) -> None:
#     matrix1 = get_matrix(json.loads(solution_output))
#     matrix2 = json.loads(expected_output)
#     assert not np.allclose(get_matrix(np.random.rand(4)), get_matrix(np.random.rand(4)))
#     assert np.allclose(matrix1, matrix2, atol=0.2)


# test_cases = [['[[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]]', '[[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]]'], ['[[ 1,  0], [ 0, -1]]', '[[ 1,  0], [ 0, -1]]']]

# for i, (input_, expected_output) in enumerate(test_cases):
#     print(f"Running test case {i} with input '{input_}'...")

#     try:
#         output = run(input_)

#     except Exception as exc:
#         print(f"Runtime Error. {exc}")

#     else:
#         if message := check(output, expected_output):
#             print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

#         else:
#             print("Correct!")