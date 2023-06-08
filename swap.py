import functools
import json
import math
# import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

def U():
    """
    This quantum function will simply contain H, T and CNOT gates.
    It will not return any value.
    """
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])

    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)
    qml.T(wires=0)

    qml.T(wires=1)
    qml.T(wires=1)
    qml.T(wires=1)
    qml.T(wires=1)
    qml.T(wires=1)
    qml.T(wires=1)
    qml.T(wires=1)

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)

    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0)
    qml.T(wires=0) 
    qml.T(wires=1)
    qml.T(wires=1)

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1) 


dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit():
    """
    Main circuit given in the statement, here the operators you have defined in U will be embedded.
    """

    # qml.CNOT(wires=[0, 1])

    # qml.Hadamard(wires=0)
    # qml.Hadamard(wires=1)
    # qml.SWAP(wires=[0,1])
# [[ 0.5  0.5  0.5  0.5]
#  [ 0.5 -0.5 -0.5  0.5]
#  [ 0.5  0.5 -0.5 -0.5]
#  [ 0.5 -0.5  0.5 -0.5]]

    # U()
    # solution = np.array([[1,0,0,0],
    # [0,0,0,1],
    # [0,0,1,0],
    # [0,1,0,0]])
    # decomp = qml.transforms.two_qubit_decomposition(solution, wires=[0, 1])
    # print(decomp)
    # qml.Hadamard(wires=0)
    # qml.Hadamard(wires=1)
    # qml.CNOT(wires=[0, 1])

    # qml.CNOT(wires=[0, 1])
    # qml.Hadamard(wires=0)
    # qml.T(wires=0)
    # qml.adjoint(qml.T(wires=1))
    # qml.Hadamard(wires=0)
    # qml.Hadamard(wires=1)
    # qml.CNOT(wires=[0, 1])
    # qml.Hadamard(wires=0)
    # qml.Hadamard(wires=1)
    # qml.adjoint(qml.T(wires=0))
    # qml.Hadamard(wires=0)
    # qml.CNOT(wires=[0, 1])
    # qml.adjoint(qml.T(wires=0))
    # qml.adjoint(qml.T(wires=0))
    # qml.T(wires=1)
    # qml.T(wires=1)

    # qml.Hadamard(wires=0)
    # qml.Hadamard(wires=1) 
    qml.CNOT(wires=[0, 1])

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    U()

    qml.CNOT(wires=[1, 0])

    U()

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
# [[1. 0. 0. 0.]
#  [0. 0. 0. 1.]
#  [0. 0. 1. 0.]
#  [0. 1. 0. 0.]]
    # U()
# [[ 0.5  0.5  0.5  0.5]
#  [ 0.5 -0.5  0.5 -0.5]
#  [ 0.5  0.5 -0.5 -0.5]
#  [ 0.5 -0.5 -0.5  0.5]]
    # qml.Hadamard(wires=0)
    # qml.Hadamard(wires=1)

    return qml.state()

# sqrt_cswap = np.array([[ 1, 0,  0,   0 ],
# [0,   0.5+0.5j,  0.5-0.5j, 0],
# [0,   0.5-0.5j , 0.5+0.5j,0],
# [0,   0, 0,1 ]])

# two_C = np.array([[1,0,0,0],
# [0,0,0,1],
# [0,1,0,0],
# [0,0,1,0]])

# C_inv = np.array([[ 0.5,  0.5 , 0.5  ,0.5],
# [ 0.5, -0.5 , 0.5 ,-0.5],
# [ 0.5 , 0.5 ,-0.5 ,-0.5],
# [ 0.5 ,-0.5 ,-0.5  ,0.5]]
# ) 

# H_inv = np.array([[ 0.5,  0.5 , 0.5  ,0.5],
# [ 0.5, -0.5 , 0.5 ,-0.5],
# [ 0.5 , 0.5 ,-0.5 ,-0.5],
# [ 0.5 ,-0.5 ,-0.5  ,0.5]]
# )
# swap = np.array([[1,0,0,0],
# [0,0,1,0],
# [0,1,0,0],
# [0,0,0,1]])
# t = np.matmul(np.linalg.inv(H_inv), two_C)
# m = np.matmul(t, np.linalg.inv(H_inv))
# n = np.linalg.inv(H_inv)
# print(m, n)
# cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
# a = np.matmul(sqrt_cswap, H_inv) 
# print(np.matmul(H_inv, a))
matrix = abs(qml.matrix(circuit)()).real.round(1)
print(matrix)
# These functions are responsible for testing the solution.

def run(input: str) -> str:
    matrix = qml.matrix(circuit)().real

    with qml.tape.QuantumTape() as tape:
        U()

    names = [op.name for op in tape.operations]
    return json.dumps({"matrix": matrix.tolist(), "gates": names})

def check(user_output: str, expected_output: str) -> str:

    parsed_output = json.loads(user_output)
    matrix_user = np.array(parsed_output["matrix"])
    gates = parsed_output["gates"]

    assert np.allclose(matrix_user, qml.matrix(qml.SWAP)(wires=[0, 1]))
    assert (
        len(set(gates)) == 3
        and "Hadamard" in gates
        and "CNOT" in gates
        and "T" in gates
    )


test_cases = [['No input', 'No output']]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")