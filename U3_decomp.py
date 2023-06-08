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
    Had = (1/np.sqrt(2)) * np.array(
            [
                [1, 1],
                [1, -1]
            ]
        ) 
    U31 = np.array([[ (math.e**(1j*params[0,0]/2)+math.e**(-1j*params[0,0]/2))/2,  -(math.e**(1j*params[0,1]))*(math.e**(1j*params[0,0]/2)-math.e**(1j*params[0,0]/2))/(2*1j)], [ (math.e**(1j*params[0,2]))*(math.e**(1j*params[0,0]/2)-math.e**(1j*params[0,0]/2))/(2*1j), (math.e**(1j*(params[0,1]+params[0,2])))*(math.e**(1j*params[0,0]/2)+math.e**(1j*params[0,0]/2))/2]])
    U32 = np.array([[ (math.e**(1j*params[1,0]/2)+math.e**(-1j*params[1,0]/2))/2,  -(math.e**(1j*params[1,1]))*(math.e**(1j*params[1,0]/2)-math.e**(1j*params[1,0]/2))/(2*1j)], [ (math.e**(1j*params[1,2]))*(math.e**(1j*params[1,0]/2)-math.e**(1j*params[1,0]/2))/(2*1j), (math.e**(1j*(params[1,1]+params[1,2])))*(math.e**(1j*params[1,0]/2)+math.e**(1j*params[1,0]/2))/2]])
    U33 = np.array([[ (math.e**(1j*params[2,0]/2)+math.e**(-1j*params[2,0]/2))/2,  -(math.e**(1j*params[2,1]))*(math.e**(1j*params[2,0]/2)-math.e**(1j*params[2,0]/2))/(2*1j)], [ (math.e**(1j*params[2,2]))*(math.e**(1j*params[2,0]/2)-math.e**(1j*params[2,0]/2))/(2*1j), (math.e**(1j*(params[2,1]+params[2,2])))*(math.e**(1j*params[2,0]/2)+math.e**(1j*params[2,0]/2))/2]])
    w1 = np.matmul(Had,U31)
    w2 = np.matmul(Had,U32)
    w3 = np.matmul(Had,U33)
    temp = np.kron(w1,w2)
    return  np.kron(temp,w3)
    # Put your code here #

def error(U, params):
    matrix = get_matrix(params)
    e = 0
    e_mat = U-matrix
    for i in range(2):
        for j in range(2):
            e += abs(e_mat[i][j])
    # Put your code here #
    return e

def train_parameters(U):

    epochs = 1000
    lr = 0.01

    grad = qml.grad(error, argnum=1)
    params = np.array([np.random.rand(3) * np.pi, np.random.rand(3) * np.pi, np.random.rand(3) * np.pi])

    for epoch in range(epochs):
        params -= lr * grad(U, params)
        print(params)
        # print(params)

    return params
def helper():
    qml.DiagonalQubitUnitary([1,-1,1,1],wires=[0,1])

def circuit():
    """
    Succession of gates that will generate the requested matrix.
    This function does not receive any arguments nor does it return any values.
    """

    # Put your solution here ...
    # You only have to put U3 or CNOT gates
    #qml.U3(0,0,0,wires=2)
    #qml.CZ(wires=[0,1])
    #qml.SWAP(wires=[1,2])
    #qml.PauliZ(wires=1)
    #qml.DiagonalQubitUnitary([1,-1,1,1],wires=[0,1])
    U = [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]

    solution = (
        1
        / np.sqrt(2)
        * np.array(
            [
                [1, 1],
                [1, -1]
            ]
        )
    )
    # decomp = qml.transforms.two_qubit_decomposition(solution, wires=[0, 1])
    # print('dec', decomp)
    # qml.Rot(-2.35619449, 3.14159259, 2.35619449, wires=[0])
    # qml.Rot(-1.57079633,1.57079633,0., wires=[1])
    # qml.CNOT(wires=[0, 1])
    # qml.Rot(3.14159265, 1.57079633,-1.57079633,  wires=[1]) 
    # qml.Rot(0.78539816,3.14159262,-0.78539816, wires=[0])
    qml.U3(np.pi, 0.75*np.pi, -0.75*np.pi, wires=0)
    qml.U3(np.pi/2, 0, -0.5*np.pi, wires=1)
    qml.CNOT(wires=[0,1])
    qml.U3(np.pi/2, -0.5*np.pi, np.pi, wires=1)
    qml.U3(np.pi, -0.25*np.pi, 0.25*np.pi, wires=0)
    qml.U3(0,np.pi,0,wires=2)
    qml.U3(np.pi/2,0,0,wires=2)
    qml.U3(0,0,0,wires=2)
    # qml.U3(np.pi/2,-np.pi/2,np.pi/2,wires=2)
    # qml.U3(0,np.pi/2,0,wires=2)
    # qml.U3(np.pi/2,-np.pi/2,np.pi/2,wires=2)
    #qml.Hadamard(wires=2)
    # qml.U3(0,0,0,wires=2)
    # solution = (
    #     1
    #     / np.sqrt(2)
    #     * np.array(
    #         [
    #             [1, 1, 0, 0, 0, 0, 0, 0],
    #             [1, -1, 0, 0, 0, 0, 0, 0],
    #             [0, 0, -1, -1, 0, 0, 0, 0],
    #             [0, 0, -1, 1, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 1, 1, 0, 0],
    #             [0, 0, 0, 0, 1, -1, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 1, 1],
    #             [0, 0, 0, 0, 0, 0, 1, -1],
    #         ]
    #     )
    # )
    # p = train_parameters(solution)
    # qml.Hadamard(wires=0)
    # qml.U3(p[0,0],p[0,1],p[0,2],wires=0) 
    # qml.Hadamard(wires=1)
    # qml.U3(p[1,0],p[1,1],p[1,2],wires=1) 
    # qml.Hadamard(wires=2)
    # qml.U3(p[2,0],p[2,1],p[2,2],wires=2) 

print(qml.matrix(circuit, wire_order=[0,1,2])().real.round(3))
# These functions are responsible for testing the solution.






def run(input: str) -> str:
    matrix = qml.matrix(circuit)().real

    with qml.tape.QuantumTape() as tape:
        circuit()

    names = [op.name for op in tape.operations]
    return json.dumps({"matrix": matrix.tolist(), "gates": names})

def check(user_output: str, expected_output: str) -> str:
    parsed_output = json.loads(user_output)
    matrix_user = np.array(parsed_output["matrix"])
    gates = parsed_output["gates"]

    solution = (
        1
        / np.sqrt(2)
        * np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, -1, -1, 0, 0, 0, 0],
                [0, 0, -1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, -1],
            ]
        )
    )
    # print('sol',solution-matrix_user)
    assert np.allclose(matrix_user, solution)
    assert len(set(gates)) == 2 and "U3" in gates and "CNOT" in gates


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




