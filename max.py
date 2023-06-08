import functools
import json
import math
# import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

@qml.qfunc_transform
def rotate_rots(tape, params):
    for op in tape.operations + tape.measurements:
        if op.name == "RX":
            if op.wires == [0]:
                qml.RX(op.parameters[0] + params[0], wires=op.wires)
            else:
                qml.RX(op.parameters[0] + params[1], wires=op.wires)
        elif op.name == "RY":
            if op.wires == [0]:
                qml.RY(op.parameters[0] + params[2], wires=op.wires)
            else:
                qml.RY(op.parameters[0] + params[3], wires=op.wires)
        elif op.name == "RZ":
            if op.wires == [0]:
                qml.RZ(op.parameters[0] + params[4], wires=op.wires)
            else:
                qml.RZ(op.parameters[0] + params[5], wires=op.wires)
        else:
            qml.apply(op)

def circuit():
    qml.RX(np.pi / 2, wires=0)
    qml.RY(np.pi / 2, wires=0)
    qml.RZ(np.pi / 2, wires=0)
    qml.RX(np.pi / 2, wires=1)
    qml.RY(np.pi / 2, wires=1)
    qml.RZ(np.pi / 2, wires=1)

def optimal_fidelity(target_params, pauli_word):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def target_circuit(target_params, pauli_word):
        """This QNode is target circuit whose effect we want to emulate"""
        # Put your code here #
        qml.PauliRot(target_params[0], pauli_word, wires=0)
        qml.CRX(target_params[1], wires=[0,1])
        qml.T(wires=0)
        qml.S(wires=1)
        return qml.state()

    @qml.qnode(dev)
    def rotated_circuit(rot_params):
        @rotate_rots(rot_params)
        def cir():
            qml.RX(np.pi / 2, wires=0)
            qml.RY(np.pi / 2, wires=0)
            qml.RZ(np.pi / 2, wires=0)
            qml.RX(np.pi / 2, wires=1)
            qml.RY(np.pi / 2, wires=1)
            qml.RZ(np.pi / 2, wires=1)
        cir()
        return qml.state()

    def train_parameters(target_params, pauli_word):
        f = []
        for i in range(5000):
            params = np.random.rand(6) * np.pi/2
            m = qml.math.fidelity(target_circuit(target_params, pauli_word), rotated_circuit(params))
            f.append(m)
        return max(f)

    return train_parameters(target_params, pauli_word)
# def optimal_fidelity():
#     target_params = [1.6,0.9]
#     pauli_word = "X"
#     """This function returns the maximum fidelity between the final state that we obtain with only
#     Pauli rotations with respect to the state we obtain with the target circuit

#     Args:
#         - target_params (list(float)): List of the two parameters in the target circuit. The first is
#         the parameter of the Pauli Rotation, the second is the parameter of the CRX gate.
#         - pauli_word: A string that is either 'X', 'Y', or 'Z', depending on the Pauli rotation
#         implemented by the target circuit.
#     Returns:
#         - (float): Maximum fidelity between the states produced by both circuits.
#     """

#     dev = qml.device("default.qubit", wires=2)

#     @qml.qnode(dev)
#     def target_circuit(target_params, pauli_word):
#         """This QNode is target circuit whose effect we want to emulate"""
#         # Put your code here #
#         qml.PauliRot(target_params[0], pauli_word, wires=0)
#         qml.CRX(target_params[1], wires=[0,1])
#         qml.T(wires=0)
#         qml.S(wires=1)
#         return qml.state()
#     # print(qml.draw(target_circuit)(target_params, pauli_word))
#     # print(qml.matrix(target_circuit)(target_params, pauli_word))
#     # U = qml.matrix(target_circuit)(target_params, pauli_word)
#     # print(U)
#     @qml.qnode(dev)
#     def rotated_circuit(rot_params):
#         """This QNode is the available circuit, with rotated parameters

#         Inputs:
#         rot_params list(float): A list containing the values of the independent rotation parameters
#         for each gate in the available circuit. The order will not matter, since you are optimizing
#         for these and will return the minimal value of a cost function (related
#         to the fidelity)
#         """
#         # Put your code here #
#         @rotate_rots(rot_params)
#         def cir():
#             qml.RX(np.pi / 2, wires=0)
#             qml.RY(np.pi / 2, wires=0)
#             qml.RZ(np.pi / 2, wires=0)
#             qml.RX(np.pi / 2, wires=1)
#             qml.RY(np.pi / 2, wires=1)
#             qml.RZ(np.pi / 2, wires=1)
#         cir()
#         # with qml.tape.QuantumTape() as tape:
#         #     circuit()
#         # print(tape.operations[1].name)
#         # rotate_rots(tape, rot_params)
#         return qml.state()

#     def get_matrix(params):
#         rx0 = np.array([[np.cos((np.pi/2+params[0])/2), -1j*np.sin((np.pi/2+params[0])/2)], [-1j*np.sin((np.pi/2+params[0])/2), np.cos((np.pi/2+params[0])/2)]])
#         ry0 = np.array([[np.cos((np.pi/2+params[1])/2), -np.sin((np.pi/2+params[1])/2)], [np.sin((np.pi/2+params[1])/2), np.cos((np.pi/2+params[1])/2)]])
#         rz0 = np.array([[np.exp(-1j*(np.pi/2+params[2])/2), 0], [0, np.exp(1j*(np.pi/2+params[2])/2)]])
#         rx1 = np.array([[np.cos((np.pi/2+params[3])/2), -1j*np.sin((np.pi/2+params[3])/2)], [-1j*np.sin((np.pi/2+params[3])/2), np.cos((np.pi/2+params[3])/2)]])
#         ry1 = np.array([[np.cos((np.pi/2+params[4])/2), -np.sin((np.pi/2+params[4])/2)], [np.sin((np.pi/2+params[4])/2), np.cos((np.pi/2+params[4])/2)]])
#         rz1 = np.array([[np.exp(-1j*(np.pi/2+params[5])/2), 0], [0, np.exp(1j*(np.pi/2+params[5])/2)]])
#         y0 = np.matmul(rx0, ry0)
#         z0 = np.matmul(y0, rz0)
#         y1 = np.matmul(rx1, ry1)
#         z1 = np.matmul(y1, rz1)
#         return  np.kron(z0, z1)

#     def error(U, params):
#         matrix = get_matrix(params)
#         e = 0
        
#         e_mat = U-np.array(matrix)
#         for i in range(4):
#             for j in range(4):
#                 e += abs(e_mat[i][j])
#         # Put your code here #
#         print(e)
#         return e
#     # Put your code here #
#     def train_parameters(U, target_params, pauli_word):

#         epochs = 1000
#         lr = 0.01
#         grad = qml.grad(error, argnum=1)
#         params = np.random.rand(6) * np.pi/2
#         for epoch in range(epochs):
#             params -= lr * grad(U, params)

#         return qml.math.fidelity(target_circuit(target_params, pauli_word), rotated_circuit(params))
#     U = np.array([[ 9.68912422e-01-4.16333634e-17j,  0.00000000e+00+0.00000000e+00j,
#   -2.47403959e-01+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j],
#  [ 0.00000000e+00+0.00000000e+00j,  4.16333634e-17+9.68912422e-01j,
#    0.00000000e+00+0.00000000e+00j,  0.00000000e+00-2.47403959e-01j],
#  [ 1.69502525e-01+1.69502525e-01j,  4.32811003e-02-4.32811003e-02j,
#    6.63825681e-01+6.63825681e-01j,  1.69502525e-01-1.69502525e-01j],
#  [ 4.32811003e-02+4.32811003e-02j, -1.69502525e-01+1.69502525e-01j,
#    1.69502525e-01+1.69502525e-01j, -6.63825681e-01+6.63825681e-01j]])
#     return train_parameters(U, target_params, pauli_word)
# These functions are responsible for testing the solution.
# print(optimal_fidelity())
def run(test_case_input: str) -> str:

    ins = json.loads(test_case_input)
    output = optimal_fidelity(*ins)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    """
    Compare solution with expected.

    Args:
            solution_output: The output from an evaluated solution. Will be
            the same type as returned.
            expected_output: The correct result for the test case.

    Raises:
            ``AssertionError`` if the solution output is incorrect in any way.
    """

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, atol=5e-2
    ), "Your calculated optimal fidelity isn't quite right."


test_cases = [['[[0.4,0.5],"Y"]', '0.9977'], ['[[1.6,0.9],"X"]', '0.9502']]

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