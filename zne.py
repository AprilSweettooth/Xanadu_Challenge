import functools
import json
import math
from copy import copy
# import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

dev_ideal = qml.device("default.mixed", wires=2)  # no noise
dev_noisy = qml.transforms.insert(qml.DepolarizingChannel, 0.05, position="all")(
    dev_ideal
)

def U(angle):
    """A quantum function containing one parameterized gate.

    Args:
        angle (float): The phase angle for an IsingXY operator
    """
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.CNOT(wires=[0, 1])
    qml.PauliZ(1)
    qml.IsingXY(angle, [0, 1])
    qml.S(1)

@qml.qnode(dev_noisy)
def circuit(angle):
    """A quantum circuit made from the quantum function U.

    Args:
        angle (float): The phase angle for an IsingXY operator
    """
    U(angle)
    return qml.state()

@qml.tape.stop_recording()
def circuit_ops(angle):
    """A function that outputs the operations within the quantum function U.

    Args:
        angle (float): The phase angle for an IsingXY operator
    """
    with qml.tape.QuantumTape() as tape:
        U(angle)
    return tape.operations


@qml.qnode(dev_noisy)
def global_fold_circuit(angle, n, s):
    """Performs the global circuit folding procedure.

    Args:
        angle (float): The phase angle for an IsingXY operator
        n: The number of times U^\dagger U is applied
        s: The integer defining L_s ... L_d.
    """
    assert s <= len(
        circuit_ops(angle)
    ), "The value of s is upper-bounded by the number of gates in the circuit."
    # Original circuit application
    U(angle)
    # Put your code here #
    # qml.execute(circuit_ops(angle), dev_noisy, gradient_fn=None)
    # qml.transforms.fold_global(U(angle), 19/3)
    # (U^\dagger U)^n
    d = len(circuit_ops(angle))
    for _ in range(n):
        for i in range(d,0,-1):
            qml.adjoint(circuit_ops(angle)[i-1].queue())
        for i in range(0,d,1):
            circuit_ops(angle)[i].queue()
        # U(angle) 
    # L_d^\dagger ... L_s^\dagger
    for i in range(d-1, s-2, -1):
        qml.adjoint(circuit_ops(angle)[i].queue())
    # L_s ... L_d
    for i in range(s-1, len(circuit_ops(angle))):
        circuit_ops(angle)[i].queue()

    return qml.state()


def fidelity(angle, n, s):
    fid = qml.math.fidelity(global_fold_circuit(angle, n, s), circuit(angle))
    print(qml.draw(global_fold_circuit)(0.4,2,3))
    return np.round_(fid, decimals=5)


# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    angle, n, s = json.loads(test_case_input)
    fid = fidelity(angle, n, s)
    return str(fid)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    print(solution_output)
    expected_output = json.loads(expected_output)
    print(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your folded circuit isn't quite right!"


test_cases = [['[0.4, 2, 3]', '0.79209']]

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