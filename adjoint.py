import functools
import json
import math
# import pandas as pd
import pennylane as qml
import pennylane.numpy as np
import scipy

def generator_info(operator):
    """Provides the generator of a given operator.

    Args:
        operator (qml.ops): A PennyLane operator

    Returns:
        (qml.ops): The generator of the operator.
        (float): The coefficient of the generator.
    """
    gen = qml.generator(operator, format="observable")
    return gen.ops[0], gen.coeffs[0]


def derivative(op_order, params, diff_idx, wires, measured_wire):
    """A function that calculates the derivative of a circuit w.r.t. one parameter.

    NOTE: you cannot use qml.grad in this function.

    Args:
        op_order (list(int)):
            This is a list of integers that defines the circuit in question.
            The entries of this list correspond to dictionary keys to op_dict.
            For example, [1,0,2] means that the circuit in question contains
            an RY gate, an RX gate, and an RZ gate in that order.

        params (np.array(float)):
            The parameters that define the gates in the circuit. In this case,
            they're all rotation angles.

        diff_idx (int):
            The index of the gate in the circuit that is to be differentiated
            with respect to. For instance, if diff_idx = 2, then the derivative
            of the third gate in the circuit will be calculated.

        wires (list(int)):
            A list of wires that each gate in the circuit will be applied to.

        measured_wire (int):
            The expectation value that needs to be calculated is with respect
            to the Pauli Z operator. measured_wire defines what wire we're
            measuring on.

    Returns:
        float: The derivative evaluated at the given parameters.
    """
    op_dict = {0: qml.RX, 1: qml.RY, 2: qml.RZ}
    dev = qml.device("default.qubit", wires=2)

    obs = qml.PauliZ(measured_wire)
    operator = op_dict[op_order[diff_idx]](params[diff_idx], wires[diff_idx])
    gen, coeff = generator_info(operator)
    @qml.qnode(dev)
    # def circuit_bra1():

    #     # Put your code here #
    #     # for i in range(len(op_order)-1, -1, -1):
    #     #     qml.adjoint(op_dict[op_order[i]](params[i], wires[i]))
    #     #     if i == len(op_order)-1-diff_idx:
    #     #         qml.adjoint(gen)
    #     for i in range(len(op_order)-1):
    #         if i == diff_idx:
    #             qml.adjoint(gen)
    #         qml.adjoint(op_dict[op_order[i]](params[i], wires[i]))
    #     return qml.state()
    # @qml.qnode(dev)
    # def circuit_ket1():

    #     # Put your code here #
    #     for i in range(len(op_order)):
    #         op_dict[op_order[i]](params[i], wires[i]) 
    #     qml.apply(obs)
    #     return qml.state()

    # @qml.qnode(dev)
    # def circuit_bra2():

    #     # Put your code here #
    #     # qml.apply(obs)
    #     for i in range(len(op_order)-1):
    #     # for i in range(len(op_order)-1, -1, -1):
    #         qml.adjoint(op_dict[op_order[i]](params[i], wires[i]))
    #     qml.apply(obs)
    #     return qml.state()

    # @qml.qnode(dev)
    # def circuit_ket2():

    #     # Put your code here #
    #     for i in range(len(op_order)):
    #         op_dict[op_order[i]](params[i], wires[i]) 
    #         if i == diff_idx:
    #             qml.apply(gen)
    #     return qml.state()
    def circuit_bra1():
        for i in range(diff_idx+1):
            op_dict[op_order[i]](params[i], wires[i])  
        qml.apply(gen) 
        for i in range(diff_idx+1, len(op_order)):
            op_dict[op_order[i]](params[i], wires[i]) 
        qml.apply(obs)
        for i in range(len(op_order)-1, -1, -1):
            qml.adjoint(op_dict[op_order[i]](params[i], wires[i])) 
        return qml.state()
    @qml.qnode(dev)
    def circuit_ket1():

        # Put your code here #
        for i in range(len(op_order)):
            op_dict[op_order[i]](params[i], wires[i]) 
        qml.apply(obs)
        for i in range(len(op_order)-1, -1, -1):
            qml.adjoint(op_dict[op_order[i]](params[i], wires[i]))
            if i == len(op_order)-1-diff_idx:
                qml.adjoint(gen)
        return qml.state()

    @qml.qnode(dev)
    def circuit_bra2():
        return qml.state()

    @qml.qnode(dev)
    def circuit_ket2():

        # Put your code here #
        for i in range(len(op_order)):
            op_dict[op_order[i]](params[i], wires[i]) 
            if i == diff_idx:
                qml.apply(gen)
        qml.apply(obs)
        for i in range(len(op_order)-1, -1, -1):
            qml.adjoint(op_dict[op_order[i]](params[i], wires[i]))
        return qml.state()
    print(qml.draw(circuit_bra1)())
    # print(qml.draw(circuit_bra2)())
    # print(qml.draw(circuit_ket1)())
    # print(qml.draw(circuit_ket2)())
    print((-1j*qml.matrix(circuit_bra1)()).real)
    bra1 = circuit_bra1()
    ket1 = circuit_ket1()
    bra2 = circuit_bra2()
    ket2 = circuit_ket2()
    # print(np.matmul(bra1, ket1)*1j*coeff)
    # print(np.matmul(bra2, ket2)*1j*coeff)
    # print(qml.matrix(circuit_bra1)())
    # m = np.matmul(np.array(qml.matrix(circuit_bra1)().tolist()), np.array(qml.matrix(circuit_ket1)().tolist()))
    # n = np.matmul(np.array(qml.matrix(circuit_bra2)().tolist()), np.array(qml.matrix(circuit_ket2)().tolist()))
    # print(1j*coeff*(n-m))
    # print(coeff)
    # print(n)
    # print(m)
    # return  (1j * coeff * ((bra2[0]*ket2[0])+(bra2[1]+ket2[2]) - (bra1[0]*ket1[0])+(bra1[1]+ket1[2]))).real
    # return  (1j * coeff * (np.matmul(bra2, ket2) - np.matmul(bra1, ket1)))
    return (-1j*qml.matrix(circuit_bra1)()).real[0,0]
print(derivative([1,0,2,1,0,1], [1.23, 4.56, 7.89, 1.23, 4.56, 7.89], 0, [1, 0, 1, 1, 1, 0], 1))

# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    op_order, params, diff_idx, wires, measured_wire = json.loads(test_case_input)
    params = np.array(params, requires_grad=True)
    der = derivative(op_order, params, diff_idx, wires, measured_wire)
    return str(der)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your derivative isn't quite right!"


test_cases = [['[[1,0,2,1,0,1], [1.23, 4.56, 7.89, 1.23, 4.56, 7.89], 0, [1, 0, 1, 1, 1, 0], 1]', '-0.2840528']]

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