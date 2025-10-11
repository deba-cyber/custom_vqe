"""Circuit construction for a few specific cases."""

import numpy as np
from qiskit import QuantumCircuit


def circ_exp_pauli(
    pauli: str,
    val: float,
    qc: QuantumCircuit,
) -> None | QuantumCircuit:
    """Generate QuantumCircuit for exponential of a pauli string.

    The construction of the circuit for exp(-i t P) is done where
    `P` is the pauli string.
    Change of basis is done on qubits where `X` or `Y` corresponding
    to standard unitary transformations.
    For `X` gate Haddamard gate is applied and for `Y` gate `Rz(np.pi/2) gate
    is applied.


    :param qc: quantum circuit in qiskit `QuantumCircuit` format.
    :param pauli: A string denoting pauli string.
    :param val: A float denoting parameter of the exponential of the pauli string.
    :return: A quantum circuit (qiskit `QuantumCircuit`) if `qc` is `None`.
    """
    if len(pauli) != qc.num_qubits:
        raise ValueError("Number of qubits in the circuit is incompatible with the input pauli string.")

    circ = QuantumCircuit(qc.num_qubits)

    # finding qubit locations with `X`,`Y` and `Z` gates.
    X_loc = [len(pauli) - i - 1 for i, j in enumerate(pauli) if j == "X"][::-1]
    Y_loc = [len(pauli) - i - 1 for i, j in enumerate(pauli) if j == "Y"][::-1]
    Z_loc = [len(pauli) - i - 1 for i, j in enumerate(pauli) if j == "Z"][::-1]

    # change of basis(U)
    if len(X_loc) != 0:
        circ.h(X_loc)

    if len(Y_loc) != 0:
        circ.rx(np.pi / 2, Y_loc)

    # min and max qubit number that is non-identity
    qid_min = min(min(X_loc, Y_loc, Z_loc))
    qid_max = max(max(X_loc, Y_loc, Z_loc))

    # qubit numbers in ascending order (except where identity)
    qid_cnot = np.sort(np.concatenate((X_loc, Y_loc, Z_loc)))

    assert (qid_min, qid_max) == (qid_cnot[0], qid_cnot[-1])

    # the first CNOT staircase
    for x, y in zip(qid_cnot, qid_cnot[1:]):
        circ.cx(x, y)

    # Rz gate
    circ.rz(2 * val, qid_max)

    # the second CNOT staircase
    for x, y in zip(qid_cnot[::-1], qid_cnot[::-1][1:]):
        circ.cx(x, y)

    # change of basis(U†)
    if len(X_loc) != 0:
        circ.h(X_loc)

    if len(Y_loc) != 0:
        circ.rx(-np.pi / 2, Y_loc)

    # circuit update
    if qc is None:
        return circ
    else:
        qc &= circ


def add_gates_4_measure(
    qc: QuantumCircuit,
    op: str,
) -> None | QuantumCircuit:
    """Add gates for `X` and `Y` basis measurements.

    This works for single qubit measurements, hence compatible
    with naive case and qubit-wise commuting case.

    :param qc: quantum circuit in qiskit `QuantumCircuit` format.
    :param op: A string denoting pauli string.
    :return: A quantum circuit (qiskit `QuantumCircuit`) if `qc` is `None`.
    """
    if len(op) != qc.num_qubits:
        raise ValueError("Number of qubits in the circuit is incompatible with the input pauli string.")

    circ = QuantumCircuit(qc.num_qubits)

    # finding qubit locations with `X` and `Y` gates.
    X_loc = [len(op) - i - 1 for i, j in enumerate(op) if j == "X"][::-1]
    Y_loc = [len(op) - i - 1 for i, j in enumerate(op) if j == "Y"][::-1]

    # change of basis(U)
    if len(X_loc) != 0:
        circ.h(X_loc)

    if len(Y_loc) != 0:
        circ.rx(np.pi / 2, Y_loc)

    # circuit update
    if qc is None:
        return circ
    else:
        qc &= circ
