"""Manipulating Pauli strings with PauliArray."""

from collections import Counter
from itertools import product

import numpy as np
from qiskit.quantum_info import SparsePauliOp


class PauliArr:
    r"""Create a PauliArr object.

    Efficient representation and manipulation of set of pauli strings
    with convenient data structures, introduced in :footcite: `dion2024efficiently`.

    References
    ==========
    .. footbibliography::

    """

    # Binary encoding of the three pauli operators and the identity into two bits: x-z decomposition.
    pauli_xz_dict = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}

    # allowed phase factors.
    phase_arr = ["-", "i", "-i"]

    # pauli operators in matrix form (numpy array).
    pauli_I = np.eye(2)
    pauli_X = np.array([[0, 1], [1, 0]])
    pauli_Y = np.array([[0, -1j], [1j, 0]])
    pauli_Z = np.array([[1, 0], [0, -1]])

    def __init__(
        self,
        paulis: list[str] | np.ndarray[str],
        coeffs: list[complex] | np.ndarray[complex] | None,
        xz_style: str = "little-endian",
    ) -> None:
        """Create PauliStr object.

        :param paulis: A list (numpy array) of strings.
        :param coeffs: A list (numpy array) of complex coefficients for a
                                pauli hamiltonian. Input can also be `None`.
        :param xz_style: Convention for x-z representation (default `little-endian`).
        """
        # Allowed characters in the string for a valid pauli string.
        allowed_chars = "+-iIXYZ"

        if xz_style not in ("little-endian", "big-endian"):
            raise ValueError("The strings must be either `little-endian` or `big-endian`.")

        # checking if valid pauli string for all strings.
        for pauli in paulis:
            if not (all(char in allowed_chars for char in pauli)):
                raise ValueError("Not a valid Pauli string.")

        if coeffs is None:
            # all coefficients are initialized to 1.
            coeffs = np.ones((1, len(paulis)), dtype=complex)

        else:
            assert len(paulis) == len(coeffs)

            # calculating number of qubits.

            selected_keys = list(PauliArr.pauli_xz_dict.values())

            counts = dict(Counter(paulis[0]))

            self._N = sum([counts[key] if key in list(counts.keys()) else 0 for key in selected_keys])

            # number of qubits initialized based on the first pauli string in the list.
            # checking if all other strings have the same number of qubits.

            for i in range(1, len(paulis)):
                counts = dict(Counter(paulis[i]))

                qbit_cnt = sum([counts[key] if key in list(counts.keys()) else 0 for key in selected_keys])

                assert qbit_cnt == self._N

            # tracking -ve signs for tableau representation.
            self._loc_ve_sgn = np.array([0 if pauli[0] != "-" else 1 for pauli in paulis])

            # --------------- #
            # dictionary form #

            self._paulidict = {}

            for x, y in zip(paulis, coeffs):
                self._paulidict[x] = y

            # --------------- #
            # symplectic form.
            self._symp_mat = np.zeros((len(paulis), 2 * self._N), dtype=int)

            # list of pauli strings (with the phases)
            paulis = list(self._paulidict.keys())

            # removing the phases.
            paulis = ["".join(char for char in pauli if char not in "+-i") for pauli in paulis]

            # only the pauli strings w/o the phases.
            self._paulis = paulis

            if xz_style == "big-endian":
                paulis = [ele[::-1] for ele in paulis]

            for row, pauli in enumerate(paulis):
                pauli_list = np.array(list(pauli), dtype=str)
                Xloc = (pauli_list == "X").astype(int)
                Yloc = (pauli_list == "Y").astype(int)
                Zloc = (pauli_list == "Z").astype(int)

                self._symp_mat[row][: self._N] += Xloc
                self._symp_mat[row][self._N :] += Zloc
                self._symp_mat[row][: self._N] += Yloc
                self._symp_mat[row][self._N :] += Yloc

            # --------------- #
            # tableau representation.
            self._tableau = np.column_stack((self._symp_mat, self._loc_ve_sgn))

            # --------------- #
            # X- and Z- blocks in matrix form (numpy array)

            self._X_block = self._symp_mat[:, : self._N]
            self._Z_block = self._symp_mat[:, self._N :]

    @classmethod
    def remove_phase_pauli(
        cls, pauli: list[str] | np.ndarray[str], coeffs: list[complex] | np.ndarray[complex] | None
    ) -> "PauliArr":
        r"""Remove phase factors i.e. `+`, `-`,`i` from a list of pauli strings.

        :param pauli: A string (pauli) with possible phase factors.
        :return: A PauliArr object.
        """
        pauli = ["".join(char for char in elem if char not in "+-i") for elem in pauli]

        return cls(pauli, coeffs)

    @staticmethod
    def pauli_2_SparsePauliOp(
        paulis: list[str] | np.ndarray[str], coeffs: list[complex] | np.ndarray[complex] | None
    ) -> SparsePauliOp:
        """Convert array of pauli strings as string form to `SparsePauliOp`.

        :param paulis: A list (or numpy array) of strings.
        :return: qiskit SparsePauliOp.
        """
        if coeffs is None:
            coeffs = np.ones(len(paulis))

        return SparsePauliOp(paulis, coeffs)

    def pauli_gr(self) -> np.ndarray[str]:
        """Generate all elements of N-qubit pauli operators.

        The pauli strings are allowed with the phase factors.
        For N-qubit, total number of elements is 4**(N+1).

        :return: A list (numpy array) of pauli strings.
        """
        pauli_arr = list(PauliArr.pauli_xz_dict.values())

        all_combs = [combination for combination in product(range(4), repeat=self._N)]

        assert len(all_combs) == 4 ** (self._N)

        str_set = []

        for combination in all_combs:
            curstr1 = "".join(pauli_arr[letter] for letter in combination)
            curstr2 = PauliArr.phase_arr[0] + curstr1
            curstr3 = PauliArr.phase_arr[1] + curstr1
            curstr4 = PauliArr.phase_arr[2] + curstr1

            str_set.extend([curstr1, curstr2, curstr3, curstr4])

        return np.array(str_set)

    @staticmethod
    def check_commute_paulis(pauli_1: str, pauli_2: str, qwc: str = False) -> bool | None:
        """Check if two Pauli strings are mutually commuting :footcite:`gokhale2019minimizing`.

        It can execute both qubitwise commuting (qwc) and general commuting (gc) based on user
        input.

        References
        ==========
        .. footbibliography::

        :raises ValueError: If Pauli strings have different lengths.
        :param pauli_1: First Pauli string.
        :param pauli_2: Second Pauli string.
        :param qwc: True / False. If checking is done based on qubit-wise / general commuting.
        :return: True / False. If the two Pauli strings are mutually commuting or not.

        """
        if len(pauli_1) != len(pauli_2):
            raise ValueError("Length of the two strings must be equal!")

        k_count = 0  # counter for non-commuting cases.
        for i, j in zip(pauli_1, pauli_2):
            if "I" not in (i, j):
                if i != j:
                    k_count += 1

        if qwc:
            return True if k_count == 0 else False
        elif k_count != 0:
            return True if k_count % 2 == 0 else False
