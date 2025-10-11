"""Unitary Coupled-Cluster ansatz."""

from itertools import combinations

import numpy as np
from qiskit_nature.second_q.operators import FermionicOp


class Fermionic_ex_UCC:
    r"""Create a UCC Fermionic Excitation object."""

    def __init__(
        self,
        id_spin_up_occ: list | np.ndarray,
        id_spin_down_occ: list | np.ndarray,
        id_spin_up_vir: list | np.ndarray,
        id_spin_down_vir: list | np.ndarray,
        chrdict: dict | str = "nosym",
    ) -> None:
        r"""Initialize the variables for UCC Fermionic Excitation class.

        Overview
        ========

        In general, the orbitals are Hartree-Fock orbitals
        corresponding to the initial configuration (ref state) wrt which
        excitations are implemented.
        There are several conventions for representing the state with some of the
        spin-orbitals occupied.
        First M/2 orbitals are spin-up , followed by next M/2 orbitals are spin-down
        for a spin-paired system.
        The most common is spin up-down for each spatial orbital usually energy ordered.

        :param id_spin_up_occ: A list (numpy array) of orbital indices for occupied up-spin electrons.
        :param id_spin_down_occ: A list (numpy array) of orbital indices for occupied down-spin electrons.
        :param id_spin_up_vir: A list (numpy array) of orbital indices for virtual up-spin electrons.
        :param id_spin_down_vir: A list (numpy array) of orbital indices for virtual down-spin electrons.
        :param chrdict: A dictionary with irreducible representations and
                array of characters for the point group symmetry of the system,
                Or a string (`nosym`) if system has no symmetry or symmetry is not invoked.

        """
        self.idarr_spinup_occ = id_spin_up_occ
        self.idarr_spindown_occ = id_spin_down_occ
        self.idarr_spinup_vir = id_spin_up_vir
        self.idarr_spindown_vir = id_spin_down_vir
        self.chrdict = chrdict

        self.idarr_all_occ = np.concatenate((self.idarr_spinup_occ, self.idarr_spindown_occ))
        self.idarr_all_vir = np.concatenate((self.idarr_spinup_vir, self.idarr_spindown_vir))

        self.idarr_all_occ.sort()
        self.idarr_all_vir.sort()

        self.N_spin_occ = len(self.idarr_all_occ)
        self.N_spin_vir = len(self.idarr_all_vir)
        self.N_spin_tot = self.N_spin_occ + self.N_spin_vir

    @staticmethod
    def _pdt_irreps(chr_table: dict) -> dict:
        """Compute tensor products of irreducible representations corresponding to point group symmetry.

        :raises ValueError: If the provided system has degeneracy.
        :param chr_table: A dictionary where each key is a str
                corresponding to the irreducible representation and
                each value corresponding to the key is the array of
                characters.
        :return: A dictionary with keys (product of irreducible representations)
                and values (a str corresponding to the irreducible representation).

        """
        # irreducible representations.
        keylist = list(chr_table.keys())
        # array of corresponding characters.
        valuelist = list(chr_table.values())

        for arr in valuelist:
            for ele in arr:
                if ele == 0 or ele > 1:
                    raise ValueError("Not handling degenerate cases here!")

        pdtkeylist = list(combinations(keylist, 2))

        tmp_pdtvaluelist = list(combinations(valuelist, 2))

        pdtvaluelist = [elem[0] * elem[1] for ele in tmp_pdtvaluelist for elem in zip(ele[0], ele[1])]
        pdtvaluelist = list(np.split(np.array(pdtvaluelist), len(tmp_pdtvaluelist)))
        pdtvaluelist = [list(ele) for ele in pdtvaluelist]

        table_keylist = [keylist[valuelist.index(ele)] for ele in pdtvaluelist]

        tp_dict = {ele[0]: ele[1] for ele in zip(pdtkeylist, table_keylist)}

        return tp_dict

    def _get_config_singles(self, N_occ_to_ex: int, N_vir_to_ex: int, spin_type: str = "conserved") -> np.ndarray:
        """Calculate Fermionic singles excitations.

        The arguments essentially define the active space
        with controls that may or maynot be spin-conserved.

        :raises ValueError: If number of electrons to excite is more than number of
                            electrons in the system, or enough number of virtual orbitals
                            are not available.
        :param N_occ_to_ex: An integer denoting number of occupied orbitals from which
                            excitations to be generated.
        :param N_vir_to_ex: An integer denoting number of virtual orbitals in which
                            excitations to be generated.
        :param spin_type : A string denoting changes in spin i.e. excitation is
                           spin-conserved or not (default is `conserved`).
        :return: A 2D numpy array (each element with 2 indices corresponding
                 to the creation/annhilation operators)
        """
        # 2D array with each element (array) containing indices of creation/annihilation spin-orbitals
        op_dagger_op = []

        # virtual orbital indices in the active space
        idarr_vir_to_ex_all_spin = self.idarr_all_vir[:N_vir_to_ex][::-1]

        # virtual up-spin orbital indices in the active space
        idarr_vir_to_ex_up_spin = [ele for ele in idarr_vir_to_ex_all_spin if ele in self.idarr_spinup_vir]
        # virtual down-spin orbitals in the active space
        idarr_vir_to_ex_down_spin = [ele for ele in idarr_vir_to_ex_all_spin if ele in self.idarr_spindown_vir]

        # - sanity checks -#
        # -----------------#
        if N_occ_to_ex > self.N_spin_occ:
            raise ValueError(
                "Number of electrons to excite for singly-excite configurations\
                    can not be more than number of electrons in the system!"
            )

        if N_vir_to_ex > self.N_spin_vir:
            raise ValueError("Not enough virtual orbitals available, please check your input!")

        # -----------------#
        match spin_type:
            case "conserved":
                for i in range(N_occ_to_ex):
                    tmp = self.idarr_all_occ[-1 - i]
                    if tmp in self.idarr_spinup_occ:
                        arr = [ele for ele in idarr_vir_to_ex_up_spin]
                        ex = list(map(lambda x: [x, tmp], arr))
                        op_dagger_op.extend(ex)
                    elif tmp in self.idarr_spindown_occ:
                        arr = [ele for ele in idarr_vir_to_ex_down_spin]
                        ex = list(map(lambda x: [x, tmp], arr))
                        op_dagger_op.extend(ex)

                return np.array(op_dagger_op)

            case "non-conserved":
                for i in range(N_occ_to_ex):
                    tmp = self.idarr_all_occ[-1 - i]
                    if tmp in self.idarr_spinup_occ:
                        arr = [ele for ele in idarr_vir_to_ex_down_spin]
                        ex = list(map(lambda x: [x, tmp], arr))
                        op_dagger_op.extend(ex)
                    elif tmp in self.idarr_spindown_occ:
                        arr = [ele for ele in idarr_vir_to_ex_up_spin]
                        ex = list(map(lambda x: [x, tmp], arr))
                        op_dagger_op.extend(ex)

                return np.array(op_dagger_op)

            case "all":
                arr = [ele for ele in idarr_vir_to_ex_all_spin]
                for i in range(N_occ_to_ex):
                    tmp = self.idarr_all_occ[-1 - i]
                    for ele in arr:
                        op_dagger_op.append([ele, tmp])

                return np.array(op_dagger_op)

    def _get_config_custom_singles(self, exarr: list | np.ndarray) -> np.ndarray:
        """Compute spin-conserved excitations from the excitation list provided in input.

        :raises ValueError: If any of the orbital indices (occupied or virtual) is incorrect
                            for the system.
        :param exarr: A 2D list (2D numpy array) of user provided excitation list (for generating
                      customized excitations).
        :return: A numpy array of excitation list.
        """
        op_dagger_op = []

        for ex in exarr:
            if ex[0] not in self.idarr_all_vir or ex[1] not in self.idarr_all_occ:
                raise ValueError("Either incorrect virtual/occupied orbital number or both!")
            elif ex[0] in self.idarr_spinup_vir and ex[1] in self.idarr_spinup_occ:
                op_dagger_op.append([ex[0], ex[1]])
            elif ex[0] in self.idarr_spindown_vir and ex[1] in self.idarr_spindown_occ:
                op_dagger_op.append([ex[0], ex[1]])
            else:
                continue

        return np.array(op_dagger_op)

    def _get_config_doubles(self, N_occ_to_ex: int, N_vir_to_ex: int, spin_type: str = "conserved") -> np.ndarray:
        """Calculate Fermionic doubles excitations.

        The arguments essentially define active space with controls
        that may or maynot be spin-conserved.

        :raises ValueError: If number of electrons to excite is less than 2,
                            or number of virtual orbitals is less than 2, or number of occupied
                            electrons in the active space can't be greater than total number
                            of electrons in the system.
        :param N_occ_to_ex: An integer denoting number of occupied orbitals from which excitations
                            are generated.
        :param N_vir_to_ex: An integer denoting number of virtual orbitals to which excitations
                            to be generated.
        :param spin_type: A string denoting changes in spin i.e. excitation is spin-conserved or not
                          (default is conserved).
        :return: A 2D numpy array (each element with 2 indices corresponding
                 to the creation/annhilation operators).
        """
        # sanity checks
        if N_occ_to_ex < 2:
            raise ValueError("Number of electrons to excite can't be less than 2 for double excitation!")

        if N_vir_to_ex < 2:
            raise ValueError("At least two virtual orbitals needed for double excitation!")

        if N_occ_to_ex > self.N_spin_occ:
            raise ValueError(
                "Number of occupied electrons in the active space can't be greater\
                than total number of electrons in the system!"
            )

        # new variables for the active space #

        # index array for virtual orbitals in the active space
        N_vir_to_ex_all_spin = self.idarr_all_vir[:N_vir_to_ex][::-1]

        # index array for occupied orbitals in the active space
        N_occ_to_ex_all_spin = self.idarr_all_occ[-N_occ_to_ex:][::-1]

        # index array of virtual up-spin orbitals in the active space
        idarr_vir_to_ex_up_spin = [ele for ele in N_vir_to_ex_all_spin if ele in self.idarr_spinup_vir]

        # index array of virtual down-spin orbitals in the active space
        idarr_vir_to_ex_down_spin = [ele for ele in N_vir_to_ex_all_spin if ele in self.idarr_spindown_vir]

        # index array of occupied up-spin orbitals in the active space
        idarr_occ_to_ex_up_spin = [ele for ele in N_occ_to_ex_all_spin if ele in self.idarr_spinup_occ]

        # index array of occupied down-spin orbitals in the active space
        idarr_occ_to_ex_down_spin = [ele for ele in N_occ_to_ex_all_spin if ele in self.idarr_spindown_occ]

        virop_up_up = list(combinations(idarr_vir_to_ex_up_spin, 2))
        virop_down_down = list(combinations(idarr_vir_to_ex_down_spin, 2))

        virop_up_down = [
            sorted([x, y], reverse=True) for x in idarr_vir_to_ex_up_spin for y in idarr_vir_to_ex_down_spin
        ]

        assert len(virop_up_down) == len(idarr_vir_to_ex_up_spin) * len(idarr_vir_to_ex_down_spin)

        occop_up_up = list(combinations(idarr_occ_to_ex_up_spin, 2))
        occop_down_down = list(combinations(idarr_occ_to_ex_down_spin, 2))

        occop_up_down = [
            sorted([x, y], reverse=True) for x in idarr_occ_to_ex_up_spin for y in idarr_occ_to_ex_down_spin
        ]

        assert len(occop_up_down) == len(idarr_occ_to_ex_up_spin) * len(idarr_occ_to_ex_down_spin)

        match spin_type:
            case "conserved":
                op_upup = [[y[0], y[1], x[0], x[1]] for x in occop_up_up for y in virop_up_up]
                op_downdown = [[y[0], y[1], x[0], x[1]] for x in occop_down_down for y in virop_down_down]
                op_updown = [[y[0], y[1], x[0], x[1]] for x in occop_up_down for y in virop_up_down]

                op_conserved = op_upup + op_downdown + op_updown

                return np.array(op_conserved)

            case "non-conserved":
                op_vir_upup_occ_updown = [[y[0], y[1], x[0], x[1]] for x in occop_up_down for y in virop_up_up]
                op_vir_downdown_occ_updown = [[y[0], y[1], x[0], x[1]] for x in occop_up_down for y in virop_down_down]
                op_vir_upup_occ_downdown = [[y[0], y[1], x[0], x[1]] for x in occop_down_down for y in virop_up_up]
                op_vir_updown_occ_downdown = [[y[0], y[1], x[0], x[1]] for x in occop_down_down for y in virop_up_down]
                op_vir_downdown_occ_upup = [[y[0], y[1], x[0], x[1]] for x in occop_up_up for y in virop_down_down]
                op_vir_updown_occ_upup = [[y[0], y[1], x[0], x[1]] for x in occop_up_up for y in virop_up_down]

                op_nonconserved = (
                    op_vir_upup_occ_updown
                    + op_vir_downdown_occ_updown
                    + op_vir_upup_occ_downdown
                    + op_vir_updown_occ_downdown
                    + op_vir_downdown_occ_upup
                    + op_vir_updown_occ_upup
                )

                return np.array(op_nonconserved)

            case "all":
                virop_all = list(combinations(N_vir_to_ex_all_spin, 2))
                occop_all = list(combinations(N_occ_to_ex_all_spin, 2))

                op_all = [[y[0], y[1], x[0], x[1]] for x in occop_all for y in virop_all]

                return np.array(op_all)

    def _get_config_custom_doubles(self, N_occ_to_ex: int, N_vir_to_ex: int, exarr: list | np.ndarray) -> np.ndarray:
        r"""Compute spin-conserved excitations from the excitation list provided in input.

        :param N_occ_to_ex: A 2D numpy array of user provided excitation (a_i^{\dag} a_j^{\dag} a_{k} a{l})
                            list (for generating customized excitations).
        :return: A numpy array excitation list.
        """
        # index array for virtual orbitals in the active space
        N_vir_to_ex_all_spin = self.idarr_all_vir[:N_vir_to_ex][::-1]

        # index array for occupied orbitals in the active space
        N_occ_to_ex_all_spin = self.idarr_all_occ[-N_occ_to_ex:][::-1]

        # index array of virtual up-spin orbitals in the active space
        idarr_vir_to_ex_up_spin = [ele for ele in N_vir_to_ex_all_spin if ele in self.idarr_spinup_vir]

        # index array of virtual down-spin orbitals in the active space
        idarr_vir_to_ex_down_spin = [ele for ele in N_vir_to_ex_all_spin if ele in self.idarr_spindown_vir]

        # index array of occupied up-spin orbitals in the active space
        idarr_occ_to_ex_up_spin = [ele for ele in N_occ_to_ex_all_spin if ele in self.idarr_spinup_occ]

        # index array of occupied down-spin orbitals in the active space
        idarr_occ_to_ex_down_spin = [ele for ele in N_occ_to_ex_all_spin if ele in self.idarr_spindown_occ]

        virop_up_up = list(combinations(idarr_vir_to_ex_up_spin, 2))
        virop_down_down = list(combinations(idarr_vir_to_ex_down_spin, 2))
        virop_up_down = [
            sorted([x, y], reverse=True) for x in idarr_vir_to_ex_up_spin for y in idarr_vir_to_ex_down_spin
        ]

        occop_up_up = list(combinations(idarr_occ_to_ex_up_spin, 2))
        occop_down_down = list(combinations(idarr_occ_to_ex_down_spin, 2))
        occop_up_down = [
            sorted([x, y], reverse=True) for x in idarr_occ_to_ex_up_spin for y in idarr_occ_to_ex_down_spin
        ]

        op = []

        for ex in exarr:
            if (ex[0], ex[1]) in virop_up_up and (ex[2], ex[3]) in occop_up_up:
                op.append(ex)
            elif (ex[0], ex[1]) in virop_down_down and (ex[2], ex[3]) in occop_down_down:
                op.append(ex)
            elif [ex[0], ex[1]] in virop_up_down and [ex[2], ex[3]] in occop_up_down:
                op.append(ex)

        return np.array(op)

    def _get_config_generalized_exc(
        self,
        N_occ_to_ex: int,
        N_vir_to_ex: int,
    ) -> tuple[np.ndarray]:
        """Compute singles and doubles excitation lists with generalized excitations :footcite: `lee2018generalized`.

        Creates all configurations with spin-conserved and spin non-conserved excitations.

        :param N_occ_to_ex: An integer denoting number of occupied orbitals.
        :param N_vir_to_ex: An integer denoting number of virtual orbitals.
        :return: A tuple of numpy arrays of singles and doubles excitations list.
        """
        # index array for virtual orbitals in the active space
        N_vir_to_ex_all_spin = self.idarr_all_vir[:N_vir_to_ex][::-1]

        # index array for occupied orbitals in the active space
        N_occ_to_ex_all_spin = self.idarr_all_occ[-N_occ_to_ex:][::-1]

        all_orb = np.sort(np.concatenate((N_occ_to_ex_all_spin, N_vir_to_ex_all_spin)))[::-1]
        singles_all_config = list(combinations(all_orb, 2))
        doubles_all_config = list(combinations(all_orb, 4))

        return (singles_all_config, doubles_all_config)

    def _get_config_pCCD(
        self, N_occ_to_ex: int, N_vir_to_ex: int, config_type: tuple[str] = ("down", "up")
    ) -> np.ndarray:
        """Compute paired coupled-cluster doubles excitations.

        It discards configurations that are not paired from all excitations generated.

        Caution
        =======
        Assumes electrons are arranged up-down manner in spin-orbitals.

        :param N_occ_to_ex: An integer denoting number of occupied orbitals from which
                            excitations to be generated.
        :param N_vir_to_ex: An integer denoting number of virtual orbitals to which
                            excitations are to be generated.
        :return: A numpy array with configurations (conforming to one spatial orbital-> another spatial orbital).

        """
        # index array for virtual orbitals in the active space
        N_vir_to_ex_all_spin = self.idarr_all_vir[:N_vir_to_ex][::-1]

        # index array for occupied orbitals in the active space
        N_occ_to_ex_all_spin = self.idarr_all_occ[-N_occ_to_ex:][::-1]

        # index array of virtual up-spin orbitals in the active space
        idarr_vir_to_ex_up_spin = [ele for ele in N_vir_to_ex_all_spin if ele in self.idarr_spinup_vir]

        # index array of virtual down-spin orbitals in the active space
        idarr_vir_to_ex_down_spin = [ele for ele in N_vir_to_ex_all_spin if ele in self.idarr_spindown_vir]

        # index array of occupied up-spin orbitals in the active space
        idarr_occ_to_ex_up_spin = [ele for ele in N_occ_to_ex_all_spin if ele in self.idarr_spinup_occ]

        # index array of occupied down-spin orbitals in the active space
        idarr_occ_to_ex_down_spin = [ele for ele in N_occ_to_ex_all_spin if ele in self.idarr_spindown_occ]

        match config_type:
            case ("down", "up"):
                spatial_configs_vir = [
                    [vir_down + 1, vir_down]
                    for vir_down in idarr_vir_to_ex_down_spin
                    if vir_down + 1 in idarr_vir_to_ex_up_spin
                ]
                spatial_configs_occ = [
                    [occ_down + 1, occ_down]
                    for occ_down in idarr_occ_to_ex_down_spin
                    if occ_down + 1 in idarr_occ_to_ex_up_spin
                ]
            case ("up", "down"):
                spatial_configs_vir = [
                    [vir_up + 1, vir_up]
                    for vir_up in idarr_vir_to_ex_up_spin
                    if vir_up + 1 in idarr_vir_to_ex_down_spin
                ]
                spatial_configs_occ = [
                    [occ_up + 1, occ_up]
                    for occ_up in idarr_occ_to_ex_up_spin
                    if occ_up + 1 in idarr_occ_to_ex_down_spin
                ]
            case _:
                raise ValueError("Invalid configuration type")

        # all the double spin-conserved excitations in the active space.
        double_ex_config = self._get_config_doubles(N_occ_to_ex, N_vir_to_ex)

        double_ex_pair = []

        for ele in double_ex_config:
            if [ele[0], ele[1]] in spatial_configs_vir and [ele[2], ele[3]] in spatial_configs_occ:
                double_ex_pair.append(list(ele))

        return np.array(double_ex_pair)

    def _restrict_orbital_symmetry(self, N_occ_to_ex: int, N_vir_to_ex: int, orb_sym: dict) -> tuple[list]:
        """Calculate restricted excitations after invoking orbital symmetry of the molecular system.

        Orbital symmetry invoked on top of configurations generated corresponding to spin-conserved cases.

        :param N_occ_to_ex: An integer denoting number of occupied orbitals from which excitations
                            are generated.
        :param N_vir_to_ex: An integer denoting number of virtual orbitals to which excitations are
                            to be generated.
        :param orb_sym: A dictionary where keys are orbital indices (0-based indexing) and
                        values are irreducible representations (data type: str)
        :return: A tuple of lists with configurations (singles and doubles) after invoking
                 orbital symmetry.
        """
        if self.chrdict == "nosym":
            raise ValueError("Orbital symmetry restriction not possible for non-symmetric system")

        # all possible multiplications for symmetry operations.
        symdict = self._pdt_irreps(self.chrdict)

        sym_order_map = {ele: i for i, ele in enumerate(self.chrdict.keys())}

        # generating spin-conserved singles and doubles excitations.
        single_ex_config = self._get_config_singles(N_occ_to_ex, N_vir_to_ex)
        double_ex_config = self._get_config_doubles(N_occ_to_ex, N_vir_to_ex)

        single_ex_config_f = [list(ele) for ele in single_ex_config if orb_sym[ele[0]] == orb_sym[ele[1]]]

        double_ex_config_f = []

        for ele in double_ex_config:
            vir_sym1 = orb_sym[ele[0]]
            vir_sym2 = orb_sym[ele[1]]
            occ_sym1 = orb_sym[ele[2]]
            occ_sym2 = orb_sym[ele[3]]

            filtered_occ = [item for item in [occ_sym1, occ_sym2] if item in sym_order_map]
            filtered_vir = [item for item in [vir_sym1, vir_sym2] if item in sym_order_map]

            sorted_occ = sorted(filtered_occ, key=lambda x: sym_order_map[x])
            sorted_vir = sorted(filtered_vir, key=lambda x: sym_order_map[x])

            if sorted_occ[0] != sorted_occ[1] and sorted_vir[0] != sorted_vir[1]:
                if symdict[(sorted_occ[0], sorted_occ[1])] == symdict[(sorted_vir[0], sorted_vir[1])]:
                    double_ex_config_f.append(list(ele))

            elif sorted_occ[0] == sorted_occ[1] and sorted_vir[0] == sorted_vir[1]:
                double_ex_config_f.append(list(ele))

        return (np.array(single_ex_config_f), np.array(double_ex_config_f))

    @staticmethod
    def _get_op_hc(Oplist: list | np.ndarray) -> tuple:
        """Compute (Op - h.c.) of an operator i.e. hermitian conjugate of an operator subtracted from an operator.

        :param Oplist: A 2D list (2D numpy array) where each element denoting an excitation operator.
        :return: A tuple of lists with operator and coefficients (Operator - h.c.).
        """
        Op_tot, coeff = [], []

        for Op in Oplist:
            Op_tot.append(Op)
            Op_tot.append(Op[::-1])
            coeff.append([1, -1])

        return (np.array(Op_tot), np.array(coeff))

    @staticmethod
    def _get_qiskit_Fermionic_Op(ExcitationOp: list | np.ndarray, Coeffarr: list | np.ndarray) -> FermionicOp:
        """Compute qiskit Type FermionicOp for second-quantized operator initially expressed as in this file.

        Assumption
        ==========
        Equal number of creation and annihilation operators normal-ordered.

        :param ExcitationOp: A 2D array (list or numpy array) of excitation operator (indices).
        :param Coeffzrr: A list (numpy array) 1D array of coefficients.
        :return: Qiskit FermionicOp (Fermionic operator in Qiskit format).
        """
        opdict = {}

        keylist = []

        for ele in ExcitationOp:
            creationOp_arr = ele[: int(len(ele) / 2)]
            annihilationOp_arr = ele[int(len(ele) / 2) :]
            str1 = "".join("+_" + str(ele) + " " for ele in creationOp_arr)
            str2 = "".join("-_" + str(ele) + " " for ele in annihilationOp_arr[:-1])
            str2 += "-_" + str(annihilationOp_arr[-1])
            keylist.append(str1 + str2)

        for i in range(len(keylist)):
            opdict[keylist[i]] = Coeffarr[i]

        Op = FermionicOp(opdict)

        return Op
