"""checking some aspects ( e.g. circuit depth ) in vqe calculation with qiskit implementation."""

import json
import os
import sys

import numpy as np
import qiskit
import qiskit_nature
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from scipy.optimize import minimize


class qiskit_vqe:
    r"""Creates object for qiskit_vqe class."""

    def __init__(
        self,
        symbols: list[list[str]],
        coords: list[list[list[str]]],
        orb_remove: list[int] | None,
        charge: int,
        multiplicity: int,
        basis: str = "sto3g",
        freezecore: bool = False,
        mapper_type: qiskit_nature.second_q.mappers = JordanWignerMapper,
    ) -> None:
        """Create qiskit_vqe object.

        :param symbols: A 2D list where each list contains strings of atom symbols.
            For only one configuration, length of 2D list is 1.
        :param coords: A 3D list where each 2D list corresponds to all (x,y,z) coordinates of the atoms.
            For only one configuration, length of 3D list is 1.
        :param orb_remove: A list of integers (orbital indices) to be removed, can be `None`.
        :param charge: An integer denoting the charge of the system.
        :param multiplicity: An integer denoting the spin multiplicity of the system. (2 * spin + 1).
        :param basis: A string denoting the basis function. Default is ``sto3g``.
        :param freezecore: Determines if ``freezecore`` option is to be applied or not. Default is ``True``.
        :param mapper_type: Fermion-Qubit mapper. Default is ``JordanWignerMapper()``.

        """
        self._symbols = symbols
        self._coords = coords
        self._remove_orbs = orb_remove
        self._basis = basis
        self._charge = charge
        self._multiplicity = multiplicity
        self._freezecore = freezecore
        self._mapper_type = mapper_type

    @staticmethod
    def build_callback(
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        estimator: Estimator,
        callback_dict: dict,
    ):
        """Compute Callback function for tracking progress of the `VQE` calculation."""

        def my_callback(current_vector):
            cur_cost = estimator.run([(ansatz, hamiltonian, [current_vector])]).result()[0].data.evs[0]

            callback_dict["prev vector"] = current_vector
            callback_dict["Iterations done"] += 1
            callback_dict["Cost history"].append(cur_cost)

            print(
                f"Iterations done: {callback_dict['Iterations done']}, current cost: {cur_cost}",
                end="\r",
                flush=True,
            )

        return my_callback

    @staticmethod
    def cost_func(
        params: list | np.ndarray,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        estimator: Estimator,
    ) -> float:
        """Compute cost function for `VQE` calculation."""
        # length of params must be equal to number of parameters in the circuit.
        assert ansatz.num_parameters == len(params)

        return estimator.run([(ansatz, hamiltonian, [params])]).result()[0].data.evs[0]

    def do_qiskit_vqe(
        self, verbose_es: int = 1, do_transpile: bool = False, do_vqe: bool = False
    ) -> list[float] | None:
        """Compute `VQE` calculation under different conditions based on user input.

        :param verbose_es: An integer denoting extent of printing to be done about the electronic
            structure calculations. Default is `1` for which minimum printing will be done without
            the Fermionic, Qubit operator and ansatz circuit.
            It can also be `2` for which Fermionic, Qubit operators and ansatz circuit are also printed.
        :param do_transpile: A boolean denoting if transpilation of the circuit is to be done. Default is `False`.
        :param do_vqe: A boolean denoting if `VQE` calculation is to be carried out. Default is `False`.
        :return: A list of floats (results of `VQE` calculations if do_vqe = True) or None (if do_vqe = False).

        """
        assert len(self._symbols) == len(self._coords)

        num_particles_dict = {}
        num_spatial_orbitals_dict = {}
        num_spin_orbitals_dict = {}
        num_alpha_dict = {}
        num_beta_dict = {}
        fermionic_op_dict = {}
        qubit_op_dict = {}

        E_vqe = []

        for i in range(len(self._symbols)):
            callback_dict = {
                "prev vector": None,
                "Iterations done": 0,
                "Cost history": [],
            }
            molecule = MoleculeInfo(
                symbols=self._symbols[i],
                coords=tuple(self._coords[i]),
                multiplicity=self._multiplicity,
                charge=self._charge,
            )

            driver = PySCFDriver.from_molecule(molecule, basis=self._basis)

            problem = driver.run()

            if self._freezecore:
                if self._remove_orbs is not None:
                    problem = FreezeCoreTransformer(freeze_core=True, remove_orbitals=self._remove_orbs).transform(
                        problem
                    )

            num_particles = problem.num_particles
            num_spatial_orbitals = problem.num_spatial_orbitals

            # Fermion-Qubit mapper
            mapper = self._mapper_type()
            # fermionic operator
            fermionic_op = problem.second_q_ops()[0]
            # qubit operator
            qubit_op = mapper.map(fermionic_op)

            num_particles_dict[i] = problem.num_particles
            num_spatial_orbitals_dict[i] = problem.num_spatial_orbitals
            num_spin_orbitals_dict[i] = problem.num_spin_orbitals
            num_alpha_dict[i] = problem.num_alpha
            num_beta_dict[i] = problem.num_beta
            fermionic_op_dict[i] = fermionic_op
            qubit_op_dict[i] = qubit_op

            init_state = HartreeFock(num_spatial_orbitals, num_particles, JordanWignerMapper())

            ansatz = UCCSD(num_spatial_orbitals, num_particles, JordanWignerMapper(), initial_state=init_state)

            #            for k in range(len(ansatz.operators)):
            #                ansatz = ansatz.decompose()

            # transpilation
            if do_transpile:
                print("Transpiling ...")
                backend = GenericBackendV2(num_qubits=ansatz.num_qubits)
                ansatz = transpile(ansatz, backend=backend, optimization_level=3)

            match verbose_es:
                case 1:
                    print("Electronic structure calculations details ...")
                    print(f"Driver: {type(driver)}")
                    print(f"Problem: {type(problem)}")
                    print(f"Number of electrons : {problem.num_particles}")
                    print(f"Number of spatial orbitals: {problem.num_spatial_orbitals}")
                    print(f"Number of alpha electrons : {problem.num_alpha}")
                    print(f"Number of beta electrons : {problem.num_beta}")
                    print(f"Nuclear repulsion energy in Hartree : {problem.nuclear_repulsion_energy} Ha")
                    print(f"Reference energy in Hartree : {problem.reference_energy} Ha")
                    print(f"Gate counts : {ansatz.count_ops()}")
                    print(f"Circuit depth : {ansatz.depth()}")
                    print(f"Number of parameters : {ansatz.num_parameters}")
                    print("===========================================")

                case 2:
                    print("Electronic structure calculations details ...")
                    print(f"Driver: {type(driver)}")
                    print(f"Problem: {type(problem)}")
                    print(f"Number of electrons : {problem.num_particles}")
                    print(f"Number of spatial orbitals: {problem.num_spatial_orbitals}")
                    print(f"Number of alpha electrons : {problem.num_alpha}")
                    print(f"Number of beta electrons : {problem.num_beta}")
                    print(f"Nuclear repulsion energy in Hartree : {problem.nuclear_repulsion_energy} Ha")
                    print(f"Reference energy in Hartree : {problem.reference_energy} Ha")
                    print("===========================================")

                    print("Operator details ...")
                    print(f"Second quantized operator : {fermionic_op}")
                    print(f"Qubit operator : {qubit_op}")
                    print("===========================================")
                    print(f"Gate counts : {ansatz.count_ops()}")
                    print(f"Circuit depth : {ansatz.depth()}")
                    print(f"Number of parameters : {ansatz.num_parameters}")

                    print(f"Ansatz circuit (UCCSD) : {ansatz}")
                    print("===========================================")

            if do_vqe:
                # number of parameters in the ansatz circuit.
                num_params = ansatz.num_parameters

                params = 2 * np.pi * np.random.random(num_params)

                estimator = Estimator()

                res = minimize(
                    qiskit_vqe.cost_func,
                    x0=params,
                    args=(ansatz, qubit_op, estimator),
                    callback=qiskit_vqe.build_callback(ansatz, qubit_op, estimator, callback_dict),
                    method="cobyla",
                    options={"maxiter": 1000},
                )

                E_vqe.append(res["fun"] + problem.nuclear_repulsion_energy)

        return E_vqe


if __name__ == "__main__":
    print(f"qiskit-version : {qiskit.__version__}")

    # Fetching input file and folder for data .

    filename = sys.argv[1]
    foldername = sys.argv[2]

    # Absolute path of the directory of the current python script.
    cur_pyscript_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the target folder.
    target_folder_path = os.path.join(cur_pyscript_dir, "..")

    # Full path to the target file.
    filepath = os.path.join(target_folder_path, filename)


    try:
        with open(filepath, "r") as file:
            lines = file.readlines()

            all_symbols, all_coords = [], []
            symbols, coords = [], []
            for line in lines:
                if not line.strip():
                    all_symbols.append(symbols)
                    all_coords.append(coords)
                    symbols = []
                    coords = []
                else:
                    cur_line = line.split()
                    symbols.append(cur_line[0])
                    coords.append([float(cur_line[1]), float(cur_line[2]), float(cur_line[3])])

            all_symbols.append(symbols)
            all_coords.append(coords)

    except FileNotFoundError:
        print(f"Error: unable to fetch file in folder {filepath}")

    # getting charge and multiplicity information from json file.
    with open("../moldata.json") as f:
        data = json.load(f)

        multiplicity = data["multiplicity"][0]
        charge = data["charge"][0]

    # creating object of qiskit_vqe class.
    qiskit_vqe_obj = qiskit_vqe(all_symbols, all_coords, None, 0, 1)

    # VQE results.
    qiskit_vqe_obj.do_qiskit_vqe(do_transpile=True)
