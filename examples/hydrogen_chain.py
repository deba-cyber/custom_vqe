"""Generating hydrogen chain model."""

import sys

import numpy as np

output_file = sys.argv[1]

bond_distances = np.arange(0.6, 3.05, 0.1)


# Number of atoms.
n_atom = 10

all_configs = []

for bond in bond_distances:

    r = 0.5 * bond / np.sin(np.pi/n_atom)

    for i in range(n_atom):
        theta = i * (2 * np.pi / n_atom)
        x = round(r * np.cos(theta), 6)
        y = round(r * np.sin(theta), 6)
        z = 0.0

        all_configs.append(['H', " ", f"{x:.8f}", " ", f"{
                           y:.8f}", " ", f"{z:.8f}\n"])


def write_mol_xyz_file(output_file: str,
                       config_arr: list,
                       linebreak: int) -> None:
    """Writing configurations to file.

    :param output_file: A string denoting output filename.
    :param config_arr: A list with the configurations.
    :param linebreak: An integer denoting linebreak point.

    """

    with open(output_file, 'w') as f:
        num_config = 0
        ctr = 0
        for ele in config_arr:
            ele = [
                " " + elem if elem not in ("H", " ") and elem[0] != "-" else elem for elem in ele]
            f.writelines(ele)
            ctr += 1

            if ctr == n_atom and num_config < len(bond_distances) - 1:
                num_config += 1
                ctr = 0
                f.writelines("\n")


if __name__ == "__main__":
    write_mol_xyz_file(output_file, all_configs, n_atom)
