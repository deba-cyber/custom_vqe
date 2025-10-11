"""Generating hydrogen molecule model."""

import sys

import numpy as np

output_file = sys.argv[1]

bond_distances = np.arange(0.5, 2.6, 0.1)

all_configs = []

n_atom = 2

for bond in bond_distances:

    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    x1 = 0.0
    y1 = 0.0
    z1 = bond

    all_configs.append(['H', " ", f"{x0:.8f}", " ", f"{y0:.8f}", " ", f"{z0:.8f}\n"])
    all_configs.append(['H', " ", f"{x1:.8f}", " ", f"{y1:.8f}", " ", f"{z1:.8f}\n"])


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
