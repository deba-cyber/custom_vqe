"""Tests for Fermionic_ex_UCC class."""

import unittest

from numpy import isclose

from custom_vqe.ansatz.ansatz_ucc import Fermionic_ex_UCC as UCC


class TestUCC(unittest.TestCase):
    """Unit test for UCC ansatz."""

    # dictionary for character table for C2v point group.
    C2V_dict = {"A1": [1, 1, 1, 1], "A2": [1, 1, -1, -1], "B1": [1, -1, 1, -1], "B2": [1, -1, -1, 1]}

    ucc_obj_lih = UCC([1, 3], [0, 2], [5, 7, 9, 11], [4, 6, 8, 10], C2V_dict)
    lih_orb_sym = {
        0: "A1",
        1: "A1",
        2: "A1",
        3: "A1",
        4: "A1",
        5: "A1",
        6: "B1",
        7: "B1",
        8: "B2",
        9: "B2",
        10: "A1",
        11: "A1",
    }

    def setUp(self):
        """Create instance of the UCC class."""
        self.uccobj_lih = UCC([1, 3], [0, 2], [5, 7, 9, 11], [4, 6, 8, 10], self.C2V_dict)

    def test_pdt_irreps(self):
        """Test for tensor product of irreducible representaions of a point group."""
        irreps_pdt = {
            ("A1", "A2"): "A2",
            ("A1", "B1"): "B1",
            ("A1", "B2"): "B2",
            ("A2", "B1"): "B2",
            ("A2", "B2"): "B1",
            ("B1", "B2"): "A2",
        }
        self.assertEqual(UCC._pdt_irreps(self.C2V_dict), irreps_pdt)

    def test_singles_doubles_excitations(self):
        """Test for basic checking of singles and doubles excitations."""
        self.assertEqual(len(self.uccobj_lih._get_config_singles(4, 8, spin_type="all")), 32)
        self.assertEqual(len(self.uccobj_lih._get_config_doubles(4, 8, spin_type="all")), 168)

    def test_reduction_orbital_symmetry(self):
        """Test for reduction of ucc excitations with invoking orbital symmetry."""
        val_singles = (32 - len(self.uccobj_lih._restrict_orbital_symmetry(4, 8, self.lih_orb_sym)[0])) / 32
        val_doubles = (168 - len(self.uccobj_lih._restrict_orbital_symmetry(4, 8, self.lih_orb_sym)[1])) / 168
        self.assertTrue(isclose(100 * val_singles, 75, atol=5))
        self.assertTrue(isclose(100 * val_doubles, 80, atol=5))

    def test_doubles_errors(self):
        """Test raises errors of doubles excitations invalid inputs."""
        with self.assertRaises(ValueError):
            self.assertEqual(
                self.uccobj_lih._get_config_doubles(1, 4),
                "Number of electrons to excite can't be less than 2 for double excitations!",
            )
        with self.assertRaises(ValueError):
            self.assertEqual(
                self.uccobj_lih._get_config_doubles(2, 10),
                "At least two virtual orbitals needed for double excitations!",
            )
