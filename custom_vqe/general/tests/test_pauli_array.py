"""Tests for Pauli Array manipulation."""

import unittest

from custom_vqe.general.pauli_array import PauliArr


class TestPauliArr(unittest.TestCase):
    """Unit test for Pauli array manipulation."""

    qwc_chk1, qwc_chk2 = "XIYIZI", "IXIYIZ"
    gc_chk1, gc_chk2 = "XX", "YY"

    pauli_len1 = "IXYZ"
    pauli_len2 = "XY"

    def setUp(self):
        """Create instance of the PauliArr class."""
        pauli_list = ["iIXYZ", "-XXXX", "-iXYYY", "YYYY"]
        coeff_list = [0.5j, 1.0 + 0.6j, 1.0, 1.0]

        self.pauliobj = PauliArr(pauli_list, coeff_list)

    def test_commute_pauli_strings_qwc(self):
        """Test for qubit-wise commutation of two pauli strings."""
        self.assertTrue(self.pauliobj.check_commute_paulis(self.qwc_chk1, self.qwc_chk2, qwc=True))
        self.assertFalse(self.pauliobj.check_commute_paulis(self.gc_chk1, self.gc_chk2, qwc=True))

    def test_commute_pauli_strings_gc(self):
        """Test for general commutation of two pauli strings."""
        self.assertTrue(self.pauliobj.check_commute_paulis(self.gc_chk1, self.gc_chk2))
        self.assertFalse(self.pauliobj.check_commute_paulis(self.qwc_chk1, self.qwc_chk2))

    def test_commute_method_error(self):
        """Test raises error if pauli strings are of inequal lenghts."""
        with self.assertRaises(ValueError):
            self.assertEqual(
                PauliArr.check_commute_paulis(self.pauli_len1, self.pauli_len2),
                "Length of the two strings must be equal!",
            )
