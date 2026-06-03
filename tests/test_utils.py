import unittest
from sigconfide.utils.utils import *
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
class TestUtils(unittest.TestCase):
    def test_detect_format(self):
        """Test the detection of data formats."""
        csv_line = 'C>A,ACA,66,83,88,78,90,46,73'
        tsv_line = 'AC>AA\t66\t83\t88\t78\t90\t46\t73'
        mutated_tsv_line = 'A[C>A]A\t66\t83\t88\t78\t90\t46\t73'  # Example same as TSV for illustration
        unknown_line = 'C>A ACA 66 83 88 78 90 46 73'

        # Test CSV format detection
        self.assertEqual(detect_format(csv_line), ('CSV Format', ','), "Should detect CSV Format")

        # Test TSV format detection
        self.assertEqual(detect_format(tsv_line), ('TSV Format', '\t'), "Should detect TSV Format")

        # Test Mutated TSV format detection
        self.assertEqual(detect_format(mutated_tsv_line), ('Mutated TSV Format', '\t'), "Should detect Mutated TSV Format")

        # Test unknown format detection
        self.assertEqual(detect_format(unknown_line), ('Unknown Format', None), "Should detect Unknown Format")

    def test_detect_format(self):
        """Test the detection of data formats."""
        with open(os.path.join(current_dir, 'data', 'format_2.dat'), 'r') as file:
            csv_line = file.read()

        with open(os.path.join(current_dir, 'data', 'format_1.dat'), 'r') as file:
            tsv_line = file.read()

        with open(os.path.join(current_dir, 'data', 'tumorBRCA.txt'), 'r') as file:
            csv_line2 = file.read()

        # Assume detect_format function exists and performs format detection
        # Test CSV format detection
        self.assertEqual(detect_format(csv_line), ('CSV Format', ','), "Should detect CSV Format")

        # Test TSV format detection
        self.assertEqual(detect_format(tsv_line), ('Mutated TSV Format', '\t'), "Should detect TSV Format")

        # Test CSV format detection
        self.assertEqual(detect_format(csv_line2), ('Mutated TSV Format', '\t'), "Should detect Mutated CSV Format")

    def test_frobenius_norm(self):
        M = np.array([[1, 2], [4, 5]])
        P = np.array([[1, 1], [2, 8]])
        E = np.array([[2, 0], [1, 1]])

        result = FrobeniusNorm(M, P, E)

        # Define the expected Frobenius norm value
        expected_result = 8.83176

        # Check if the result matches the expected result
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_is_wholenumber(self):
        # Test with whole numbers
        self.assertTrue(is_wholenumber(2), "2 should be identified as a whole number")
        self.assertTrue(is_wholenumber(0), "0 should be identified as a whole number")
        self.assertTrue(is_wholenumber(-3), "-3 should be identified as a whole number")

        # Test with a number very close to whole number (within tolerance)
        self.assertTrue(is_wholenumber(2.000000000000001),
                        "2.000000000000001 should be identified as a whole number")

        # Test with non-whole numbers
        self.assertFalse(is_wholenumber(3.5), "3.5 should not be identified as a whole number")
        self.assertFalse(is_wholenumber(-1.2), "-1.2 should not be identified as a whole number")

        # Test with numbers close to whole numbers within the default tolerance
        self.assertTrue(is_wholenumber(2.0000000000000001),
                        "2.0000000000000001 should be identified as a whole number within default tolerance")
        self.assertTrue(is_wholenumber(-3.0000000000000001),
                        "-3.0000000000000001 should be identified as a whole number within default tolerance")


class TestLoadSamplesFile(unittest.TestCase):
    def test_load_csv(self):

        samples, names = load_samples_file(os.path.join(current_dir, 'data', 'test_csv.csv'))

        expected_result = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        np.testing.assert_array_equal(samples, expected_result)

    def test_load_tsv(self):
        samples, names = load_samples_file(os.path.join(current_dir, 'data', 'test_tsv.tsv'))

        expected_result = np.array([[2, 3], [5, 6]])
        np.testing.assert_array_equal(samples, expected_result)

    def test_load_breast_signatures_reordered(self):
        # Breast_Signatures.csv is loaded via load_signatures_file.
        # It has 96 SBS rows but sorted differently.
        # Check that it gets reordered to standard SBS order.
        signatures, names = load_signatures_file(os.path.join(current_dir, 'data', 'Breast_Signatures.csv'))
        self.assertEqual(signatures.shape, (96, 28))
        # Raw file has row index 4 (5th data row, line 7 of the file) as "C[C>A]A".
        # In standard SBS order, this is at index 24.
        # Let's verify that the values for row index 4 from the raw file are indeed at index 24 of the loaded signatures.
        # Raw row values: 0.000893614751327785, 0.0076476519510486, ...
        np.testing.assert_almost_equal(signatures[24, 0], 0.000893614751327785)
        np.testing.assert_almost_equal(signatures[24, 1], 0.0076476519510486)

    def test_load_format_2_reordered(self):
        # format_2.dat has Trinucleotide and Mutation type columns and is SBS.
        # Check that it is correctly standardized and reordered.
        samples, names = load_samples_file(os.path.join(current_dir, 'data', 'format_2.dat'))
        self.assertEqual(samples.shape, (96, 100))
        # Raw row index 4 in format_2.dat (5th data row, line 6 of file) is C>A, CCA -> C[C>A]A.
        # Values: 66, 92, 72, 81...
        # In standard SBS, C[C>A]A is at index 24.
        np.testing.assert_array_equal(samples[24, :4], [66, 92, 72, 81])


