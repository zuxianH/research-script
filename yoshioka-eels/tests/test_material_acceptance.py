import unittest

from benchmarks.material_evidence import relative_difference


class MaterialAcceptanceTests(unittest.TestCase):
    def test_finite_signal_disappearing_cannot_pass_as_zero_division_case(self):
        significant = relative_difference(1e-7, 0.)
        self.assertIsNone(significant['relative_change'])
        self.assertGreater(significant['acceptance_error'], .01)
        self.assertEqual(relative_difference(0., 0.)['acceptance_error'], 0.)
        # Existing B7 weak scale 1e-18 and 1% target allow only <1e-20
        # absolute changes at zero; they do not excuse a finite lost signal.
        self.assertLess(relative_difference(1e-22, 0.)['acceptance_error'], .01)
