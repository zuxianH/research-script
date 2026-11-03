import unittest
import numpy as np
from scipy import constants
from yoshioka.free_propagation import TransverseGrid
from yoshioka.pair_intermediate_momentum import integer_beams, propagate_intermediate_resolved
from yoshioka.pair_dispersion_sensitivity import (
    PARAXIAL_ID, SENSITIVITY_ID, kinetic_rates, propagate_dispersion_control)


class DispersionSensitivityTests(unittest.TestCase):
    def test_paraxial_specialization_matches_verified_resolved_evolution(self):
        rng = np.random.default_rng(3112026)
        grid = TransverseGrid(6e-10, 8e-10, 3, 3)
        out = integer_beams(3)
        linear = (rng.normal(size=(2, 5, 5)) + 1j * rng.normal(size=(2, 5, 5))) * constants.elementary_charge
        quadratic = (rng.normal(size=(4, 9)) + 1j * rng.normal(size=(4, 9))) * constants.elementary_charge
        omega = np.array([2e14, 7e14])
        shifts = np.array([[2e9, -3e9], [-1e9, 2e9]])
        for substeps in (1, 3):
            reference = propagate_intermediate_resolved(grid, omega, shifts, 1e-9, 9,
                lambda i: linear, lambda i: quadratic, 5, out, substeps)
            actual = propagate_dispersion_control(reference, 1e-9,
                lambda i: linear, lambda i: quadratic, PARAXIAL_ID)
            for key in ['direct_amplitudes', 'sequential_amplitudes', 'total_amplitudes',
                        'ordered_by_intermediate', 'first_amplitudes']:
                np.testing.assert_array_equal(actual[key], reference[key])

    def test_scalar_rate_and_explicit_unsupported_cone_rejection(self):
        k0 = 1.3e12
        radius = np.array([0., 1e-6, .1, .5, .75])
        q = np.column_stack((radius * k0, np.zeros(len(radius))))
        exact = kinetic_rates(q, k0, SENSITIVITY_ID)
        np.testing.assert_allclose(exact, k0 * radius**2 / (1 + np.sqrt(1 - radius**2)), rtol=3e-16)
        self.assertEqual(exact[0], 0.)
        self.assertGreater(exact[3], kinetic_rates(q, k0, PARAXIAL_ID)[3])
        with self.assertRaisesRegex(ValueError, 'Q>=k0'):
            kinetic_rates(np.array([[k0, 0.]]), k0, SENSITIVITY_ID)
        with self.assertRaisesRegex(ValueError, 'identifier'):
            kinetic_rates(q, k0, 'unregistered')


if __name__ == '__main__':
    unittest.main()
