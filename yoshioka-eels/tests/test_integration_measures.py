"""Measure, accelerated atomic sum, absolute spectra and checkpoint regressions."""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid

from benchmarks.born_quadrature import born_batch, checkpointed_born, line_spectra
from benchmarks.bulk_eels import displacement_for_q, primitive_atom_mapping
from benchmarks.eels_observables import absolute_spectrum
from benchmarks.workflow_io import ResourceLimits
from yoshioka.electron_phonon import LobatoSilicon, full_space_born
from yoshioka.integration_measures import detector_measure, physical_observables, uniform_bz_nodes
from tests.test_eels_convergence import synthetic_lattice


def small_spec(model):
    return {'positions_m': model.positions_m.tolist(), 'covariance_m2': (np.eye(3)*1.8e-23).tolist(),
            'transverse_lengths_m': [model.cell_m[0, 0], model.cell_m[1, 1]],
            'specimen_thickness_m': model.cell_m[2, 2], 'incident_energy_ev': 60000}


class MeasureTests(unittest.TestCase):
    def test_normalized_constant_unique_nodes_and_shifted_mesh(self):
        for shift in ((0, 0, 0), (.5, .5, .5)):
            q, w = uniform_bz_nodes(7, shift=shift)
            self.assertAlmostEqual(np.sum(w), 1.)
            self.assertAlmostEqual(np.sum(3.7*w), 3.7)
            self.assertEqual(len(np.unique(q, axis=0)), 7**3)
            self.assertEqual(np.sum(np.linalg.norm(q, axis=1) == 0), int(shift[0] == 0))
            a, _ = uniform_bz_nodes(7, 0, 13, shift)
            b, _ = uniform_bz_nodes(7, 13, None, shift)
            np.testing.assert_array_equal(q, np.concatenate((a, b)))

    def test_smooth_density_and_genuine_point_mass(self):
        n = 100
        x = (np.arange(n)+.5)/n
        xx, yy = np.meshgrid(x, x)
        # Midpoint rule integrates this linear density exactly on unit square.
        answer = detector_measure(2+xx+yy, 1/n**2, np.ones_like(xx, dtype=bool), [.3, .7], [True, False])
        self.assertAlmostEqual(answer, 3.3)
        # Changing continuum cell area cannot rescale an isolated point mass.
        self.assertEqual(detector_measure([0], [1e-9], [True], [.3], [True]), .3)

    def test_diagnostic_point_is_not_an_extra_detector_weight(self):
        values = {'central': .6, 'annulus': .4, 'exact_forward_q_zero': .01}
        physical = physical_observables(values)
        self.assertAlmostEqual(sum(physical.values()), 1.)
        self.assertEqual(values['central'], physical['central'])

    def test_absolute_spectrum_area_and_independent_direct_gaussians(self):
        energies = np.array([0., .12345, 3.4567, 10.])
        weights = np.array([.2, .1, .3, .1])
        grid = np.arange(-15, 25, .025)
        fast = line_spectra(energies, weights, grid)[:, 0]
        exact = absolute_spectrum(energies*1e-3*constants.elementary_charge, weights, grid)
        self.assertLess(trapezoid(abs(fast-exact), grid)/sum(weights), 5e-6)
        self.assertAlmostEqual(trapezoid(fast, grid), .7, places=12)
        self.assertLess(np.max(abs(fast-exact)), 1e-6)

    def test_vectorized_sum_vs_independent_old_atomic_loop(self):
        model = synthetic_lattice()
        spec = small_spec(model)
        q = np.array([[0., 0., 0.], [.12, -.23, .31]])
        qw = np.ones(2)/8
        energies, amplitudes, qxy, angles, mask = born_batch(model, model.positions_m, np.array(spec['covariance_m2']), spec['transverse_lengths_m'], 28e-10, 12e-10, q, qw, 60000*constants.elementary_charge)
        basis, _ = primitive_atom_mapping(model, model.positions_m)
        omega, vectors = model.modes(q)
        positions = model.positions_m.copy()
        positions[:, 2] += 12e-10
        for i in range(2):
            selected = mask[i]
            d = displacement_for_q(model, model.positions_m, basis, q[i], omega[i, selected], vectors[i][:, selected], 8)
            reference = full_space_born(LobatoSilicon.from_abtem_table(), positions, d, np.array(spec['covariance_m2']), qxy[i], omega[i, selected], 60000*constants.elementary_charge, np.prod(spec['transverse_lengths_m']), 28e-10)
            reference *= angles[i] <= 20+1e-12
            error = np.linalg.norm(amplitudes[i, selected]-reference)/np.linalg.norm(reference)
            self.assertLess(error, 2e-13)

    def test_checkpoint_resume_equivalence_and_corrupt_prefix_rejected(self):
        model = synthetic_lattice()
        spec = small_spec(model)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            folder, state, _ = checkpointed_born(model, spec, 3, (0, 0, 0), a, ResourceLimits(60, 2), batch_size=5, max_new_q=7)
            self.assertEqual(state['completed_q'], 7)
            _, resumed, ar = checkpointed_born(model, spec, 3, (0, 0, 0), a, ResourceLimits(60, 2), batch_size=5)
            _, fresh, br = checkpointed_born(model, spec, 3, (0, 0, 0), b, ResourceLimits(60, 2), batch_size=5)
            self.assertEqual(resumed['status'], 'COMPLETE')
            self.assertEqual(resumed['resumed_from_q'], 7)
            for key in ar:
                np.testing.assert_array_equal(ar[key], br[key])
            path = folder/'state.json'
            bad = json.loads(path.read_text())
            bad['completed_q'] = 28
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                checkpointed_born(model, spec, 3, (0, 0, 0), a, ResourceLimits(60, 2))


if __name__ == '__main__':
    unittest.main()
