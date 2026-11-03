import unittest
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from yoshioka.detector_cells import cell_fractions, circle_rectangle_area
from benchmarks.detector_cubature import checkpointed_cells
from benchmarks.workflow_io import ResourceLimits
from tests.test_eels_convergence import synthetic_lattice
from yoshioka.lattice_dynamics import finite_periodic_modes


class DetectorCellTests(unittest.TestCase):
    def test_circle_rectangle_area_against_independent_quadrature(self):
        rng = np.random.default_rng(98231)
        for _ in range(30):
            x0, x1 = np.sort(rng.uniform(-2, 2, 2))
            y0, y1 = np.sort(rng.uniform(-2, 2, 2))
            r = 1.7
            def height(x):
                extent = np.sqrt(max(0, r*r-x*x))
                return max(0., min(y1, extent)-max(y0, -extent)) if abs(x) < r else 0.
            expected = quad(height, x0, x1, epsabs=1e-10, points=[v for v in (-r, 0, r) if x0 < v < x1], limit=200)[0]
            self.assertAlmostEqual(float(circle_rectangle_area(x0, x1, y0, y1, r)), expected, delta=2e-8)

    def test_partition_constant_area_and_no_detector_renormalization(self):
        for spacing in ((.7, .9), (.31, .27)):
            x = np.arange(-22, 22, spacing[0])
            y = np.arange(-22, 22, spacing[1])
            xx, yy = np.meshgrid(x, y)
            centers = np.column_stack((xx.ravel(), yy.ravel()))
            detectors, pixels, halves = cell_fractions(centers, spacing)
            area = np.prod(spacing)
            np.testing.assert_allclose(np.sum(pixels, axis=-1), detectors[:, 0], atol=2e-11)
            np.testing.assert_allclose(np.sum(halves, axis=-1), detectors[:, 0], atol=2e-11)
            self.assertAlmostEqual(np.sum(detectors[:, 0])*area, np.pi*20**2, places=8)
            self.assertAlmostEqual(np.sum(detectors[:, 1])*area, np.pi*5**2, places=8)

    def test_smooth_angular_integral_converges(self):
        expected = 2*np.pi*(1-np.exp(-.5))
        errors = []
        for h in (.2, .1, .05):
            x = np.arange(-1.2, 1.2+h/2, h)
            xx, yy = np.meshgrid(x, x)
            areas = circle_rectangle_area(xx-h/2, xx+h/2, yy-h/2, yy+h/2, 1.)
            estimate = np.sum(np.exp(-.5*(xx**2+yy**2))*areas)
            errors.append(abs(estimate-expected))
        self.assertLess(errors[-1], errors[0]/8)

    def test_cell_cubature_checkpoint_resume_matches_uninterrupted(self):
        model = synthetic_lattice()
        modes = finite_periodic_modes(model, (2, 2, 2))
        spec = {'positions_m': modes['positions_m'].tolist(), 'covariance_m2': (np.eye(3)*1.8e-23).tolist(),
                'incident_energy_ev': 60000, 'transverse_lengths_m': [2*model.cell_m[0, 0], 2*model.cell_m[1, 1]],
                'specimen_thickness_m': 2*model.cell_m[2, 2]}
        class StopAfterTwoBatches:
            calls = 0
            def check(self, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    raise TimeoutError('test controlled stop')
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _, partial, _ = checkpointed_cells(model, spec, 3, [0, 0, 0], Path(a), StopAfterTwoBatches(), batch=5)
            self.assertEqual(partial['completed_q'], 10)
            _, resumed, arrays = checkpointed_cells(model, spec, 3, [0, 0, 0], Path(a), ResourceLimits(60, 2), batch=5)
            _, fresh, reference = checkpointed_cells(model, spec, 3, [0, 0, 0], Path(b), ResourceLimits(60, 2), batch=5)
            self.assertEqual(resumed['status'], 'COMPLETE')
            for name in arrays:
                np.testing.assert_array_equal(arrays[name], reference[name])


if __name__ == '__main__':
    unittest.main()
