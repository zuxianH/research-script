import tempfile
from pathlib import Path
import unittest

import numpy as np

from yoshioka.spectral_data import conservative_rebin, gaussian_broaden_bin_weights, inspect_numeric_array, kubo_factor, require_verified_metadata


class SpectralDataTests(unittest.TestCase):
    def test_conservative_rebin_absolute_area_and_crop(self):
        weights = np.array([.2, .6, .3])
        source = np.array([0., 1., 3., 4.])
        target = np.array([0., .5, 1.5, 2.5, 4.])
        rebinned = conservative_rebin(weights, source, target)
        self.assertAlmostEqual(sum(rebinned), 1.1)
        cropped = conservative_rebin(weights, source, [1., 2., 3.])
        np.testing.assert_allclose(cropped, [.3, .3])
        density = weights/np.diff(source)
        np.testing.assert_allclose(conservative_rebin(density, source, target, source_density=True), rebinned)

    def test_multiple_axes_preserve_total(self):
        values = np.arange(6).reshape(2, 3)
        a = conservative_rebin(values, [0, 1, 2, 3], [0, 1.5, 3], axis=1)
        b = conservative_rebin(a, [0, 1, 2], [0, .5, 2], axis=0)
        self.assertAlmostEqual(np.sum(b), np.sum(values))

    def test_forward_broadening_retains_full_area_outside_crop(self):
        broadened, extra = gaussian_broaden_bin_weights([.2, .3], .919, 2.)
        self.assertAlmostEqual(float(np.sum(broadened)), .5, places=14)
        self.assertLess(float(np.sum(broadened[extra:extra+2])), .5)

    def test_kubo_zero_limit_extremes_and_balance(self):
        energy = np.array([-1e6, -60., 0., 60., 1e6])
        factors = kubo_factor(energy, 300.)
        self.assertEqual(factors[2], 1.)
        self.assertTrue(np.all(np.isfinite(factors)))
        self.assertTrue(np.all(factors >= 0))
        self.assertAlmostEqual(factors[1]/factors[3], np.exp(-60/(.08617333262145178*300)))

    def test_missing_metadata_blocks_quantitative_comparison(self):
        with self.assertRaises(ValueError):
            require_verified_metadata({'energy': {'status': 'USER_SUPPLIED'}}, ['energy'])
        require_verified_metadata({'energy': {'status': 'VERIFIED'}}, ['energy'])

    def test_read_only_inspection_and_no_axis_inference(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'array.npy'
            np.save(path, np.array([[1., -2.], [np.nan, 3.]]))
            original = path.read_bytes()
            data, report = inspect_numeric_array(path, chunk_size=1)
            self.assertFalse(data.flags.writeable)
            self.assertEqual(report['finite'], 3)
            self.assertEqual(report['nonfinite'], 1)
            self.assertEqual(report['negative_real'], 1)
            self.assertNotIn('axis_order', report)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
