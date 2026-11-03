import unittest

import numpy as np
from scipy.integrate import trapezoid
from scipy import constants

from yoshioka.mode_spectra import (PLANCK_MEV_PS, fit_windowed_mode, hann_energy_kernel,
    hann_fwhm_mev, periodogram_modes)


class ModeSpectrumTests(unittest.TestCase):
    def test_window_area_resolution_and_duration_scaling(self):
        energy = np.linspace(-30, 30, 100001)
        self.assertAlmostEqual(trapezoid(hann_energy_kernel(energy, 20), energy), 1., places=9)
        self.assertAlmostEqual(hann_fwhm_mev(20)*2, hann_fwhm_mev(10), places=10)
        self.assertAlmostEqual(hann_fwhm_mev(20)/(PLANCK_MEV_PS/20), 1.44058258, places=6)

    def test_harmonic_mode_energy_axis_and_unresolved_width(self):
        cadence = 5e-15
        t = np.arange(4000)*cadence
        expected = 60.
        signal = np.cos(2*np.pi*(expected*1e-3*constants.elementary_charge/constants.h)*t)
        energy, power = periodogram_modes(signal, cadence)
        fitted = fit_windowed_mode(energy, power, expected, 20.)
        self.assertLess(abs(fitted['center_mev']-expected), fitted['energy_bin_mev']/2)
        self.assertLess(fitted['fitted_intrinsic_FWHM_mev'], 2*fitted['Hann_power_window_FWHM_mev'])



if __name__ == '__main__':
    unittest.main()
