import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import trapezoid
from scipy import constants

from yoshioka.mode_spectra import (PLANCK_MEV_PS, fit_windowed_mode, hann_energy_kernel,
    hann_fwhm_mev, periodogram_modes)
from yoshioka.molecular_dynamics import checkpointed_md
from benchmarks.workflow_io import ROOT, ResourceLimits


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

    def test_actual_MOD_restart_has_no_missing_or_duplicate_frames(self):
        from ase.io import read
        unit = read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
        settings = {'repetitions': [2, 2, 2], 'production_steps': 400, 'sample_stride': 1,
            'equilibration_steps': 20, 'temperature_k': 300, 'seed': 82731,
            'timestep_fs': 1, 'thermostat_time_fs': 100}
        class StopAfterCheckpoint:
            calls = 0
            def check(self, **kwargs):
                self.calls += 1
                if self.calls == 5:
                    raise TimeoutError('test controlled MD stop')
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            folder, partial = checkpointed_md(unit, ROOT/'Si.tersoff.mod', settings, a, StopAfterCheckpoint())
            self.assertEqual(partial['completed_frames'], 200)
            _, resumed = checkpointed_md(unit, ROOT/'Si.tersoff.mod', settings, a, ResourceLimits(60, 2))
            reference, fresh = checkpointed_md(unit, ROOT/'Si.tersoff.mod', settings, b, ResourceLimits(60, 2))
            self.assertEqual(resumed['completed_frames'], 400)
            self.assertEqual(resumed['resumed_from_frames'], 200)
            for name in ('displacements_m', 'velocities_m_s', 'forces_n'):
                actual = np.load(folder/(name+'.npy'))
                expected = np.load(reference/(name+'.npy'))
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 2e-11)


if __name__ == '__main__':
    unittest.main()
