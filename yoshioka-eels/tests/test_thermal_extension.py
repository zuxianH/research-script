"""Numerical controls for new thermal identifiability and material comparison."""
import unittest

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid

from yoshioka.thermal_model_validation import (
    PLANCK_MEV_PS, fit_mode_profile, linewidth_gate, mode_periodogram,
    window_fwhm_mev, window_power_kernel,
)


class FiniteWindowTests(unittest.TestCase):
    def test_window_density_and_duration_scaling(self):
        # Independent quadrature over a wide physical frequency interval.
        for kind, atol in [('hann', 1e-7), ('rectangular', 3e-4)]:
            t=25.; extent=1000 if kind=='rectangular' else 200
            e=np.linspace(-extent*PLANCK_MEV_PS/t,extent*PLANCK_MEV_PS/t,500001)
            integral=trapezoid(window_power_kernel(e,t,kind),e)
            self.assertLess(abs(integral-1),atol)
            self.assertAlmostEqual(window_fwhm_mev(t,kind)/window_fwhm_mev(2*t,kind),2.,places=10)

    def test_periodogram_parseval_and_energy_units(self):
        n=2048;dt=5e-15;k=311;t=np.arange(n)*dt
        x=np.cos(2*np.pi*k*np.arange(n)/n)
        e,p=mode_periodogram(x,dt,'rectangular')
        frequency_energy=constants.h*k/(n*dt)/(1e-3*constants.elementary_charge)
        self.assertAlmostEqual(e[np.argmax(p*(e>0))],frequency_energy,places=10)
        df=1/(n*dt)
        self.assertAlmostEqual(float(np.sum(p)*df),.5,places=12)

    def test_undamped_harmonic_line_does_not_create_lifetime(self):
        n=4000;dt=5e-15;k=332
        x=np.cos(2*np.pi*k*np.arange(n)/n)
        e,p=mode_periodogram(x,dt,'hann')
        center=constants.h*k/(n*dt)/(1e-3*constants.elementary_charge)
        fit=fit_mode_profile(e,p,center,n*dt*1e12,'hann',2.)
        gate=linewidth_gate([fit])
        self.assertEqual(gate['status'],'NOT_RESOLVED')
        self.assertIsNone(gate['lifetime_population_ps'])
        self.assertFalse(gate['profile_zero_width_excluded_diagnostic'])

    def test_stable_resolved_width_convention(self):
        # Exercise the physical lifetime conversion only after every gate.
        rows=[]
        for width in [1.18,1.2,1.22]:
            rows.append({'candidate_FWHM_mev':width,'fit_success':True,
                'fit_relative_RMS_residual':.01,'window_FWHM_mev':.06,
                'profile':[{'FWHM_mev':0.,'mean_squared_residual':.02},
                           {'FWHM_mev':width,'mean_squared_residual':.0001}]})
        gate=linewidth_gate(rows)
        self.assertEqual(gate['status'],'RESOLVED_IN_TESTED_CLASSICAL_MODEL')
        expected=constants.hbar/(1.2e-3*constants.elementary_charge)*1e12
        self.assertAlmostEqual(gate['lifetime_population_ps'],expected)




if __name__=='__main__':unittest.main()
