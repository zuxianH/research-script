"""Statistical safety checks for complete-observable control variates."""
import importlib
import unittest
import numpy as np

m=importlib.import_module('benchmarks.14_two_phonon_eels.pair_sampling_audit')


class PairSamplingAuditTests(unittest.TestCase):
    def test_independent_coarse_integral_error_is_retained(self):
        coarse=np.arange(8,dtype=float)[:,None]+2
        baseline=np.array([1.,2.,6.,12.,14.,15.,16.,30.])[:,None]
        r=m.paired_control_variate(coarse,coarse,baseline,'independent')
        np.testing.assert_allclose(r['mean'],baseline.mean(axis=0))
        np.testing.assert_allclose(r['standard_error']**2,baseline.var(axis=0,ddof=1)/8)
        self.assertGreater(float(r['halfwidth95'][0]),0)
        self.assertFalse(r['coarse_integral_treated_as_exact'])

    def test_shared_coarse_baseline_cannot_invent_variance_reduction(self):
        coarse=np.arange(8,dtype=float)[:,None]+2
        fine=3*coarse+np.array([.1,.3,.4,.1,.5,.7,.2,.9])[:,None]
        r=m.paired_control_variate(fine,coarse,coarse,'paired')
        np.testing.assert_allclose(r['mean'],fine.mean(axis=0))
        np.testing.assert_allclose(r['standard_error']**2,fine.var(axis=0,ddof=1)/8)
        np.testing.assert_allclose(r['total_variance_over_fine'],1)
        self.assertGreater(float(r['baseline_correction_mean_covariance'][0]),0)

    def test_independent_baseline_can_make_a_correlated_control_unhelpful(self):
        coarse=np.arange(8,dtype=float)[:,None]+1;fine=coarse+.1
        baseline=100*coarse
        r=m.paired_control_variate(fine,coarse,baseline,'independent')
        self.assertLess(float(r['correction_variance_over_fine'][0]),1e-25)
        self.assertGreater(float(r['total_variance_over_fine'][0]),100)
        with self.assertRaises(ValueError):m.paired_control_variate(fine,coarse,baseline,'unknown')

    def fixture(self):
        raw=np.ones((8,len(m.WINDOWS),4))
        spectrum=sum(np.exp(-.5*((m.AXIS-x)/.84932)**2) for x in (25,50,72,105,140))
        spectra=np.broadcast_to(spectrum[None,:,None],(8,len(m.AXIS),4)).copy()
        return raw,spectra

    def test_complete_scope_and_all_windows_are_required(self):
        raw,spec=self.fixture()
        with self.assertRaises(ValueError):m.control_variate_acceptance(raw,raw,raw,spec,spec,spec,'paired')
        with self.assertRaises(ValueError):m.control_variate_acceptance(raw[:,:1],raw[:,:1],raw[:,:1],spec,spec,spec,'paired',complete_coherent_pair=True)
        bad=raw.copy();bad[:,m.WINDOW_NAMES.index('gain_125_160'),1]=[.01,.01,.01,.01,.01,.01,.01,10.]
        r=m.control_variate_acceptance(bad,raw,raw,spec,spec,spec,'paired',complete_coherent_pair=True,
            coarse_prefix_spectra=spec.mean(axis=0),numerical_status='NUMERICALLY_CONVERGED_DECLARED_TARGET')
        self.assertTrue(r['all_gates']['integrated_loss_all_detectors'])
        self.assertFalse(r['all_gates']['all_energy_windows_all_detectors'])
        self.assertFalse(r['all_gates']['declared_target_accepted'])

    def test_missing_prefix_prevents_shape_and_feature_pass_and_signed_correction_survives(self):
        raw,spec=self.fixture(); fine=spec.copy();fine[:,100:110,0]=-1
        r=m.control_variate_acceptance(raw,raw,raw,fine,spec,spec,'paired',complete_coherent_pair=True)
        self.assertFalse(r['all_gates']['shape']);self.assertFalse(r['all_gates']['features_identified_and_stable'])
        self.assertEqual(r['negative_spectral_bins'],10)
        np.testing.assert_allclose(r['mean_spectrum'][100:110,0],-1)

    def test_raw_vectors_do_not_mix_sign_sectors_or_lose_gain_windows(self):
        e=np.array([[100.,-100.,20.,-20.]])
        w=np.arange(1,17,dtype=float).reshape(1,4,4)
        r=m.raw_vector(e,w)
        np.testing.assert_allclose(r[0,m.WINDOW_NAMES.index('all_loss')],w[0,0]+w[0,2])
        np.testing.assert_allclose(r[0,m.WINDOW_NAMES.index('all_gain')],w[0,1]+w[0,3])
        np.testing.assert_allclose(r[0,m.WINDOW_NAMES.index('gain_90_125')],w[0,1])
        np.testing.assert_allclose(r[0,m.WINDOW_NAMES.index('loss_125_160')],0)

if __name__=='__main__':unittest.main()
