import unittest
import numpy as np
from benchmarks.complete_acceptance import evaluate,paired_ratio
from benchmarks.two_phonon_analysis import WINDOWS

class CompleteAcceptanceTests(unittest.TestCase):
    def fixture(self):
        return {k:{'mean':[1.]*4,'halfwidth95':[.01]*4} for k in WINDOWS},[{'shape_pass':True,'all_peak_gate_pass':True} for _ in range(4)]
    def test_loss_pass_cannot_hide_failed_window_or_gain(self):
        raw,shape=self.fixture();raw['loss_125_160']['halfwidth95'][2]=.2
        gates=evaluate(raw,shape,'NUMERICALLY_CONVERGED_DECLARED_TARGET')
        self.assertTrue(gates['integrated_loss_all_detectors']);self.assertFalse(gates['all_energy_windows_all_detectors']);self.assertFalse(gates['declared_target_accepted'])
        raw,shape=self.fixture();raw['all_gain']['halfwidth95'][0]=.2
        self.assertFalse(evaluate(raw,shape,'NUMERICALLY_CONVERGED_DECLARED_TARGET')['declared_target_accepted'])
    def test_statistical_pass_does_not_promote_missing_numerical_convergence(self):
        raw,shape=self.fixture();a=evaluate(raw,shape)
        self.assertTrue(a['statistical_gates_passed']);self.assertFalse(a['declared_target_accepted'])
        self.assertTrue(evaluate(raw,shape,'NUMERICALLY_CONVERGED_DECLARED_TARGET')['declared_target_accepted'])
        shape[1]['all_peak_gate_pass']=False
        self.assertFalse(evaluate(raw,shape,'NUMERICALLY_CONVERGED_DECLARED_TARGET')['declared_target_accepted'])
    def test_ratio_retains_shared_sample_covariance(self):
        y=np.array([1.,2.,3.,4.]);x=2*y
        result=paired_ratio(x,y)
        self.assertEqual(result['ratio'],2);self.assertEqual(result['standard_error'],0)
        self.assertGreater(result['numerator_denominator_batch_covariance'],0)

if __name__=='__main__':unittest.main()
