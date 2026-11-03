"""Finite-reference FFT derivatives and non-vacuum transition definitions."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pair_finite_validity',ROOT/'benchmarks/14_two_phonon_eels/pair_finite_validity.py')
fv=importlib.util.module_from_spec(spec);spec.loader.exec_module(fv)
pspec=importlib.util.spec_from_file_location('finite_pruned',ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/finite_validity/pruned_driver.py')
pruned=importlib.util.module_from_spec(pspec);pspec.loader.exec_module(pruned)
sys.path.insert(0,str(ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/finite_validity'))
gspec=importlib.util.spec_from_file_location('finite_grouped',ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/finite_validity/grouped_driver.py')
grouped=importlib.util.module_from_spec(gspec);gspec.loader.exec_module(grouped)
fspec=importlib.util.spec_from_file_location('finite_fixed',ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/finite_validity/fixed_bath_driver.py')
fixed=importlib.util.module_from_spec(fspec);fspec.loader.exec_module(fixed)


class FiniteValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=fv.RealSpaceForceConstants.load(fv.MODEL)
        cls.bath=fv.single_gamma_bath(cls.model)
        cls.h=fv.CoordinateMaterialAction(cls.model,4,3,17,bath=cls.bath,quadrature_order=18)
        cls.t=fv.TaylorFFT(cls.h)
        cls.dense=fv.DenseDisplacedMaterial(cls.bath,4,3,17)

    def test_actual_si_derivatives_and_leading_amplitudes(self):
        rng=np.random.default_rng(51613);x=rng.normal(size=self.h.shape)+1j*rng.normal(size=self.h.shape)
        derivatives=self.dense.derivative_hamiltonians()
        for order in range(3):
            expected=derivatives[order,8]@x.ravel()
            self.assertLess(np.linalg.norm(self.t.apply_order(x,8,order).ravel()-expected)/np.linalg.norm(expected),2e-12)
        wave=np.zeros((4,*self.h.shape),complex);wave[0,0,1,1]=1
        for i in range(17):wave=self.t.step(wave,i)
        self.assertLess(np.linalg.norm(wave.reshape(4,-1)-self.dense.leading()),2e-12)

    def test_augmented_adjoint(self):
        rng=np.random.default_rng(1721);a=rng.normal(size=(4,*self.h.shape))+1j*rng.normal(size=(4,*self.h.shape))
        b=rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
        lhs=np.vdot(a,self.t.augmented(b,8));rhs=np.vdot(self.t.augmented(a,8,True),b)
        self.assertLess(abs(lhs-rhs)/max(abs(lhs),abs(rhs)),2e-12)

    def test_changed_two_pruning_keeps_complex_direct_and_sequential(self):
        action=pruned.PrunedTaylorFFT(self.h);wave=np.zeros((4,*self.h.shape),complex);wave[0,0,1,1]=1
        for i in range(self.h.steps):wave=action.step(wave,i)
        reference=self.dense.leading().reshape((4,*self.h.shape))
        where=np.flatnonzero(self.h.basis[:,0]==2)[0]
        self.assertLess(np.linalg.norm(wave[2:,where]-reference[2:,where]),2e-12)
        # The checked target is changed occupation; disconnected elastic returns
        # are excluded explicitly rather than passed through this pruning.
        invalid=wave.copy();invalid[0,1,1,1]=1
        with self.assertRaisesRegex(ValueError,'non-vacuum'):
            action.step(invalid,0)

    def test_material_n1_initial_uses_relative_transition_class(self):
        initial=self.dense.initial_state([1]);full=self.dense.propagate(initial=initial)
        orders=self.dense.leading(initial=initial)
        obs=fv.observables(self.h,full.reshape(self.h.shape),orders.reshape(4,*self.h.shape),[1])
        selected=self.dense.basis[:,0]==3
        expected=float(np.sum(abs(full.reshape(self.h.shape)[selected])**2))
        self.assertAlmostEqual(obs['all_retained_beams']['full'],expected,places=22)
        wrong=float(np.sum(abs(full.reshape(self.h.shape)[self.dense.basis[:,0]==2])**2))
        self.assertGreater(wrong/expected,100)

    def test_repeated_mode_normalization_and_covariance_bookkeeping(self):
        repeated=fv.single_gamma_bath(self.model,4)
        self.assertAlmostEqual(np.linalg.norm(repeated.displacement_m)/np.linalg.norm(self.bath.displacement_m),1)
        full1=self.bath.inactive_covariance_m2+np.einsum('iam,ibm->iab',self.bath.displacement_m,self.bath.displacement_m)
        full4=repeated.inactive_covariance_m2+np.einsum('iam,ibm->iab',repeated.displacement_m,repeated.displacement_m)
        self.assertLess(np.max(abs(full4-np.tile(full1,(4,1,1)))),1e-36)

    def test_grouped_atom_profiles_preserve_repeated_material_operator(self):
        bath=fv.single_gamma_bath(self.model,4)
        original=fv.CoordinateMaterialAction(self.model,3,3,17,bath=bath,quadrature_order=16)
        action=grouped.GroupedCoordinateAction(self.model,3,3,17,bath=bath,quadrature_order=16)
        rng=np.random.default_rng(876);x=rng.normal(size=original.shape)+1j*rng.normal(size=original.shape)
        self.assertEqual(len(action.profiles),4)
        expected=original.apply(x,8)
        self.assertLess(np.linalg.norm(action.apply(x,8)-expected)/np.linalg.norm(expected),2e-12)

    def test_primary_thickness_windows_keep_the_same_canonical_bath(self):
        full=fixed.fixed_bath(self.model,4)
        for cells in (1,2,4):
            part=fixed.fixed_bath(self.model,cells);count=len(part.positions_m)
            self.assertTrue(np.array_equal(part.displacement_m,full.displacement_m[:count]))
            self.assertTrue(np.array_equal(part.inactive_covariance_m2,full.inactive_covariance_m2[:count]))
            self.assertTrue(np.array_equal(part.omega_rad_s,full.omega_rad_s))
            self.assertAlmostEqual(np.linalg.norm(part.displacement_m)/np.linalg.norm(full.displacement_m),np.sqrt(cells/4))


if __name__=='__main__':unittest.main()
