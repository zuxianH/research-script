"""Independent unit/force/normalization checks for Benchmark 5."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.constants import Boltzmann, elementary_charge, hbar

from benchmarks.workflow_io import ROOT, sha256
from yoshioka.lattice_dynamics import RealSpaceForceConstants, covariance_mesh, finite_periodic_modes, real_space_from_supercell
from yoshioka.mod_forces import central_force_constants, complete_force_constants, correct_sum_rules, force_constant_residuals, potential_metadata


class HarmonicLatticeUnitTests(unittest.TestCase):
    def test_angular_fourier_convention_for_known_spring_chain(self):
        k, mass = 3.0, 2.0
        model = RealSpaceForceConstants(np.eye(3), np.zeros((1,3)), np.array([mass]), np.zeros(3,dtype=int), np.zeros(3,dtype=int),
                                       np.array([[0,0,0],[1,0,0],[-1,0,0]]), np.array([2*k*np.eye(3),-k*np.eye(3),-k*np.eye(3)]))
        for q in (0, .1, .25, .5):
            expected = 2*k*(1-np.cos(2*np.pi*q))/mass
            np.testing.assert_allclose(model.dynamical([q,0,0]), expected*np.eye(3), atol=1e-14)
        np.testing.assert_allclose(model.reciprocal_m_inv, 2*np.pi*np.eye(3))

    def test_translation_reconstruction_and_sum_rule_projection(self):
        columns=np.array([2*np.eye(3),-np.eye(3),-np.eye(3)]).reshape(3,1,3,1,3)
        fc=complete_force_constants(columns,(3,1,1))
        self.assertLess(force_constant_residuals(fc)['permutation'],1e-15)
        self.assertLess(force_constant_residuals(fc)['translational'],1e-15)
        rng=np.random.default_rng(902)
        noisy=fc+rng.normal(scale=.001,size=fc.shape)
        corrected=correct_sum_rules(noisy)
        self.assertLess(force_constant_residuals(corrected)['permutation'],1e-14)
        self.assertLess(force_constant_residuals(corrected)['translational'],1e-14)
        np.testing.assert_allclose(correct_sum_rules(corrected),corrected,atol=1e-14)

    def test_actual_potential_metadata(self):
        data=potential_metadata(ROOT/'Si.tersoff.mod')
        self.assertEqual(data['pair_style'],'tersoff/mod')
        self.assertAlmostEqual(data['pair_cutoff_angstrom'],3.3)
        self.assertEqual(data['parameters_metal_units']['beta_ters'],1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ordinary.tersoff';path.write_text('Si Si Si 1 2 3')
            with self.assertRaises(ValueError):potential_metadata(path)


LATTICE_AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ('lammps','hiphive','phonopy'))


@unittest.skipUnless(LATTICE_AVAILABLE,'install requirements-lattice.txt for independent MOD/hiPhive checks')
class MODLatticeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmarks.abtem_reference import load_structure
        cls.input_hash=sha256(ROOT/'Si_110_temp300.lmp')
        cls.potential_hash=sha256(ROOT/'Si.tersoff.mod')
        cls.unit,_=load_structure(ROOT/'Si_110_temp300.lmp')
        a=5.429000041011736
        cls.unit.set_cell(np.diag([a,a/np.sqrt(2),a/np.sqrt(2)]),scale_atoms=True)
        cls.reps=(2,3,3)
        columns,cls.supercell,_=central_force_constants(cls.unit,cls.reps,.0025,ROOT/'Si.tersoff.mod')
        cls.raw=complete_force_constants(columns,cls.reps)
        cls.fc=correct_sum_rules(cls.raw)
        cls.model,cls.range=real_space_from_supercell(cls.unit,cls.reps,cls.fc)

    def test_actual_range_symmetry_and_translation_modes(self):
        self.assertGreater(self.range['maximum_nonzero_range_angstrom'],3.3)
        self.assertLess(self.range['maximum_nonzero_range_angstrom'],4)
        self.assertLess(force_constant_residuals(self.fc)['translational'],1e-12)
        q=np.array([[0,0,0],[.1,.2,.3],[.5,0,0]])
        d=self.model.dynamical(q)
        self.assertLess(np.linalg.norm(d-d.swapaxes(-1,-2).conj())/np.linalg.norm(d),1e-12)
        energies=self.model.modes(q)[0]*hbar/(1e-3*elementary_charge)
        self.assertEqual(sum(abs(energies[0])<1e-3),3)
        self.assertGreater(np.min(energies[1:]),0)
        self.assertEqual(sha256(ROOT/'Si_110_temp300.lmp'),self.input_hash)
        self.assertEqual(sha256(ROOT/'Si.tersoff.mod'),self.potential_hash)

    def test_full_model_covariance_equals_q_weighted_covariance(self):
        modes=finite_periodic_modes(self.model,(2,2,2))
        self.assertEqual(modes['removed_rigid_mode_count'],3)
        self.assertEqual(modes['eigenvectors'].shape,(32,3,93))
        np.testing.assert_allclose(np.sum(abs(modes['eigenvectors'])**2,axis=(0,1)),1,atol=2e-14)
        qcov,_=covariance_mesh(self.model,2,(0,300))
        d=modes['displacements_m']
        for index,temperature in enumerate((0,300)):
            factor=np.ones(93)
            if temperature:
                x=hbar*modes['omega_rad_s']/(Boltzmann*temperature)
                factor=1+2*np.exp(-x)/(-np.expm1(-x))
            full=np.einsum('iam,ibm,m->iab',d,d.conj(),factor).reshape(8,4,3,3).mean(0)
            self.assertLess(np.linalg.norm(full-qcov[index])/np.linalg.norm(qcov[index]),1e-11)

    def test_independent_hiphive_fit_and_heldout_forces(self):
        from hiphive import ClusterSpace
        from benchmarks.si_lattice_dynamics import _paired_dataset,_fit,_heldout
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):cs=ClusterSpace(self.unit,[4.0])
            reference,train=_paired_dataset(self.unit,self.reps,.001,4,179,ROOT/'Si.tersoff.mod',Path(directory))
            _,validation=_paired_dataset(self.unit,self.reps,.001,3,893,ROOT/'Si.tersoff.mod',Path(directory))
            potential,_=_fit(cs,reference,train,4,0)
            errors,fc=_heldout(potential,reference,validation,self.fc)
            self.assertLess(errors['heldout_relative_force_rms_error'],.01)
            self.assertLess(np.linalg.norm(fc-self.fc)/np.linalg.norm(self.fc),1e-3)

    def test_phonopy_phase_units_and_primitive_folding(self):
        from benchmarks.si_lattice_dynamics import _phonopy_validation
        with tempfile.TemporaryDirectory() as directory:
            errors=_phonopy_validation(self.unit,self.reps,self.fc,self.model,np.array([[0,0,0],[.13,.21,.3]]),Path(directory))
            self.assertLess(errors['maximum_energy_difference_mev'],1e-3)
            self.assertLess(errors['maximum_degenerate_projector_error'],1e-6)
            self.assertEqual(errors['primitive_atom_count'],2)
            self.assertEqual(errors['oriented_atom_count'],4)
            self.assertLess(errors['folded_six_plus_six_vs_twelve_maximum_energy_error_mev'],1e-3)
            reconstructed=np.array(errors['primitive_cartesian_positions_angstrom'])[errors['oriented_atom_to_primitive_basis']]+np.array(errors['oriented_atom_primitive_lattice_translation'])@np.array(errors['primitive_cell_angstrom'])
            np.testing.assert_allclose(reconstructed,self.unit.positions,atol=1e-7,rtol=0)


if __name__=='__main__':unittest.main()
