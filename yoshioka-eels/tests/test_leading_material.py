"""Actual-Si fixed-phonon-H epsilon validation and normalization controls."""
import unittest
from pathlib import Path

import numpy as np
from scipy import constants

from yoshioka.coupled_channels import channel_basis
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.leading_reference import (DenseDisplacedMaterial,analytic_phase_blocks,
    displacement_taylor_blocks,gamma_material_bath,normal_ordered_bath,
    quadrature_phase_blocks,thermal_initial_weights,finite_material_bath,repeated_material_bath)
from yoshioka.reciprocal_material import ReciprocalMaterialHamiltonian
from yoshioka.coordinate_channel_action import CoordinateMaterialAction

ROOT=Path(__file__).resolve().parents[1]


class LeadingMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        cls.h=DenseDisplacedMaterial(gamma_material_bath(cls.model),p=3,n=3,steps=17)
        cls.orders=cls.h.leading()

    def test_infinite_quadratic_product_has_identical_mode_and_exchange_factors(self):
        basis=channel_basis(2,2);where={tuple(s):i for i,s in enumerate(basis)}
        _,v1,v2=displacement_taylor_blocks(basis,[[.2,.3]])
        self.assertAlmostEqual(v2[0,where[(2,0)],0].real,-.2**2/np.sqrt(2))
        self.assertAlmostEqual(v2[0,where[(1,1)],0].real,-.2*.3)
        self.assertAlmostEqual(v2[0,where[(0,0)],0].real,-.5*(.2**2+.3**2))
        # The top retained state still includes its virtual n+1 contribution.
        self.assertAlmostEqual(v2[0,where[(2,0)],where[(2,0)]].real,-.5*(5*.2**2+.3**2))

    def test_closed_phase_matches_independent_quadrature(self):
        basis=channel_basis(2,4);xi=np.array([[.07,-.21],[.8,.4]])
        self.assertLess(np.linalg.norm(analytic_phase_blocks(basis,xi)-quadrature_phase_blocks(basis,xi,32)),2e-13)

    def test_actual_silicon_fixed_phonon_h_scaling_and_complex_controls(self):
        h=self.h;direct,sequential=self.orders[2:]
        reference=h.project_total(direct+sequential,2)
        previous=None
        for epsilon in (.5,.25,.125):
            exact=h.project_total(h.propagate(epsilon)/epsilon**2,2)
            error=np.linalg.norm(exact-reference)/np.linalg.norm(reference)
            if previous is not None: self.assertTrue(3.8<previous/error<4.2)
            previous=error
        self.assertLess(previous,2e-5)
        for linear,quadratic,target in ((False,True,direct),(True,False,sequential)):
            exact=h.project_total(h.propagate_taylor(.125,linear,quadratic)/.125**2,2)
            self.assertLess(np.linalg.norm(exact-h.project_total(target,2))/np.linalg.norm(h.project_total(target,2)),2e-4)
        # Both identical and distinct final occupation sectors carry signal.
        final=(direct+sequential).reshape(h.shape)
        for state in ((0,2),(1,1),(2,0)):
            index=next(i for i,s in enumerate(h.basis) if tuple(s)==state)
            self.assertGreater(np.linalg.norm(final[index]),1e-6)

    def test_actual_silicon_external_matrix_free_reference(self):
        old=ReciprocalMaterialHamiltonian(self.model,3,3,17)
        wave,_=old.propagate()
        self.assertLess(np.linalg.norm(wave.ravel()-self.h.propagate()),2e-11)

    def test_normal_order_counterterm_and_thermal_tail_are_explicit(self):
        bath,var=normal_ordered_bath(gamma_material_bath(self.model),300)
        h=DenseDisplacedMaterial(bath,3,3,17,normal_order_variances=var)
        self.assertLess(np.linalg.norm(h.propagate()-self.h.propagate()),2e-11)
        omega=np.array([.031,.071])*constants.elementary_charge/constants.hbar
        basis=channel_basis(2,6)
        weights,tail=thermal_initial_weights(omega,300,basis)
        self.assertAlmostEqual(float(np.sum(weights))+tail,1)
        self.assertGreater(tail,0)
        self.assertLess(float(np.sum(weights)),1)

    def test_non_gamma_matrix_free_action_and_explicit_longitudinal_rejection(self):
        bath=finite_material_bath(self.model,(2,1,1),(12,13))
        dense=DenseDisplacedMaterial(bath,p=3,n=3,steps=17)
        action=ReciprocalMaterialHamiltonian(self.model,3,3,17,bath=bath)
        wave,_=action.propagate()
        self.assertLess(np.linalg.norm(wave.ravel()-dense.propagate()),2e-11)
        coordinate=CoordinateMaterialAction(self.model,3,3,17,bath=bath)
        fft_wave,_=coordinate.propagate()
        self.assertLess(np.linalg.norm(fft_wave.ravel()-dense.propagate()),2e-11)
        scaled=CoordinateMaterialAction(self.model,3,3,17,bath=bath,displacement_scale=.125)
        scaled_wave,_=scaled.propagate()
        self.assertLess(np.linalg.norm(scaled_wave.ravel()-dense.propagate(.125)),2e-11)
        longitudinal=finite_material_bath(self.model,(2,1,1),(14,15))
        with self.assertRaisesRegex(ValueError,'transverse displacements only'):
            ReciprocalMaterialHamiltonian(self.model,2,3,17,bath=longitudinal)
        with self.assertRaisesRegex(ValueError,'transverse displacements only'):
            CoordinateMaterialAction(self.model,2,3,17,bath=longitudinal)

    def test_repeated_real_material_modes_and_longitudinal_dense_quadrature(self):
        bath=repeated_material_bath(self.model,2)
        self.assertLess(max(m['mechanical_eigen_residual'] for m in bath.labels),1e-10)
        self.assertEqual(len(bath.positions_m),16)
        longitudinal=finite_material_bath(self.model,(2,1,1),(14,15))
        dense=DenseDisplacedMaterial(longitudinal,p=2,n=3,steps=17)
        self.assertLess(np.linalg.norm(dense.propagate()-dense.propagate(method='quadrature',quadrature_order=24)),2e-11)
