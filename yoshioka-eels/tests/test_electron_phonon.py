"""Atomic Fourier, Bloch, source integration, and independent Born tests."""

import unittest

import numpy as np
from scipy.constants import elementary_charge,hbar

from benchmarks.born_reference import finite_window_born,single_atom_fourier_audit
from benchmarks.eels_observables import detector_masks
from benchmarks.one_phonon_born import BornConfig
from yoshioka.electron_phonon import (FirstOrderSourcePropagator,LobatoSilicon,derivative_symbol,fourth_order_derivative,full_space_born,numerical_transition_source,transverse_modes)
from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber


class ElectronPhononTests(unittest.TestCase):
    def setUp(self):
        self.energy=60000*elementary_charge
        self.grid=TransverseGrid(12e-10,10e-10,128,128)

    def test_fourier_potential_units_charge_sign_and_angular_wavevector(self):
        potential=LobatoSilicon.from_abtem_table()
        audit=single_atom_fourier_audit(potential)
        self.assertLess(audit['maximum_relative_error'],1e-10)
        self.assertTrue(all(row['negative_interaction_energy'] for row in audit['tests']))
        self.assertLess(potential.radial_energy_j(1e-10),0)

    def test_fourth_order_gradient_symbol_and_convergence(self):
        errors=[]
        for n in (16,32,64):
            z=np.arange(n)/n
            field=np.exp(2j*np.pi*3*z)
            derivative=fourth_order_derivative(field,1/n)
            np.testing.assert_allclose(derivative,derivative_symbol(6*np.pi,1/n)*field,atol=2e-13)
            errors.append(np.linalg.norm(derivative-6j*np.pi*field)/np.linalg.norm(6*np.pi*field))
        self.assertLess(errors[-1],errors[-2]/10)

    def test_actual_longitudinal_solver_against_known_gaussian_pulse(self):
        n=512;length=20e-10;z=np.arange(n)*length/n;sigma=1e-10;center=length/2
        omega=np.array([30e-3*elementary_charge/hbar,60e-3*elementary_charge/hbar])
        indices=np.array([0,1,2]);q=np.array([0,2*np.pi/self.grid.lx_m,4*np.pi/self.grid.lx_m])
        amplitude=np.array([[1,2j,-1],[2,1j,3]])*1e-22
        source=amplitude[:,:,None]*np.exp(-.5*((z-center)/sigma)**2)
        propagator=FirstOrderSourcePropagator(self.grid,self.energy,omega,indices)
        numerical,_=propagator.propagate(source,length)
        kappa=q[None]**2/(2*electron_wavenumber(self.energy))+omega[:,None]/electron_speed(self.energy)
        reference=-1j/(hbar*electron_speed(self.energy))*amplitude*np.sqrt(2*np.pi)*sigma*np.exp(-.5*(kappa*sigma)**2)*np.exp(-1j*kappa*(length-center))
        self.assertLess(np.linalg.norm(numerical-reference)/np.linalg.norm(reference),1e-10)
        zero,_=propagator.propagate(source,length,g=0)
        np.testing.assert_array_equal(zero,0)

    def test_bloch_boundary_phase_and_decoupled_momentum_subset(self):
        grid=TransverseGrid(12e-10,10e-10,16,16)
        shift=np.array([.37*2*np.pi/grid.lx_m,-.21*2*np.pi/grid.ly_m])
        indices=np.array([0,1,17]);omega=np.array([6e13]);distance=2e-10
        propagator=FirstOrderSourcePropagator(grid,self.energy,omega,indices,shift)
        qx,qy=np.meshgrid(2*np.pi*np.fft.fftfreq(grid.nx,grid.dx_m),2*np.pi*np.fft.fftfreq(grid.ny,grid.dy_m))
        physical=np.stack((qx,qy),axis=-1).reshape(-1,2)[indices]+shift
        expected=np.exp(-1j*distance*(np.sum(physical**2,axis=-1)/(2*electron_wavenumber(self.energy))+omega[0]/electron_speed(self.energy)))
        np.testing.assert_allclose(propagator.multiplier(distance)[0],expected,atol=1e-15)
        rng=np.random.default_rng(139)
        full_source=rng.normal(size=(1,256,32))*1e-22
        restricted,_=propagator.propagate(full_source[:,indices],distance)
        full,_=FirstOrderSourcePropagator(grid,self.energy,omega,np.arange(256),shift).propagate(full_source,distance)
        np.testing.assert_allclose(restricted,full[:,indices],atol=1e-16,rtol=0)
        scanned=propagator.propagate_scan(full_source[:,indices],distance)
        np.testing.assert_allclose(scanned,restricted,atol=1e-16,rtol=1e-12)

    def test_independent_finite_window_reference_including_longitudinal_coupling(self):
        potential=LobatoSilicon.from_abtem_table();length=24e-10
        positions=np.array([[3e-10,4e-10,12e-10]])
        displacements=np.zeros((1,3,3),complex)
        displacements[0]=np.eye(3)*4e-12
        covariance=np.eye(3)*1.8e-23
        q=np.array([[0,0],[2*np.pi/self.grid.lx_m,0]])
        indices=np.array([0,1]);omega=np.ones(3)*60e-3*elementary_charge/hbar
        _,source=numerical_transition_source(potential,positions,displacements,covariance,q,self.grid,length,2048)
        numerical,_=FirstOrderSourcePropagator(self.grid,self.energy,omega,indices).propagate(source,length)
        finite,_=finite_window_born(potential,positions,displacements,covariance,q,omega,self.energy,self.grid.lx_m*self.grid.ly_m,length)
        full=full_space_born(potential,positions,displacements,covariance,q,omega,self.energy,self.grid.lx_m*self.grid.ly_m,length)
        self.assertLess(np.linalg.norm(numerical-finite)/np.linalg.norm(finite),1e-4)
        self.assertLess(np.linalg.norm(finite-full)/np.linalg.norm(full),1e-9)
        self.assertGreater(abs(numerical[2,0]),1e-10)
        self.assertLess(abs(numerical[2,0]-full[2,0])/abs(full[2,0]),1e-7)

    def test_fixed_detector_partition_and_invalid_configuration(self):
        masks=detector_masks(np.array([0,5,10,20]))
        partition=sum(masks[name].astype(int) for name in ('central_0_5_mrad','annulus_5_10_mrad','annulus_10_20_mrad'))
        np.testing.assert_array_equal(partition,np.ones(4))
        for values in ({'temperature_k':300},{'kinetic_energy_ev':0},{'maximum_angle_mrad':15},{'g_values':(.5,1)},{'longitudinal_samples':(31,)},{'selected_grid':256}):
            with self.assertRaises(ValueError):BornConfig(**values)


if __name__=='__main__':unittest.main()
