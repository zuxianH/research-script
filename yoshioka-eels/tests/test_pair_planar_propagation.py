import unittest
import numpy as np
from scipy import constants
from yoshioka.free_propagation import TransverseGrid
from yoshioka.elastic_propagation import electron_speed
from yoshioka.pair_planar_propagation import propagate_planar_pair
from yoshioka.pair_galerkin import propagate_galerkin_pair,to_product


class PlanarPairPropagationTests(unittest.TestCase):
    def test_exact_contact_jump_and_target_energy_phase(self):
        g=TransverseGrid(5e-10,7e-10,3,3);p=TransverseGrid(g.lx_m,g.ly_m,5,5)
        length=2e-9;omega=np.array([.04,.065])*constants.elementary_charge/constants.hbar
        strength=np.array([1.,.8,.4,.6])*constants.elementary_charge*1e-12
        class Fields:
            contact_depths_m=np.array([.731e-9])
            quadratic_contact_fourier=np.zeros((1,4,5,5),complex)
            quadratic_contact_fourier[0,:,0,0]=strength
            def fields_at(self,z):return dict(background=np.zeros((5,5)),linear=np.zeros((2,5,5)),quadratic=np.zeros((4,5,5)))
        fields=Fields()
        r=propagate_planar_pair(g,p,omega,np.zeros((2,2)),length,19,fields,2)
        energies=constants.hbar*np.array([omega.sum(),-omega.sum(),omega[0]-omega[1],omega[1]-omega[0]])
        hv=constants.hbar*electron_speed(60000*constants.elementary_charge)
        expected=-1j*strength/hv*np.exp(-1j*energies*(length-fields.contact_depths_m[0])/hv)
        np.testing.assert_allclose(r['direct_amplitudes'][:,0,0],expected,rtol=1e-12,atol=1e-18)
        self.assertEqual(r['contact_planes_applied'],1)
        np.testing.assert_array_equal(r['sequential_amplitudes'],0)

    def test_same_projected_generator_without_contacts(self):
        g=TransverseGrid(8e-10,9e-10,3,3);p=TransverseGrid(g.lx_m,g.ly_m,5,5)
        rng=np.random.default_rng(113147);unit=constants.elementary_charge
        v=(rng.normal(size=(2,5,5))+1j*rng.normal(size=(2,5,5)))*unit
        q=(rng.normal(size=(4,5,5))+1j*rng.normal(size=(4,5,5)))*unit
        bg=rng.normal(size=(5,5))*unit;omega=np.array([4e13,9e13]);shifts=np.array([[3e9,-4e9],[-2e9,1e9]])
        class Fields:
            contact_depths_m=np.array([])
            def fields_at(self,z):return dict(background=bg,linear=v,quadratic=q)
        a=propagate_planar_pair(g,p,omega,shifts,1e-9,7,Fields(),2)
        b=propagate_galerkin_pair(g,p,omega,shifts,1e-9,7,lambda i:v,lambda i:q,lambda i:bg,2)
        for key in ('direct_amplitudes','sequential_amplitudes','total_amplitudes'):
            np.testing.assert_allclose(a[key],b[key],rtol=1e-13,atol=1e-16)

    def test_actual_planar_fields_with_longitudinal_contact(self):
        from yoshioka.pair_planar_fields import PairPlanarFields
        g=TransverseGrid(8e-10,9e-10,3,3);p=TransverseGrid(g.lx_m,g.ly_m,5,5)
        d=np.array([[[.003,.002],[.001,-.001],[.002,.004]]])*1e-10
        fields=PairPlanarFields(np.array([[1.2e-10,1.7e-10,8.3e-10]]),p,2e-9,d,np.zeros((2,2)),np.zeros((3,3)))
        r=propagate_planar_pair(g,p,np.array([4e13,9e13]),np.zeros((2,2)),2e-9,19,fields,2)
        self.assertEqual(r['contact_planes_applied'],1)
        self.assertTrue(np.all(np.isfinite(r['total_amplitudes'])))
        self.assertGreater(np.linalg.norm(r['direct_amplitudes']),0)

    def test_exact_rigid_translation_identity_with_interval_means(self):
        from yoshioka.pair_planar_averages import AveragedPairPlanarFields
        from yoshioka.dynamical_source import full_transverse_vectors
        g=TransverseGrid(8e-10,9e-10,5,5);p=TransverseGrid(g.lx_m,g.ly_m,9,9)
        positions=np.array([[1.2e-10,1.7e-10,8.3e-10],[3.1e-10,4.9e-10,1.11e-9]])
        d=np.zeros((2,3,2));d[:,0,0]=.003e-10;d[:,1,1]=.002e-10
        fields=AveragedPairPlanarFields(positions,p,2e-9,d,np.zeros((2,2)),np.zeros((3,3)))
        r=propagate_planar_pair(g,p,np.zeros(2),np.zeros((2,2)),2e-9,19,fields,2)
        Q=full_transverse_vectors(g)
        expected=-(Q[:,0]*d[0,0,0])*(Q[:,1]*d[0,1,1])*r['word_coefficients'][0].ravel()
        for a in r['total_amplitudes']:
            np.testing.assert_allclose(a.ravel(),expected,rtol=1e-11,atol=1e-17)


if __name__=='__main__':unittest.main()
