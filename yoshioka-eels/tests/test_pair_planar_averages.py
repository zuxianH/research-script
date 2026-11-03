"""Regular interval means, contact separation, and actual field assembly."""
import unittest
import numpy as np
from numpy.polynomial.legendre import leggauss

from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.free_propagation import TransverseGrid
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.pair_planar_fields import PairPlanarFields,planar_lobato_kernels,planar_second_derivative_contact
from yoshioka.pair_planar_averages import AveragedPairPlanarFields,averaged_planar_lobato_kernels


class PlanarAveragesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.potential=LobatoSilicon.from_abtem_table()

    def test_high_Q_near_plane_means_against_independent_gauss_quadrature(self):
        period=8e-10;width=.1e-10
        q2=np.array([0.,1.,100.,10000.])*1e20
        offsets=np.array([-1.9,-.2,-.1,0.,1.3])*1e-10
        actual=averaged_planar_lobato_kernels(q2,offsets,width,period,self.potential)
        root,weight=leggauss(192)
        expected=[]
        for offset in offsets:
            z=offset+(root+1)*width/2
            kernels=planar_lobato_kernels(q2,z,self.potential,period)
            expected.append([weight@h/2 for h in kernels])
        for derivative in range(3):
            wanted=np.array([value[derivative] for value in expected])
            self.assertLess(np.linalg.norm(actual[derivative]-wanted)/np.linalg.norm(wanted),2e-12)

    def test_full_period_zeroth_moments_and_delta_are_separate(self):
        period=3.1e-10;q2=np.array([0.,1.,10.,10000.])*1e20
        h,h1,h2=averaged_planar_lobato_kernels(q2,0.,period,period,self.potential)
        np.testing.assert_allclose(h,self.potential.fourier_energy_j_m3(q2)/period,rtol=1e-13,atol=0.)
        np.testing.assert_array_equal(h1,0.)
        contact=planar_second_derivative_contact(self.potential)
        np.testing.assert_allclose(h2,-contact/period,rtol=1e-13,atol=0.)

    def test_tiny_interval_one_sided_limit_and_plane_crossing_rejection(self):
        period=8e-10;width=1e-22;q2=np.array([0.,1.,10000.])*1e20
        h,h1,h2=averaged_planar_lobato_kernels(q2,0.,width,period,self.potential)
        at=planar_lobato_kernels(q2,0.,self.potential,period)
        self.assertLess(np.linalg.norm(h-at[0])/np.linalg.norm(at[0]),2e-8)
        self.assertLess(np.linalg.norm(h2-at[2])/np.linalg.norm(at[2]),2e-8)
        # The regular h' is discontinuous: the interval starting at the
        # plane tends to its RIGHT limit, not the symmetric point value0.
        contact=planar_second_derivative_contact(self.potential)
        np.testing.assert_allclose(h1,contact/2,rtol=2e-8,atol=0.)
        with self.assertRaises(ValueError):
            averaged_planar_lobato_kernels(q2,-.02e-10,.1e-10,period,self.potential)

    def test_finite_atom_assembly_and_transverse_translation_derivatives(self):
        grid=TransverseGrid(5e-10,4e-10,5,5);period=6e-10
        positions=np.array([[0.,0.,1.2e-10],[2.5e-10,2e-10,2.7e-10]])
        rng=np.random.default_rng(20341)
        d=(rng.normal(size=(2,3,2))+1j*rng.normal(size=(2,3,2)))*1e-12
        shifts=np.array([[.17,-.23],[-.31,.19]])*1e10
        mean=AveragedPairPlanarFields(positions,grid,period,d,shifts,potential=self.potential,chunk=7)
        base=PairPlanarFields(positions,grid,period,d,shifts,potential=self.potential,chunk=7)
        left,right=positions[:,2]
        result=mean.averaged_fourier(left,right)
        root,weight=leggauss(192)
        reference={key:np.zeros_like(value) for key,value in result.items()}
        for r,w in zip(root,weight):
            at=base.regular_fourier((left+right)/2+(right-left)*r/2)
            for key in reference:reference[key]+=w*at[key]/2
        for key in result:
            self.assertLess(np.linalg.norm(result[key]-reference[key])/np.linalg.norm(reference[key]),2e-12)
        np.testing.assert_array_equal(mean.quadratic_contact_fourier,base.quadratic_contact_fourier)
        with self.assertRaises(ValueError):mean.averaged_fourier(0.,period)
        rigid=np.zeros_like(d);rigid[:,0,0]=.01e-10;rigid[:,1,1]=.015e-10
        fields=AveragedPairPlanarFields(positions,grid,period,rigid,np.zeros((2,2)),potential=self.potential)
        coefficients=fields.averaged_fourier(left,right)
        Q=full_transverse_vectors(grid).reshape(*grid.shape,2)
        a=Q[...,0]*rigid[0,0,0];b=Q[...,1]*rigid[0,1,1]
        np.testing.assert_allclose(coefficients['linear'][0],-1j*a*coefficients['background'],rtol=2e-13,atol=1e-36)
        np.testing.assert_allclose(coefficients['linear'][1],-1j*b*coefficients['background'],rtol=2e-13,atol=1e-36)
        for value in coefficients['quadratic']:
            np.testing.assert_allclose(value,-a*b*coefficients['background'],rtol=2e-13,atol=1e-36)
        np.testing.assert_array_equal(fields.quadratic_contact_fourier,0.)


if __name__=='__main__':unittest.main()
