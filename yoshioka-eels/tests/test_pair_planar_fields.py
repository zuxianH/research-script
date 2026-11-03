"""Independent transform, weak-distribution and displaced-atom checks."""
import unittest
import numpy as np
from scipy import fft
from numpy.polynomial.legendre import leggauss

from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.free_propagation import TransverseGrid
from yoshioka.pair_planar_fields import (
    planar_lobato_kernels,planar_second_derivative_contact,
    PlanarDerivativeField,PairPlanarFields,
)


def nodes_weights(boundaries,order=256):
    roots,weights=leggauss(order)
    x=[];w=[]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        x.append((lo+hi)/2+(hi-lo)/2*roots)
        w.append((hi-lo)/2*weights)
    return np.concatenate(x),np.concatenate(w)


class PlanarPairFieldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.potential=LobatoSilicon.from_abtem_table()

    def test_exact_periodic_image_sum_against_many_infinite_line_images(self):
        period=1e-10
        q2=np.array([0.,1.,7.])*1e20
        offsets=np.array([-1.,-.49,.25,1.1])*period
        actual=planar_lobato_kernels(q2,offsets,self.potential,period)
        images=[planar_lobato_kernels(q2,offsets+j*period,self.potential) for j in range(-30,31)]
        for order in range(3):
            expected=sum(image[order] for image in images)
            self.assertLess(np.linalg.norm(actual[order]-expected)/np.linalg.norm(expected),2e-13)

    def test_regular_plus_contact_fourier_transform_is_original_3d_lobato(self):
        period=8e-10;atom=2.7e-10
        z,w=nodes_weights([0.,atom,period])
        q2=np.array([0.,1.,9.])*1e20
        kernels=planar_lobato_kernels(q2,z-atom,self.potential,period)
        contact=planar_second_derivative_contact(self.potential)
        errors=[]
        for mode in (0,1,7,31):
            kz=2*np.pi*mode/period
            phase=np.exp(-1j*kz*z)
            h=self.potential.fourier_energy_j_m3(q2+kz*kz)*np.exp(-1j*kz*atom)
            expected=(h,1j*kz*h,-kz*kz*h)
            for derivative in range(3):
                actual=(w*phase)@kernels[derivative]
                if derivative==2:actual+=contact*np.exp(-1j*kz*atom)
                # A zero derivative coefficient is checked against a physical
                # component scale rather than a relative error to zero.
                scale=max(np.max(abs(expected[derivative])),abs(contact)*(period**(2-derivative)))
                errors.append(np.max(abs(actual-expected[derivative]))/scale)
        self.assertLess(max(errors),2e-10)

    def test_huge_z_fft_weak_integral_requires_the_delta_contact(self):
        period=8e-10;atom=2.73e-10;q2=1.21e20
        center=atom+.13e-10;concentration=18.
        def test(z):return np.exp(concentration*(np.cos(2*np.pi*(z-center)/period)-1))
        z,w=nodes_weights([0.,atom,period])
        regular=planar_lobato_kernels(q2,z-atom,self.potential,period)[2]
        regular_integral=np.dot(w,regular*test(z))
        contact=planar_second_derivative_contact(self.potential)*test(atom)
        expected=regular_integral+contact
        for samples in (4097,32769):
            dz=period/samples;kz=2*np.pi*np.fft.fftfreq(samples,dz)
            coefficients=-kz*kz*self.potential.fourier_energy_j_m3(q2+kz*kz)*np.exp(-1j*kz*(atom-dz/2))/period
            values=fft.ifft(coefficients)*samples
            actual=dz*np.sum(values*test((np.arange(samples)+.5)*dz))
            self.assertLess(abs(actual-expected)/abs(expected),2e-10)
            self.assertGreater(abs(actual-regular_integral)/abs(expected),.1)

    def test_actual_displaced_atom_weak_derivatives_include_mixed_and_zz_terms(self):
        period=8e-10;atom=2.71e-10;grid=TransverseGrid(5e-10,4e-10,3,3)
        position=np.array([[.13e-10,.27e-10,atom]])
        first=np.array([[.02,.01,.015]])*1e-10
        second=np.array([[-.012,.018,.02]])*1e-10
        shift=np.array([.17,-.23])*1e10
        linear=PlanarDerivativeField(position,grid,period,shift,first=first,potential=self.potential)
        quadratic=PlanarDerivativeField(position,grid,period,shift,first=first,second=second,potential=self.potential)
        # Fixed physical output G=(1,0), no detector interpolation or fit.
        at=(0,1);Q=linear.qxy.reshape(3,3,2)[at];q2=np.dot(Q,Q)
        def test(z):return np.exp(14*(np.cos(2*np.pi*(z-atom-.17e-10)/period)-1))
        z,w=nodes_weights([0.,atom,period])
        leading1=sum(weight*test(zp)*linear.regular_fourier(zp)[at] for zp,weight in zip(z,w))
        leading2=sum(weight*test(zp)*quadratic.regular_fourier(zp)[at] for zp,weight in zip(z,w))
        leading2+=quadratic.contact_fourier[0][at]*test(atom)
        def displaced(u):
            shifted=atom+u[2]
            zz,ww=nodes_weights([0.,shifted,period])
            h=planar_lobato_kernels(q2,zz-shifted,self.potential,period)[0]
            return np.exp(-1j*np.dot(Q,position[0,:2]+u[:2]))*np.dot(ww,h*test(zz))/(grid.lx_m*grid.ly_m)
        errors1=[];errors2=[]
        for epsilon in (.5,.25,.125,.0625):
            a=epsilon*first[0];b=epsilon*second[0]
            actual1=(displaced(a)-displaced(-a))/(2*epsilon)
            actual2=(displaced(a+b)-displaced(a-b)-displaced(-a+b)+displaced(-a-b))/(4*epsilon**2)
            errors1.append(abs(actual1-leading1)/abs(leading1))
            errors2.append(abs(actual2-leading2)/abs(leading2))
        self.assertLess(errors1[-1],2e-5)
        self.assertLess(errors2[-1],2e-5)
        self.assertGreater(errors1[0]/errors1[-1],40)
        self.assertGreater(errors2[0]/errors2[-1],40)

    def test_pair_complex_conjugation_periodicity_and_explicit_covariance_rejection(self):
        grid=TransverseGrid(5e-10,4e-10,5,5);period=7e-10
        positions=np.array([[0.,0.,1.3e-10],[2.5e-10,2e-10,3.7e-10]])
        rng=np.random.default_rng(1341)
        d=(rng.normal(size=(2,3,2))+1j*rng.normal(size=(2,3,2)))*1e-12
        shifts=np.array([[.17,-.23],[-.31,.19]])*1e10
        fields=PairPlanarFields(positions,grid,period,d,shifts,potential=self.potential,chunk=7)
        a=fields.fields_at(2.17e-10);b=fields.fields_at(2.17e-10+period)
        for key in a:np.testing.assert_allclose(a[key],b[key],rtol=2e-12,atol=1e-35)
        np.testing.assert_allclose(a['quadratic'][0].conj(),a['quadratic'][1],rtol=2e-12,atol=1e-35)
        np.testing.assert_allclose(a['quadratic'][2].conj(),a['quadratic'][3],rtol=2e-12,atol=1e-35)
        for plane in range(len(fields.contact_depths_m)):
            contact=fields.contact_field(plane)
            np.testing.assert_allclose(contact[0].conj(),contact[1],rtol=2e-12,atol=1e-45)
            np.testing.assert_allclose(contact[2].conj(),contact[3],rtol=2e-12,atol=1e-45)
        with self.assertRaises(ValueError):
            PairPlanarFields(positions,grid,period,d,shifts,covariance=np.eye(3)*1e-24,potential=self.potential)


if __name__=='__main__':unittest.main()
