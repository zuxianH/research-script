import tempfile,unittest
from pathlib import Path
import numpy as np
from scipy import constants
from yoshioka.free_propagation import TransverseGrid
from yoshioka.pair_galerkin import propagate_galerkin_pair,to_product,frequency_indices,pair_structure
from yoshioka.pair_intermediate_momentum import (integer_beams,propagate_intermediate_resolved,
    cumulative_amplitudes,projected_trace,build_fourier_sources)
from yoshioka.leading_two_phonon import leading_pair_trace
from yoshioka.bulk_thermal_source import occupations
from yoshioka.fourier_depth_fields import complex_depth_field
from yoshioka.quadratic_fields import quadratic_depth_field


class IntermediateMomentumTests(unittest.TestCase):
    def fixture(self,n=3,m=5):
        rng=np.random.default_rng(991817)
        g=TransverseGrid(6e-10,8e-10,n,n);p=TransverseGrid(g.lx_m,g.ly_m,m,m)
        linear=(rng.normal(size=(2,m,m))+1j*rng.normal(size=(2,m,m)))*30*constants.elementary_charge
        quad=(rng.normal(size=(4,m,m))+1j*rng.normal(size=(4,m,m)))*constants.elementary_charge
        omega=np.array([15.,39.])*constants.elementary_charge/constants.hbar
        shift=np.array([[.12,-.21],[-.19,.16]])*1e10
        return g,p,linear,quad,omega,shift

    def test_sum_matches_existing_projected_rk4_all_words_and_signs(self):
        g,p,linear,quad,omega,shift=self.fixture()
        out=integer_beams(3);q=quad[:,out[:,0]%5,out[:,1]%5]
        for sub in (1,2,3):
            actual=propagate_intermediate_resolved(g,omega,shift,9e-10,7,
                lambda i:linear,lambda i:q,5,out,sub,lambda i:q*.31)
            expected=propagate_galerkin_pair(g,p,omega,shift,9e-10,7,
                lambda i:to_product(linear,5),lambda i:to_product(quad,5),lambda i:np.zeros((5,5)),sub)
            for key in ('direct_amplitudes','sequential_amplitudes','total_amplitudes'):
                np.testing.assert_allclose(actual[key],expected[key].reshape(4,-1),rtol=3e-13,atol=1e-16)
            words,lookup,*_=pair_structure(omega,shift)
            ordered=actual['ordered_by_intermediate'].sum(axis=-1)
            for sign in range(4):
                for order in range(2):
                    first=int(actual['first_letter_indices'][sign,order]);second=int(actual['second_letter_indices'][sign,order])
                    np.testing.assert_allclose(ordered[sign,order],expected['word_coefficients'][lookup[(second,first)]].ravel(),rtol=3e-13,atol=1e-16)
            for letter in range(4):
                np.testing.assert_allclose(actual['first_amplitudes'][letter],expected['word_coefficients'][lookup[(letter,)]].ravel(),rtol=3e-13,atol=1e-16)
            np.testing.assert_allclose(actual['direct_contact_amplitudes'],.31*actual['direct_amplitudes'],rtol=1e-13,atol=1e-17)

    def test_higher_support_extraction_matches_independent_lower_support(self):
        g,p,linear,quad,omega,shift=self.fixture(5,9)
        out=integer_beams(3);q=quad[:,out[:,0]%9,out[:,1]%9]
        big=propagate_intermediate_resolved(g,omega,shift,7e-10,5,lambda i:linear,lambda i:q,9,out,2)
        smallg=TransverseGrid(g.lx_m,g.ly_m,3,3);idx=frequency_indices(5,9)
        linear_small=linear[...,idx[:,None],idx]
        small=propagate_intermediate_resolved(smallg,omega,shift,7e-10,5,lambda i:linear_small,lambda i:q,5,out,2)
        h=big['intermediate_integer_labels'];mask=np.broadcast_to(np.all(abs(h)<=1,axis=1),(1,4,2,len(h)))
        extracted=cumulative_amplitudes(big,mask)
        np.testing.assert_allclose(extracted['sequential'][0],small['sequential_amplitudes'],rtol=2e-13,atol=1e-17)
        np.testing.assert_allclose(extracted['total'][0],small['total_amplitudes'],rtol=2e-13,atol=1e-17)

    def test_coherent_cumulative_keeps_cross_shell_and_direct_interference(self):
        y=np.zeros((4,2,1,2),complex);y[:,0,0]=[1.,-1.]
        r=dict(ordered_by_intermediate=y,direct_amplitudes=np.ones((4,1)),intermediate_integer_labels=np.array([[0,0],[0,1]]))
        masks=np.broadcast_to(np.array([[1,0],[1,1]])[:,None,None,:],(2,4,2,2))
        c=cumulative_amplitudes(r,masks)
        np.testing.assert_array_equal(c['sequential'][:,0,0],[1.,0.])
        np.testing.assert_array_equal(abs(c['total'][:,0,0])**2,[4.,1.])
        self.assertNotEqual(abs(y[0,0,0,0])**2+abs(y[0,0,0,1])**2,abs(y[0].sum())**2)

    def test_subset_trace_preserves_detector_thermal_and_proposal_factors(self):
        g,p,linear,quad,omega,shift=self.fixture();out=integer_beams(3)
        r=propagate_intermediate_resolved(g,omega,shift,5e-10,3,lambda i:linear,
             lambda i:quad[:,out[:,0]%5,out[:,1]%5],5,out)
        iw=np.array([.37,19.]);nb=occupations(omega,300)
        got=projected_trace(r['direct_amplitudes'],r['sequential_amplitudes'],omega,nb,g,r['final_shifts'],out,iw)
        old=leading_pair_trace(r['direct_amplitudes'].reshape(4,3,3),r['sequential_amplitudes'].reshape(4,3,3),omega,nb,g,r['final_shifts'],iw)
        for key in ('weights','direct_weights','sequential_weights','interference_weights','energies_mev'):
            np.testing.assert_allclose(got[key],old[key],rtol=1e-14,atol=1e-30)

    def test_required_transfer_inventory_reproduces_pinned_field_coefficients(self):
        rng=np.random.default_rng(15181);g=TransverseGrid(5e-10,4e-10,3,3)
        positions=np.array([[.1e-10,.2e-10,1.2e-10],[2.5e-10,2e-10,3.1e-10]])
        d=(rng.normal(size=(2,3,2))+1j*rng.normal(size=(2,3,2)))*1e-12
        shifts=np.array([[.12,-.21],[-.19,.16]])*1e10;out=integer_beams(3);L=8e-10;steps=33
        with tempfile.TemporaryDirectory() as folder:
            linear,quad,T=build_fourier_sources(positions,d,shifts,g,L,steps,out,folder)
            pg=TransverseGrid(g.lx_m,g.ly_m,T,T)
            for side in range(2):
                ref=complex_depth_field(positions,np.zeros((3,3)),pg,L,steps,Path(folder)/f'ref{side}.npy',d[:,:,side:side+1],shifts[side])
                np.testing.assert_allclose(linear[:,side],ref[:,0].reshape(steps,T,T),rtol=2e-12,atol=1e-33)
                ref._mmap.close()
            a=[];b=[];ss=[]
            for sa,sb in ((1,1),(-1,-1),(1,-1),(-1,1)):
                a.append(d[:,:,0].conj() if sa==1 else d[:,:,0]);b.append(d[:,:,1].conj() if sb==1 else d[:,:,1]);ss.append(sa*shifts[0]+sb*shifts[1])
            ref=quadratic_depth_field(positions,np.zeros((3,3)),pg,L,steps,Path(folder)/'refquad.npy',np.stack(a,axis=-1),np.stack(b,axis=-1),np.array(ss))
            idx=(out[:,0]%T)*T+out[:,1]%T
            np.testing.assert_allclose(quad[:,0],ref[:,:,idx],rtol=2e-12,atol=1e-33)
            ref._mmap.close();linear._mmap.close();quad._mmap.close()


if __name__=='__main__':unittest.main()
