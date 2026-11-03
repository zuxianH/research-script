import unittest
import numpy as np
from scipy import constants,fft
from scipy.linalg import expm
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.elastic_propagation import electron_speed
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.pair_galerkin import to_product,from_product,propagate_galerkin_pair,pair_structure
from yoshioka.leading_two_phonon import propagate_leading_pair
from yoshioka.two_phonon import SIGNS


class GalerkinPairTests(unittest.TestCase):
    def test_exact_transfer_convolution_and_padding_independence(self):
        rng=np.random.default_rng(817144);n=5
        c=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        f=rng.normal(size=(9,9))+1j*rng.normal(size=(9,9))
        # Independent integer convolution: retained |g|<=2, transfers <=4.
        g=np.rint(np.fft.fftfreq(n)*n).astype(int)
        expected=np.array([[sum(c[k,l]*f[(y-g[k])%9,(x-g[l])%9]
                       for k in range(n) for l in range(n)) for x in g] for y in g])
        for m in (9,11,15):
            field=to_product(f,m);wave=to_product(c,m)
            np.testing.assert_allclose(from_product(field*wave,n),expected,rtol=1e-13,atol=1e-13)
        with self.assertRaises(ValueError):
            propagate_galerkin_pair(TransverseGrid(1e-9,1e-9,5,5),TransverseGrid(1e-9,1e-9,7,7),
                 np.ones(2),np.zeros((2,2)),1e-9,1,lambda i:None,lambda i:None,lambda i:None)

    def fixture(self,m):
        rng=np.random.default_rng(312345);n=3
        grid=TransverseGrid(1e-9,1.3e-9,n,n);product=TransverseGrid(grid.lx_m,grid.ly_m,m,m)
        omega=np.array([.031,.067])*constants.elementary_charge/constants.hbar
        shifts=np.array([[.13,-.19],[-.23,.21]])*(2*np.pi/np.array([grid.lx_m,grid.ly_m]))
        v=(rng.normal(size=(2,5,5))+1j*rng.normal(size=(2,5,5)))*constants.elementary_charge
        q=(rng.normal(size=(4,5,5))+1j*rng.normal(size=(4,5,5)))*constants.elementary_charge
        b=rng.normal(size=(5,5))*constants.elementary_charge
        b=(b+np.conj(b[np.ix_((-np.arange(5))%5,(-np.arange(5))%5)]))/2
        return grid,product,omega,shifts,to_product(v,m),to_product(q,m),to_product(b,m).real

    def test_against_dense_projected_hamiltonian_with_rk_refinement(self):
        g,p,o,s,v,q,b=self.fixture(7);n=g.nx;m=p.nx
        words,lookup,e,sh,final=pair_structure(o,s);w=len(words)
        all_e=np.r_[e,e[final]];all_sh=np.r_[sh,sh[final]]
        def matrix(real):
            fc=fft.fft2(real)/m**2;freq=np.rint(np.fft.fftfreq(n)*n).astype(int)
            return np.array([[fc[(ay-by)%m,(ax-bx)%m] for by in freq for bx in freq]
                             for ay in freq for ax in freq])
        h=np.zeros(((w+4)*n*n,(w+4)*n*n),complex)
        B=matrix(b)
        for i in range(w+4):h[i*9:(i+1)*9,i*9:(i+1)*9]=B+np.eye(9)*all_e[i]
        sources=np.r_[v,v.conj()]
        for i,a in enumerate(words[1:],1):h[i*9:(i+1)*9,lookup[a[1:]]*9:(lookup[a[1:]]+1)*9]+=matrix(sources[a[0]])
        for i in range(4):h[(w+i)*9:(w+i+1)*9,:9]+=matrix(q[i])
        length=1.2e-9;steps=2;dz=length/steps
        Q=full_transverse_vectors(g)[None]+all_sh[:,None]
        kinetic=np.exp(-.5j*dz*np.sum(Q*Q,axis=-1)/(2*electron_wavenumber(60000*constants.elementary_charge))).ravel()
        local=expm(-1j*dz/(constants.hbar*electron_speed(60000*constants.elementary_charge))*h)
        c=np.zeros((w+4)*9,complex);c[0]=1.
        for i in range(steps):c=kinetic*(local@(kinetic*c))
        c=c.reshape(-1,n,n)
        errors=[]
        for sub in (1,2,4):
            actual=propagate_galerkin_pair(g,p,o,s,length,steps,lambda i:v,lambda i:q,lambda i:b,sub)
            errors.append(np.linalg.norm(actual['direct_amplitudes']-c[w:]))
        self.assertGreater(errors[0]/errors[1],12)
        self.assertLess(errors[-1],2e-9)
        for i,(a,b) in enumerate(SIGNS):
            expected=c[lookup[(0 if a==1 else 2,1 if b==1 else 3)]]+c[lookup[(1 if b==1 else 3,0 if a==1 else 2)]]
            np.testing.assert_allclose(actual['sequential_amplitudes'][i],expected,rtol=2e-6,atol=1e-9)

    def test_collocation_limit_matches_existing_exact_local_action(self):
        g,p,o,s,v,q,b=self.fixture(5)
        # On the same grid RK4 solves the old collocation local exponential.
        product=TransverseGrid(g.lx_m,g.ly_m,3,3)
        v=from_product(v,3);q=from_product(q,3);b=from_product(b,3)
        v=to_product(v,3);q=to_product(q,3);b=to_product(b,3).real
        args=(g,o,s,1e-11,3,lambda i:v,lambda i:q,lambda i:b)
        expected=propagate_leading_pair(*args)
        actual=propagate_galerkin_pair(g,product,o,s,1e-11,3,lambda i:v,lambda i:q,lambda i:b,
                                      local_substeps=2,require_dealiased=False)
        for key in ('direct_amplitudes','sequential_amplitudes'):
            np.testing.assert_allclose(actual[key],expected[key],rtol=1e-10,atol=1e-16)


if __name__=='__main__':unittest.main()
