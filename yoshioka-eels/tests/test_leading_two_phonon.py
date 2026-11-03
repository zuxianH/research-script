import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy import constants, fft
from scipy.linalg import expm
from yoshioka.free_propagation import TransverseGrid, electron_wavenumber
from yoshioka.elastic_propagation import electron_speed
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.quadratic_fields import quadratic_depth_field
from yoshioka.fourier_depth_fields import fourier_depth_field
from yoshioka.leading_two_phonon import propagate_leading_pair, leading_pair_trace
from yoshioka.thermal_dyson_words import propagate_words
from yoshioka.two_phonon import SIGNS


class LeadingTwoPhononTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(521174)
        grid = TransverseGrid(1e-9, 1.3e-9, 3, 3)
        omega = np.array([.031, .067])*constants.elementary_charge/constants.hbar
        shifts = np.array([[.37, -.19], [-.23, .41]])*(2*np.pi/np.array([grid.lx_m, grid.ly_m]))
        sources = (rng.normal(size=(3, 2, 3, 3))+1j*rng.normal(size=(3, 2, 3, 3)))*constants.elementary_charge
        quadratic = (rng.normal(size=(3, 4, 3, 3))+1j*rng.normal(size=(3, 4, 3, 3)))*constants.elementary_charge
        quadratic[:, 1] = quadratic[:, 0].conj(); quadratic[:, 3] = quadratic[:, 2].conj()
        bg = rng.normal(size=(3, 3, 3))*constants.elementary_charge
        return grid, omega, shifts, sources, quadratic, bg

    def test_nonzero_bloch_complex_amplitude_against_independent_dense_orders(self):
        grid, omega, shifts, sources, quadratic, bg = self.fixture()
        length = 1.5e-9; dz = length/3
        hv = constants.hbar*electron_speed(60000*constants.elementary_charge)
        actual = propagate_leading_pair(grid, omega, shifts, length, 3, lambda i:sources[i],
                                        lambda i:quadratic[i], lambda i:bg[i])
        states = np.array([(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)])
        lookup = {tuple(s):i for i,s in enumerate(states)}
        raising = []
        for m in range(2):
            a = np.zeros((6,6))
            for j,s in enumerate(states):
                f = s.copy(); f[m] += 1
                if tuple(f) in lookup:a[lookup[tuple(f)],j] = np.sqrt(s[m]+1)
            raising.append(a)
        signed = raising+[a.T for a in raising]
        energy = states@(constants.hbar*omega)
        Q = full_transverse_vectors(grid)[None]+(states@shifts)[:,None]
        kinetic = np.exp(-.5j*dz*np.sum(Q*Q,axis=-1).reshape(6,3,3)/(2*electron_wavenumber(60000*constants.elementary_charge)))
        wave = np.zeros((3,6,3,3),complex);wave[0,0]=1/np.sqrt(grid.lx_m*grid.ly_m)
        for z in range(3):
            wave = fft.ifft2(fft.fft2(wave,axes=(-2,-1))*kinetic[None],axes=(-2,-1))
            h0 = np.diag(np.repeat(energy,9)+np.tile(bg[z].ravel(),6))
            h1 = sum(np.kron(a,np.diag(f.ravel())) for a,f in zip(signed,np.r_[sources[z],sources[z].conj()]))
            # Quadratic cross vertex projected directly in the Fock basis.
            # Construct full ladders on a product basis before restricting, so
            # intermediate states are not falsely removed by a p=2 projector.
            ad = np.diag(np.sqrt(np.arange(1,4)),-1)
            full = [np.kron(ad,np.eye(4)),np.kron(np.eye(4),ad)]
            full += [a.T for a in full]
            keep = [4*a+b for a,b in states]
            h2 = np.zeros_like(h1)
            for j,(sa,sb) in enumerate(SIGNS):
                a=0 if sa==1 else 2; b=1 if sb==1 else 3
                op=(full[a]@full[b])[np.ix_(keep,keep)]
                h2 += np.kron(op,np.diag(quadratic[z,j].ravel()))
            augmented = np.kron(np.eye(3),h0)+np.kron(np.diag(np.ones(2),-1),h1)+np.kron(np.diag([1.],-2),h2)
            wave = (expm(-1j*dz/hv*augmented)@wave.ravel()).reshape(3,6,3,3)
            wave = fft.ifft2(fft.fft2(wave,axes=(-2,-1))*kinetic[None],axes=(-2,-1))
        expected = fft.fft2(wave[2],axes=(-2,-1))*np.sqrt(grid.lx_m*grid.ly_m)/9
        combined = actual['direct_amplitudes'][0]+actual['sequential_amplitudes'][0]
        np.testing.assert_allclose(combined,expected[lookup[(1,1)]],rtol=2e-11,atol=2e-14)
        self.assertGreater(np.linalg.norm(actual['direct_amplitudes']),0)
        self.assertGreater(np.linalg.norm(actual['sequential_amplitudes']),0)

    def test_vectorized_matches_scalar_and_sequential_foundation(self):
        g,o,s,v,q,b=self.fixture()
        args=(g,o,s,1.5e-9,3,lambda i:v[i],lambda i:q[i],lambda i:b[i])
        a=propagate_leading_pair(*args,vectorized=True);c=propagate_leading_pair(*args,vectorized=False)
        for key in ('word_coefficients','direct_amplitudes','sequential_amplitudes'):
            np.testing.assert_allclose(a[key],c[key],rtol=2e-13,atol=1e-14)
        _,old,_,_=propagate_words(g,o,s,1.5e-9,3,lambda i:v[i],lambda i:b[i])
        from yoshioka.thermal_dyson_words import words_through_two
        keep=[words_through_two(2).index(w) for w in a['words']]
        np.testing.assert_allclose(a['word_coefficients'],old[keep],rtol=2e-13,atol=1e-14)
        full=propagate_leading_pair(*args,retain_all_words=True)
        for key in ('direct_amplitudes','sequential_amplitudes'):
            np.testing.assert_allclose(a[key],full[key],rtol=2e-13,atol=1e-14)

    def test_quadratic_field_is_displaced_atom_mixed_derivative(self):
        g=TransverseGrid(8e-10,9e-10,5,5);steps=17;length=2e-9
        positions=np.array([[1.3,2.1,7.2],[4.1,3.3,10.7]])*1e-10
        a=np.array([[.017,-.011,.023],[-.021,.013,.007]])*1e-10
        b=np.array([[.009,.015,-.019],[.014,-.017,.011]])*1e-10
        cov=np.eye(3)*2e-23
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            actual=quadratic_depth_field(positions,cov,g,length,steps,p/'q.npy',a[:,:,None],b[:,:,None],[[0,0]])
            delta=.002; parts=[]
            for ia,ib in ((1,1),(1,-1),(-1,1),(-1,-1)):
                v=fourier_depth_field(positions+delta*(ia*a+ib*b),cov,g,length,steps,p/f'{ia}{ib}.npy')
                parts.append(np.array(v));v._mmap.close()
            reference=(parts[0]-parts[1]-parts[2]+parts[3])/(4*delta**2)
            relative=np.linalg.norm(actual-reference)/np.linalg.norm(actual)
            self.assertLess(relative,2e-6)
            half=quadratic_depth_field(positions,cov,g,length,steps,p/'half.npy',a[:,:,None],b[:,:,None],[[0,0]],multipliers=.5)
            np.testing.assert_allclose(half,np.array(actual)/2,rtol=1e-14)
            actual._mmap.close();half._mmap.close()

    def test_direct_sequential_signed_interference_and_thermal_average(self):
        g,o,s,v,q,b=self.fixture();n=np.array([.37,1.2])
        d=np.array(q[0]);seq=np.array(q[1]);sh=np.zeros((4,2))
        actual=leading_pair_trace(d,seq,o,n,g,sh,[2,3])
        np.testing.assert_allclose(actual['weights'],actual['direct_weights']+actual['sequential_weights']+actual['interference_weights'],rtol=1e-14)
        negative=leading_pair_trace(d,-d,o,n,g,sh,[2,3])
        np.testing.assert_array_equal(negative['weights'],0)
        self.assertTrue(np.all(negative['interference_weights']<=0))
        # Sum independently over initial thermal occupations, without taking a
        # thermal mean amplitude or allowing degenerate orthogonal states to interfere.
        expected=[]
        for k,(sa,sb) in enumerate(SIGNS):
            factor=0.
            for i in range(160):
                for j in range(160):
                    p=(n[0]/(n[0]+1))**i/(n[0]+1)*(n[1]/(n[1]+1))**j/(n[1]+1)
                    factor+=p*(i+(sa==1))*(j+(sb==1))
            expected.append(.5*6*factor*np.sum(abs(d[k]+seq[k])**2))
        np.testing.assert_allclose(actual['weights'][:,0],expected,rtol=3e-13)

if __name__=='__main__':unittest.main()
