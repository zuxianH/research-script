import unittest
import numpy as np
from scipy import constants
from scipy.linalg import expm
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.elastic_propagation import electron_speed
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.pair_galerkin import pair_structure,to_product,from_product
from yoshioka.pair_depth_controls import (contact_aligned_mesh,subdivide_mesh,pack_regular,
    coarsen_regular_table,FrozenPlanarTable,propagate_table_pair)


class PairDepthControlsTests(unittest.TestCase):
    def setUp(self):
        self.g=TransverseGrid(8e-10,9e-10,3,3);self.p=TransverseGrid(8e-10,9e-10,5,5)
        self.length=2e-9;self.omega=np.array([4e13,9e13]);self.shifts=np.array([[3e9,-4e9],[-2e9,1e9]])

    def test_atomic_means_are_additive_and_contacts_are_separate(self):
        from yoshioka.pair_planar_averages import AveragedPairPlanarFields
        positions=np.array([[1.2e-10,1.7e-10,7.31e-10]])
        d=np.array([[[.003,.002],[.001,-.001],[.002,.004]]])*1e-10
        f=AveragedPairPlanarFields(positions,self.p,self.length,d,self.shifts,np.zeros((3,3)))
        coarse=contact_aligned_mesh(self.length,3,f.contact_depths_m);fine=subdivide_mesh(coarse,2)
        direct=np.array([pack_regular(f.averaged_fields(a,b)) for a,b in zip(coarse[:-1],coarse[1:])])
        refined=np.array([pack_regular(f.averaged_fields(a,b)) for a,b in zip(fine[:-1],fine[1:])])
        merged=coarsen_regular_table(refined,fine,coarse)
        self.assertLess(np.linalg.norm(merged-direct)/np.linalg.norm(direct),2e-12)
        t=FrozenPlanarTable(coarse,merged,f.contact_depths_m,f.quadratic_contact_fourier)
        self.assertEqual(len(t.contact_depths_m),1)
        self.assertGreater(np.linalg.norm(t.quadratic_contact_fourier),0)
        self.assertFalse(t.regular.flags.writeable)
        self.assertFalse(t.nodes.flags.writeable)

    def test_fixed_field_does_not_change_when_propagation_refines(self):
        rng=np.random.default_rng(774);coarse=np.array([0,.67e-9,self.length]);fine=subdivide_mesh(coarse,2)
        values=(rng.normal(size=(2,7,5,5))+1j*rng.normal(size=(2,7,5,5)))*constants.elementary_charge
        values[:,0]=values[:,0].real
        table=FrozenPlanarTable(coarse,values,[],np.zeros((0,4,5,5)))
        np.testing.assert_array_equal(table.propagation_cells(fine),[0,0,1,1])
        np.testing.assert_array_equal(table.propagation_cells(subdivide_mesh(fine,2)),np.repeat([0,1],4))
        with self.assertRaisesRegex(ValueError,'discontinuity'):
            table.propagation_cells(np.array([0,self.length]))
        with self.assertRaisesRegex(ValueError,'exact table boundary'):
            FrozenPlanarTable(coarse,values,[.731e-9],np.zeros((1,4,5,5)))

    def test_exact_contact_jump_on_nonuniform_fixed_mesh(self):
        plane=.731e-9;nodes=contact_aligned_mesh(self.length,3,[plane])
        regular=np.zeros((len(nodes)-1,7,5,5),complex);contact=np.zeros((1,4,5,5),complex)
        strength=np.array([1.,.8,.4,.6])*constants.elementary_charge*1e-12;contact[0,:,0,0]=strength
        table=FrozenPlanarTable(nodes,regular,[plane],contact)
        result=propagate_table_pair(self.g,self.p,self.omega,np.zeros((2,2)),table,subdivide_mesh(nodes,7),4)
        energies=constants.hbar*np.array([self.omega.sum(),-self.omega.sum(),self.omega[0]-self.omega[1],self.omega[1]-self.omega[0]])
        hv=constants.hbar*electron_speed(60000*constants.elementary_charge)
        expected=-1j*strength/hv*np.exp(-1j*energies*(self.length-plane)/hv)
        np.testing.assert_allclose(result['direct_amplitudes'][:,0,0],expected,rtol=1e-11,atol=1e-18)
        self.assertEqual(result['contact_planes_applied'],1)
        np.testing.assert_array_equal(result['sequential_amplitudes'],0)

    def test_noncommuting_projected_generator_against_independent_dense_exponential(self):
        rng=np.random.default_rng(9871);unit=constants.elementary_charge
        block=(rng.normal(size=(7,5,5))+1j*rng.normal(size=(7,5,5)))*unit*8
        block[0]=block[0].real
        table=FrozenPlanarTable([0,self.length],block[None],[],np.zeros((0,4,5,5)))
        words,lookup,energies,shifts,final=pair_structure(self.omega,self.shifts);w=len(words)
        parents=np.array([lookup[a[1:]] for a in words[1:]]);letters=np.array([a[0] for a in words[1:]])
        all_e=np.r_[energies,energies[final]];all_s=np.r_[shifts,shifts[final]]
        q=full_transverse_vectors(self.g)[None]+all_s[:,None]
        hv=constants.hbar*electron_speed(60000*unit)
        kinetic=hv*np.sum(q*q,axis=-1).reshape(-1,3,3)/(2*electron_wavenumber(60000*unit))
        size=(w+4)*9
        def generator(a):
            a=a.reshape(w+4,3,3);real=to_product(a,5)
            v=(block[0][None]+all_e[:,None,None])*real
            sources=np.r_[block[1:3],block[1:3].conj()]
            v[1:w]+=sources[letters]*real[parents];v[w:]+=block[3:]*real[0]
            return (-1j/hv*(from_product(v,3)+kinetic*a)).ravel()
        eye=np.eye(size);matrix=np.column_stack([generator(eye[:,i]) for i in range(size)])
        initial=np.zeros(size);initial[0]=1;reference=(expm(self.length*matrix)@initial).reshape(w+4,3,3)
        errors=[]
        for count in (4,8):
            result=propagate_table_pair(self.g,self.p,self.omega,self.shifts,table,np.linspace(0,self.length,count+1),8)
            observed=np.r_[result['word_coefficients'],result['direct_amplitudes']]
            errors.append(np.linalg.norm(observed-reference))
        self.assertGreater(errors[0]/errors[1],3.8)
        self.assertLess(errors[1],1e-3)


if __name__=='__main__':unittest.main()
