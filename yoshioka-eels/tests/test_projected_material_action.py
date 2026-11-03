"""Projected material actions and independent bounded-field controls."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from scipy import constants
from yoshioka.channel_spectra import transition_lines,broaden_transition_lines,angular_detector_masks
from yoshioka.coupled_channels import channel_basis,oscillator_phase_block,thermal_channel_weights
from yoshioka.projected_material_action import ProjectedMaterialAction,phase_blocks,field_preflight
from yoshioka.checkpointed_channel_action import checkpointed_action


class Limits:
    def __init__(self,count=100): self.count=count
    def check(self,*_):
        self.count-=1
        if self.count<0: raise TimeoutError('test interruption')


def toy_bath():
    return SimpleNamespace(displacement_m=np.array([[[.02e-10,0],[0,.03e-10],[0,0]]]),
        omega_rad_s=np.array([8e13,9e13]),inactive_covariance_m2=np.array([np.eye(3)*.01e-20]),
        labels=['x','y'],cell_m=np.diag([5.,4.,6.])*1e-10,positions_m=np.array([[.8,1.2,2.]])*1e-10)


class ProjectedMaterialActionTests(unittest.TestCase):
    def test_basis_and_large_oscillator_projection(self):
        for p,count in ((2,6),(3,10),(4,15)):
            basis=channel_basis(2,p);self.assertEqual(len(basis),count)
            self.assertTrue(np.all(basis.sum(axis=1)<=p))
        basis=channel_basis(2,2);xi=np.array([[.4,.7],[-.2,.3]])
        block=phase_blocks(basis,xi,32)
        for i in range(2):
            np.testing.assert_allclose(block[i],oscillator_phase_block(basis,xi[i]),atol=3e-14)
            np.testing.assert_allclose(block[i],oscillator_phase_block(basis,xi[i],quadrature_order=32),atol=3e-14)
        # A projected infinite phase block is not a unitary truncated exponential.
        self.assertGreater(np.linalg.norm(block[0].conj().T@block[0]-np.eye(6)),.1)

    def test_thermal_absolute_weights(self):
        basis=channel_basis(2,2);quantum=np.array([.5,.8])*constants.k*100
        weights=thermal_channel_weights(basis,quantum,100)
        self.assertLess(weights.sum(),1)
        self.assertAlmostEqual(weights[0],(1-np.exp(-.5))*(1-np.exp(-.8)))

    def test_signed_and_degenerate_changed_lines(self):
        basis=np.array([[0,0],[1,0],[0,1]])
        waves=np.zeros((1,3,1,1),complex);waves[0,:,0,0]=np.sqrt([.2,.3,.5])
        lines=transition_lines(waves,[1],basis,[.058*constants.e]*2,[.7],np.ones((1,1,1),bool))
        np.testing.assert_allclose(lines['energy_loss_mev'],[[-58,0,0]])
        np.testing.assert_array_equal(lines['returning_state'],[[False,True,False]])
        spectrum=broaden_transition_lines(lines,np.linspace(-180,180,7201))
        self.assertAlmostEqual(spectrum['display_window_mass'][0],.7,places=12)
        self.assertAlmostEqual(spectrum['changed_per_mev'].sum()*.05,.49,places=12)

    def test_detector_partition(self):
        q=np.zeros((1,6,2));q[0,:,0]=np.tan(np.array([0,4,5,9,10,19])*1e-3)*1e12
        _,masks=angular_detector_masks(q,1e12)
        np.testing.assert_array_equal(masks[0],masks[1:].sum(axis=0))

    def test_memory_counts_actual_atoms_and_copies(self):
        small=field_preflight(16,33,769,769,6)
        self.assertEqual(small['full_complex_profile_bytes'],16*16*65*65*769)
        # One mapped file can remain fully resident before construction flush.
        large=field_preflight(4,97,769,1537,6)
        self.assertGreater(large['estimated_peak_rss_bytes'],16*193*193*1537)
        self.assertEqual(large['mapped_construction_profile_bytes'],16*193*193*1537)
        self.assertGreater(field_preflight(16,65,1537,1537,6,storage='full')['estimated_peak_rss_bytes'],4*1024**3)
        self.assertLess(field_preflight(16,65,1537,1537,6)['estimated_peak_rss_bytes'],4*1024**3)

    def test_independent_field_and_step_grids(self):
        with tempfile.TemporaryDirectory() as d:
            base=ProjectedMaterialAction(None,2,3,17,17,d,bath=toy_bath())
            refined=ProjectedMaterialAction(None,2,3,33,17,d,bath=toy_bath())
            # Recover the exact trigonometric polynomial from the base grid.
            values=np.array(base.profiles[0]);coeff=np.fft.fft(values,axis=0)/17
            k=np.fft.fftfreq(17)*17
            expected=np.exp(2j*np.pi*((8+.5)/33-.5/17)*k)@coeff
            np.testing.assert_allclose(refined.profiles[0][8],expected,rtol=1e-12,atol=1e-35)
            rng=np.random.default_rng(81);wave=rng.normal(size=base.shape)+1j*rng.normal(size=base.shape)
            full=ProjectedMaterialAction(None,2,3,17,17,d,bath=toy_bath(),storage='full')
            np.testing.assert_array_equal(base.apply(wave,7),full.apply(wave,7))
            matrix=np.column_stack([base.apply(e.reshape(base.shape),7).ravel() for e in np.eye(wave.size)])
            np.testing.assert_allclose(matrix,matrix.conj().T,rtol=1e-12,atol=1e-34)
            with self.assertRaises(ValueError):
                f=Path(d)/base.field_digest/'atom_000.npy';a=np.load(f);a[0,0]+=1
                np.save(f,a)
                ProjectedMaterialAction(None,2,3,17,17,d,bath=toy_bath())

    def test_paired_fields_preserve_vacuum_tail_hermiticity(self):
        with tempfile.TemporaryDirectory() as d:
            h=ProjectedMaterialAction(None,2,3,17,65,d,bath=toy_bath())
            for index in (0,1,8,16):
                h.set_slice(index)
                np.testing.assert_allclose(h.potential,h.potential.swapaxes(-1,-2).conj(),rtol=0,atol=1e-34)

    def test_checkpoint_exact_resume_nonvacuum_and_zero_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            h=ProjectedMaterialAction(None,2,3,7,7,Path(d)/'fields',bath=toy_bath(),initial_index=1)
            initial=np.zeros(h.shape,complex);initial[1,1,1]=1
            direct=initial.copy()
            for i in range(h.steps):direct=h.exponential_action(direct,i)
            partial,state=checkpointed_action(h,Path(d)/'resume',{},Limits(3),checkpoint_slices=2,initial_wave=initial)
            self.assertEqual(state['completed_slices'],2)
            final,state=checkpointed_action(h,Path(d)/'resume',{},Limits(),checkpoint_slices=2,initial_wave=initial)
            self.assertEqual(state['status'],'COMPLETE')
            np.testing.assert_array_equal(final,direct)
            zero,state=checkpointed_action(h,Path(d)/'zero',{},Limits(1),checkpoint_slices=2,initial_wave=initial)
            self.assertEqual(state['completed_slices'],0);np.testing.assert_array_equal(zero,initial)
            final,state=checkpointed_action(h,Path(d)/'zero',{},Limits(),checkpoint_slices=2,initial_wave=initial)
            np.testing.assert_array_equal(final,direct)
            with self.assertRaises(ValueError):
                checkpointed_action(h,Path(d)/'resume',{},Limits(),initial_wave=np.roll(initial,1,axis=0))

if __name__=='__main__':unittest.main()
