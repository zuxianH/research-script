"""Independent full-cell versus symmetry-sector complex amplitude checks."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from yoshioka.lateral_symmetry import LateralTranslationSector
from yoshioka.projected_material_action import ProjectedMaterialAction
from yoshioka.channel_spectra import angular_detector_masks,transition_lines,broaden_transition_lines
from yoshioka.free_propagation import electron_wavenumber
from tests.test_projected_material_action import toy_bath

class LateralSymmetryTests(unittest.TestCase):
    def test_full_operator_and_complex_propagation(self):
        base=toy_bath()
        base.positions_m=np.vstack([base.positions_m,base.positions_m+[0,0,1e-10]])
        base.displacement_m=np.repeat(base.displacement_m,2,axis=0)
        base.inactive_covariance_m2=np.repeat(base.inactive_covariance_m2,2,axis=0)
        full=SimpleNamespace(**vars(base))
        full.positions_m=np.vstack([base.positions_m+x*base.cell_m[0]+y*base.cell_m[1] for x,y in np.ndindex(2,3)])
        full.cell_m=np.diag([2,3,1])@base.cell_m
        full.displacement_m=np.tile(base.displacement_m,(6,1,1));full.inactive_covariance_m2=np.tile(base.inactive_covariance_m2,(6,1,1))
        sector=LateralTranslationSector((2,3),(7,7));small=sector.reduce_bath(full)
        np.testing.assert_array_equal(small.displacement_m,base.displacement_m)
        with tempfile.TemporaryDirectory() as d:
            large=ProjectedMaterialAction(None,2,7,7,17,Path(d)/'large',bath=full)
            reduced=ProjectedMaterialAction(None,2,3,7,17,Path(d)/'small',bath=small,coalesce_profiles=True)
            plain=ProjectedMaterialAction(None,2,3,7,17,Path(d)/'plain',bath=small)
            self.assertEqual(reduced.preflight['profile_templates'],1)
            rng=np.random.default_rng(71);wave=rng.normal(size=reduced.shape)+1j*rng.normal(size=reduced.shape);wave/=np.linalg.norm(wave)
            a=large.apply(sector.expand(wave),3);b=sector.expand(reduced.apply(wave,3))
            self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-12)
            self.assertLess(np.linalg.norm(plain.apply(wave,3)-reduced.apply(wave,3))/np.linalg.norm(a),1e-12)
            u=np.zeros(reduced.shape,complex);u[0,1,1]=1;v=sector.expand(u)
            for index in range(7):u=reduced.exponential_action(u,index);v=large.exponential_action(v,index)
            self.assertLess(np.linalg.norm(v-sector.expand(u)),1e-12)
            np.testing.assert_allclose(large.qxy[sector.iy[:,None],sector.ix[None,:]],reduced.qxy,rtol=1e-15,atol=1e-5)
            _,ml=angular_detector_masks(large.qxy,electron_wavenumber(large.energy_j));_,ms=angular_detector_masks(reduced.qxy,electron_wavenumber(reduced.energy_j))
            ll=transition_lines(v[None],[0],large.basis,[1e-20,1.1e-20],[1.],ml);ls=transition_lines(u[None],[0],reduced.basis,[1e-20,1.1e-20],[1.],ms)
            np.testing.assert_allclose(ll['probability'],ls['probability'],rtol=1e-12,atol=1e-20)
            np.testing.assert_allclose(ls['probability'][:,:,0],ls['probability'][:,:,1:].sum(axis=-1),atol=1e-15)
            spectrum=broaden_transition_lines(ls,np.linspace(-180,180,7201))
            np.testing.assert_allclose(spectrum['display_window_mass'],ls['probability'].sum(axis=(0,1)),atol=1e-12)

    def test_mapping_and_broken_symmetry_guards(self):
        sector=LateralTranslationSector((5,7),(225,161));self.assertEqual(sector.reduced_shape,(33,33))
        u=np.ones((6,33,33),complex);v=sector.expand(u)
        self.assertEqual(v.shape,(6,225,161));self.assertEqual(np.linalg.norm(u),np.linalg.norm(v))
        with self.assertRaises(ValueError):LateralTranslationSector((0,7),(225,161))
        with self.assertRaises(ValueError):sector.reduce_bath(toy_bath())
        with self.assertRaises(ValueError):sector.expand(np.ones((3,3)))


class FiniteCovarianceTests(unittest.TestCase):
    def test_full_covariance_and_specimen_normalization(self):
        from scipy import constants
        from yoshioka.lattice_dynamics import RealSpaceForceConstants,finite_periodic_modes
        from yoshioka.finite_snap_bath import finite_optical_bath
        # Two-atom isotropic spring lattice: optical triplet and dispersive acoustics.
        ii=[0,1,0,1];jj=[0,1,1,0];shifts=[[0,0,0]]*4
        blocks=[np.eye(3)*4,np.eye(3)*4,-np.eye(3)*4,-np.eye(3)*4]
        for a in range(2):
            for axis in range(3):
                for sign in (-1,1):
                    shift=np.zeros(3,int);shift[axis]=sign
                    ii.extend([a,a]);jj.extend([a,a]);shifts.extend([[0,0,0],shift]);blocks.extend([np.eye(3),-np.eye(3)])
        model=RealSpaceForceConstants(np.eye(3)*1e-9,np.array([[.2,.2,.2],[.7,.7,.7]])*1e-9,
            np.ones(2)*28*constants.atomic_mass,np.array(ii),np.array(jj),np.array(shifts),np.array(blocks))
        bath,e=finite_optical_bath(model,(3,2,2),10.)
        full=finite_periodic_modes(model,(3,2,2));w=full['omega_rad_s']
        factor=1+2/np.expm1(constants.hbar*w/(constants.k*10))
        reference=np.einsum('iam,ibm,m->iab',full['displacements_m'],full['displacements_m'],factor)
        np.testing.assert_allclose(e['full_covariance_m2'],reference,rtol=1e-12,atol=1e-35)
        np.testing.assert_allclose(bath.inactive_covariance_m2+e['active_covariance_m2'],reference,rtol=1e-12,atol=1e-35)
        np.testing.assert_allclose(e['gram'],np.eye(2),atol=1e-12)
        small,_=finite_optical_bath(model,(1,1,2),10.)
        np.testing.assert_allclose(bath.displacement_m[:4],small.displacement_m/np.sqrt(6),atol=1e-25)

if __name__=='__main__':unittest.main()
