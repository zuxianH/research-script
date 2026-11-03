"""Bloch embedding, exact checkpoint resumption, resource and error-table tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.constants import atomic_mass,hbar

from benchmarks.bulk_eels import checkpointed_case,displacement_for_q,mesh_modes,primitive_atom_mapping
from benchmarks.convergence_provenance import reusable_provenance
from benchmarks.si_eels_convergence import ConvergenceConfig,_sequence_status,case_comparison
from benchmarks.workflow_io import ResourceLimits
from yoshioka.lattice_dynamics import RealSpaceForceConstants,finite_periodic_modes


def synthetic_lattice():
    # A declared spring model for numerical unit tests, never a substitute for
    # the supplied MOD forces in the material benchmark.
    k,kc=10.0,5.0;cell=np.diag([5e-10,4e-10,4e-10]);positions=np.array([[0,0,0],[1.25e-10,1e-10,1e-10]])
    ii=[];jj=[];shifts=[];tensors=[]
    for atom in range(2):
        ii.extend([atom,atom]);jj.extend([atom,1-atom]);shifts.extend([[0,0,0],[0,0,0]]);tensors.extend([(6*k+kc)*np.eye(3),-kc*np.eye(3)])
        for direction in np.eye(3,dtype=int):
            for sign in (-1,1):ii.append(atom);jj.append(atom);shifts.append(sign*direction);tensors.append(-k*np.eye(3))
    return RealSpaceForceConstants(cell,positions,np.ones(2)*28.085*atomic_mass,np.array(ii),np.array(jj),np.array(shifts),np.array(tensors))


class ConvergenceTests(unittest.TestCase):
    def test_reporting_only_changes_preserve_verified_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);material_path=base/'material.json';born_path=base/'born.json'
            material_path.write_text('{"report_revision":1}');born_path.write_text('{}')
            artifacts={name:{'sha256':name} for name in ('equilibrium','force_constants','hiphive_potential','covariances')}
            material={'artifacts':artifacts,'status':'PASS','checks':{'independent_reference':True}}
            born={'artifacts':{'modes':{'sha256':'physical_modes'}},'numerical_status':'PASS','checks':{'Born':True}}
            spec={'force_constant_model':{'manifest':str(material_path),'sha256':'revision1','status':'PASS'},'positions_m':[[0,0,0]]}
            original,first=reusable_provenance(spec,material,material_path,born,born_path,base)
            material_path.write_text('{"report_revision":2}')
            spec['force_constant_model']['sha256']='revision2'
            reused,second=reusable_provenance(spec,material,material_path,born,born_path,base)
            self.assertEqual(original,reused)
            self.assertEqual(first['scientific_input_contract_sha256'],second['scientific_input_contract_sha256'])
            self.assertNotEqual(first['current_validation_manifests'],second['current_validation_manifests'])
            material['artifacts']['force_constants']['sha256']='different_force_constants'
            changed,third=reusable_provenance(spec,material,material_path,born,born_path,base)
            self.assertNotEqual(reused,changed)
            self.assertNotEqual(second['scientific_input_contract_sha256'],third['scientific_input_contract_sha256'])

    def test_primitive_bloch_embedding_has_full_model_normalization_once(self):
        model=synthetic_lattice();finite=finite_periodic_modes(model,(2,2,2));positions=finite['positions_m']
        basis,translations=primitive_atom_mapping(model,positions)
        self.assertEqual(translations.shape,(16,3))
        q=np.array([.5,0,0]);omega,vectors=model.modes(q)
        displacement=displacement_for_q(model,positions,basis,q,omega,vectors,8)
        mass=np.tile(model.masses_kg,8)
        embedded=displacement*np.sqrt(2*mass[:,None,None]*omega[None,None,:]/hbar)
        np.testing.assert_allclose(np.sum(abs(embedded)**2,axis=(0,1)),1,atol=1e-14)
        full=finite['eigenvectors'].reshape(48,-1)
        dynamical=(full*finite['omega_rad_s'][None]**2)@full.conj().T
        residual=dynamical@embedded.reshape(48,6)-embedded.reshape(48,6)*omega[None]**2
        self.assertLess(np.linalg.norm(residual)/np.linalg.norm(dynamical),1e-12)
        weighted=displacement_for_q(model,positions,basis,q,omega,vectors,64)
        np.testing.assert_allclose(weighted,displacement/np.sqrt(8),atol=1e-24)

    def test_only_rigid_Gamma_translations_are_removed(self):
        _,_,omega,_,mask=mesh_modes(synthetic_lattice(),3)
        self.assertEqual(np.sum(~mask),3)
        self.assertEqual(np.sum(mask),6*27-3)
        self.assertTrue(np.all(omega[mask]>0))

    def test_actual_interrupted_case_resumes_to_identical_weights(self):
        model=synthetic_lattice();modes=finite_periodic_modes(model,(2,2,2));cov=np.eye(3)*1.8e-23
        settings={'mesh':3,'grid':16,'samples':64,'padding_angstrom':8}
        with tempfile.TemporaryDirectory(prefix='eels-resume-') as directory,tempfile.TemporaryDirectory(prefix='eels-uninterrupted-') as other:
            partial,_=checkpointed_case(model,modes['positions_m'],model.cell_m,(2,2,2),cov,settings,Path(directory),ResourceLimits(120,2,10),{'test':'resume'},max_new_q=3)
            self.assertEqual(partial['status'],'INCOMPLETE');self.assertEqual(partial['completed_q'],3)
            resumed,_=checkpointed_case(model,modes['positions_m'],model.cell_m,(2,2,2),cov,settings,Path(directory),ResourceLimits(120,2,10),{'test':'resume'})
            fresh,_=checkpointed_case(model,modes['positions_m'],model.cell_m,(2,2,2),cov,settings,Path(other),ResourceLimits(120,2,10),{'test':'resume'})
            self.assertEqual(resumed['status'],'COMPLETE');self.assertEqual(resumed['resumed_from_q'],3)
            self.assertEqual(resumed['physical_mode_count'],159)
            with np.load(resumed['artifact']) as a,np.load(fresh['artifact']) as b:
                np.testing.assert_array_equal(a['mode_detector_weights'],b['mode_detector_weights'])
            cached,new=checkpointed_case(model,modes['positions_m'],model.cell_m,(2,2,2),cov,settings,Path(directory),ResourceLimits(120,2,10),{'test':'resume'})
            self.assertFalse(new);self.assertEqual(cached['observables'],resumed['observables'])

    def test_preallocation_resource_guards(self):
        with self.assertRaises(MemoryError):ResourceLimits(120,.01,10).check(estimated_bytes=1024**3)
        limits=ResourceLimits(1e-12,2,10)
        with self.assertRaises(TimeoutError):limits.check()
        limits=ResourceLimits(120,2,1);limits.completed=1
        with self.assertRaises(TimeoutError):limits.check()

    def test_named_worst_observable_and_successive_gates(self):
        config=ConvergenceConfig()
        with tempfile.TemporaryDirectory() as directory:
            grid=np.linspace(-10,100,101);profile=np.exp(-((grid-60)/2)**2)
            paths=[]
            for index,factor in enumerate((1.02,1)):
                path=Path(directory)/f'{index}.npz';np.savez(path,energy_grid_mev=grid,**{f'spectrum_{name}':factor*profile for name in ('full_0_20_mrad','central_0_5_mrad','annulus_5_10_mrad','annulus_10_20_mrad')});paths.append(path)
            common={'settings':{},'peak_positions_mev':{'peak':60}}
            coarse={**common,'artifact':str(paths[0]),'observables':{'declared_detector':1.02}}
            fine={**common,'artifact':str(paths[1]),'observables':{'declared_detector':1.0}}
            comparison=case_comparison(coarse,fine,config)
            self.assertFalse(comparison['passed'])
            self.assertEqual(comparison['observable_changes']['worst_observable'],'declared_detector')
            self.assertAlmostEqual(comparison['spectral_changes']['maximum_relative_L1'],.02)
        self.assertEqual(_sequence_status([{'passed':True}],2),'INCOMPLETE')
        self.assertEqual(_sequence_status([{'passed':True},{'passed':True}],2),'PASS')
        self.assertEqual(_sequence_status([{'passed':True},{'passed':False}],2),'INCOMPLETE')


if __name__=='__main__':unittest.main()
