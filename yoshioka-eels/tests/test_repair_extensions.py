import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.constants import elementary_charge, hbar

from benchmarks.repair_b7 import finest_sequence_passes
from yoshioka.coupled_channels import channel_basis, oscillator_phase_block, linear_displacement_block
from yoshioka.dynamical_source import plane_field, propagate_incident, propagate_source_block
from yoshioka.free_propagation import TransverseGrid
from yoshioka.matrix_free_channels import ProductPhase, PhaseTerm, LocalChannelOperator, LadderOperator, representation_bytes
from yoshioka.source_streaming import streamed_source_exit, mapped_slice_provider
from yoshioka.comparison_metadata import comparison_availability, REQUIREMENTS


class RepairExtensionTests(unittest.TestCase):
    def test_finest_sequence_preserves_failures_without_poisoning(self):
        self.assertFalse(finest_sequence_passes([{'physical_passed':False},{'physical_passed':True}]))
        self.assertTrue(finest_sequence_passes([{'physical_passed':False},{'physical_passed':True},{'physical_passed':True}]))
        self.assertFalse(finest_sequence_passes([{'physical_passed':True},{'physical_passed':True},{'physical_passed':False}]))

    def test_tensor_and_bounded_tiles_match_nonlinear_projected_operator(self):
        basis=channel_basis(3,3);xi=[.3,-.2,.4]
        dense=oscillator_phase_block(basis,xi,48)
        rng=np.random.default_rng(371)
        wave=rng.normal(size=(len(basis),2,3))+1j*rng.normal(size=(len(basis),2,3))
        for maximum in (0,10000):
            operator=ProductPhase(basis,xi,48,maximum_tensor_entries=maximum,tile=7)
            np.testing.assert_allclose(operator.apply(wave),np.einsum('ij,jyx->iyx',dense,wave),atol=5e-15)
            np.testing.assert_allclose(operator.apply(wave,True),np.einsum('ji,jyx->iyx',dense.conj(),wave),atol=5e-15)

    def test_sparse_ladder_and_full_operator_adjoint_hermiticity(self):
        basis=channel_basis(2,3);grid=TransverseGrid(1e-9,1e-9,3,3)
        ladder=LadderOperator(basis,[.3,-.7])
        np.testing.assert_allclose(ladder.matrix.toarray(),linear_displacement_block(basis,[.3,-.7]))
        phase=ProductPhase(basis,[.7,-.2])
        coefficient=np.exp(1j*np.arange(9).reshape(3,3))*.2*elementary_charge
        operator=LocalChannelOperator(grid,basis@np.array([.04,.07])*elementary_charge,[PhaseTerm(coefficient,phase)],g=2)
        rng=np.random.default_rng(231)
        x=rng.normal(size=operator.shape)+1j*rng.normal(size=operator.shape)
        y=rng.normal(size=operator.shape)+1j*rng.normal(size=operator.shape)
        lhs=np.vdot(x,operator.apply(y));rhs=np.vdot(operator.apply(x),y)
        self.assertLess(abs(lhs-rhs)/max(abs(lhs),abs(rhs)),2e-14)
        a=operator.linear_operator(-3j)
        np.testing.assert_allclose(a.rmatvec(x.ravel()),3j*operator.apply(x).ravel(),atol=1e-32,rtol=1e-14)

    def test_streamed_signed_channels_match_existing_atomic_propagation(self):
        grid=TransverseGrid(6e-10,5e-10,7,7);length=2e-9;steps=65
        pos=np.array([[1.1e-10,2.2e-10,9e-10],[3.2e-10,1.4e-10,1.1e-9]])
        d=np.random.default_rng(738).normal(size=(2,3,3))*1e-12
        energy=60000*elementary_charge;omega=np.array([.02,.06,-.03])*elementary_charge/hbar
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            background=plane_field(pos,np.eye(3)*2e-23,grid,length,steps,directory/'bg.npy')
            source=plane_field(pos,np.eye(3)*2e-23,grid,length,steps,directory/'src.npy',d)
            _,history,_=propagate_incident(grid,energy,background,length,directory/'inc.npy')
            expected,_=propagate_source_block(grid,energy,omega,background,source,history,length)
            bg=mapped_slice_provider(background,grid);src=mapped_slice_provider(source,grid)
            actual=streamed_source_exit(grid,energy,omega,length,steps,src,lambda i:history[i],lambda i:bg(i)[0])
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),2e-14)
            for m in range(3):
                single=streamed_source_exit(grid,energy,omega[m:m+1],length,steps,lambda i:src(i)[m:m+1],lambda i:history[i],lambda i:bg(i)[0])
                self.assertLess(np.linalg.norm(single-actual[m:m+1])/np.linalg.norm(single),2e-14)

    def test_memory_breakdown_distinguishes_state_and_dense_coupling(self):
        d=representation_bytes(93,2,65,65)
        self.assertEqual(d['channels'],4465)
        self.assertEqual(d['bare_state_bytes'],301834000)
        self.assertGreater(d['dense_local_coupling_bytes'],1e12)
        self.assertLess(d['iterative_working_estimate_bytes'],4*1024**3)

    def test_cost_metadata_does_not_block_conditional_spectra(self):
        metadata={key:{'status':'USER_SUPPLIED'} for key in REQUIREMENTS['conditional_spectral_shape']}
        metadata['array_identity']={'status':'VERIFIED'}
        metadata['analysis_energy_axis']={'status':'ANALYSIS_ASSUMPTION'}
        result=comparison_availability(metadata)
        self.assertEqual(result['conditional_spectral_shape']['status'],'AVAILABLE_CONDITIONALLY')
        self.assertEqual(result['runtime_memory']['status'],'BLOCKED_BY_METADATA')
        self.assertEqual(result['absolute_intensity']['status'],'BLOCKED_BY_METADATA')
        metadata['analysis_energy_axis']['status']='MISSING'
        self.assertEqual(comparison_availability(metadata)['conditional_spectral_shape']['status'],'BLOCKED_BY_METADATA')

    def test_depth_layout_and_incident_equivalence(self):
        from yoshioka.depth_fields import convert_depth_field,depth_provider,incident_depth
        grid=TransverseGrid(6e-10,5e-10,7,7);length=2e-9;steps=65
        pos=np.array([[1.1e-10,2.2e-10,9e-10]])
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            old=plane_field(pos,np.eye(3)*2e-23,grid,length,steps,directory/'old.npy')
            depth,_=convert_depth_field(old,grid,directory/'depth.npy')
            a,ha,_=propagate_incident(grid,60000*elementary_charge,old,length,directory/'a.npy')
            b,hb,_=incident_depth(grid,60000*elementary_charge,depth,length,directory/'b.npy')
            np.testing.assert_allclose(a,b,rtol=2e-14,atol=1e-5)
            np.testing.assert_allclose(ha,hb,rtol=2e-14,atol=1e-5)

    def test_depth_construction_batch_equivalence(self):
        from yoshioka.depth_fields import depth_field as initial,depth_provider
        from yoshioka.depth_fields_batched import depth_field as batched
        grid=TransverseGrid(6e-10,5e-10,33,33);length=2e-9;steps=65
        pos=np.array([[1.1e-10,2.2e-10,9e-10],[3e-10,1.5e-10,1.2e-9]])
        displacement=np.random.default_rng(928).normal(size=(2,3,3))*1e-12
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            a,_=initial(pos,np.eye(3)*2e-23,grid,length,steps,directory/'a.npy',displacement)
            b,_=batched(pos,np.eye(3)*2e-23,grid,length,steps,directory/'b.npy',displacement)
            self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),2e-14)

    def test_streamed_material_prefix_resume_equivalence(self):
        from benchmarks.repair_9a_depth import calculate_streamed_case
        from benchmarks.workflow_io import ROOT,ResourceLimits
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        settings={'grid':5,'slices':17,'thickness_repeats':2,'phonon_z':2,'temperature_k':0,'padding_angstrom':12,'elastic_background':True}
        class Stop:
            calls=0
            def check(self,**kwargs):
                self.calls+=1
                if self.calls>=4:raise TimeoutError('controlled test prefix')
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            partial=calculate_streamed_case(model,settings,Path(a),Stop(),1)
            self.assertEqual(partial['completed_modes'],2)
            resumed=calculate_streamed_case(model,settings,Path(a),ResourceLimits(60,2),4)
            fresh=calculate_streamed_case(model,settings,Path(b),ResourceLimits(60,2),4)
            pa=Path(resumed['checkpoint']).parent;pb=Path(fresh['checkpoint']).parent
            for name in ('loss','gain'):
                np.testing.assert_allclose(np.load(pa/(name+'.npy')),np.load(pb/(name+'.npy')),atol=1e-17,rtol=2e-13)

    def test_known_intrinsic_line_and_actual_hann_response(self):
        from scipy.signal import fftconvolve
        from yoshioka.mode_spectra import hann_energy_kernel,fit_windowed_mode
        energy=np.arange(40,80.0005,.0005);center=60.2;width=.8;duration=100
        lorentz=(width/2)/(np.pi*((energy-center)**2+(width/2)**2))
        offsets=(np.arange(len(energy))-len(energy)//2)*.0005
        power=fftconvolve(lorentz,hann_energy_kernel(offsets,duration),mode='same')*.0005
        sampled=np.arange(45,75,.041356676969)
        fitted=fit_windowed_mode(sampled,np.interp(sampled,energy,power),60.,duration)
        self.assertLess(abs(fitted['center_mev']-center),.005)
        self.assertLess(abs(fitted['fitted_intrinsic_FWHM_mev']-width),.01)

    def test_gemm_plane_contraction_equivalence(self):
        from benchmarks.dynamical_si_eels import source_specimen
        from benchmarks.workflow_io import ROOT
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        from yoshioka.source_fields_gemm import plane_field_gemm
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        pos,d,omega,nbar,cov,diag=source_specimen(model,52,2,0)
        pos[:,2]+=12e-10;grid=TransverseGrid(2*model.cell_m[0,0],2*model.cell_m[1,1],17,17)
        length=52*model.cell_m[2,2]+24e-10;d=d[:,:,[0,22,86]]
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            old=plane_field(pos,cov,grid,length,129,directory/'a.npy',d)
            new=plane_field_gemm(pos,cov,grid,length,129,directory/'b.npy',d)
            self.assertLess(np.linalg.norm(old-new)/np.linalg.norm(old),1e-12)
            self.assertLess(np.max(abs(old-new)),np.linalg.norm(old)*1e-13)

    def test_dyson_operator_adjoint(self):
        from yoshioka.coordinate_channel_action import CoordinateMaterialAction
        from yoshioka.dyson_channel_action import DysonChannelAction
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        from benchmarks.workflow_io import ROOT
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        action=DysonChannelAction(CoordinateMaterialAction(model,2,3,33),2)
        rng=np.random.default_rng(192)
        x=rng.normal(size=action.shape)+1j*rng.normal(size=action.shape)
        y=rng.normal(size=action.shape)+1j*rng.normal(size=action.shape)
        lhs=np.vdot(x,action.apply(y,15));rhs=np.vdot(action.apply(x,15,True),y)
        self.assertLess(abs(lhs-rhs)/max(abs(lhs),abs(rhs)),2e-13)

    def test_channel_slice_checkpoint_resume_equivalence_and_checksum(self):
        from yoshioka.coordinate_channel_action import CoordinateMaterialAction
        from yoshioka.checkpointed_channel_action import checkpointed_action
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        from benchmarks.workflow_io import ROOT,ResourceLimits
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        h=CoordinateMaterialAction(model,2,3,33);contract={'physical_test':'identical33slices'}
        class Stop:
            calls=0
            def check(self):
                self.calls+=1
                if self.calls==20:raise TimeoutError('controlled slice stop')
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            _,partial=checkpointed_action(h,a,contract,Stop(),8)
            self.assertEqual(partial['completed_slices'],16)
            actual,state=checkpointed_action(h,a,contract,ResourceLimits(60,2),8)
            expected,fresh=checkpointed_action(h,b,contract,ResourceLimits(60,2),8)
            np.testing.assert_allclose(actual,expected,atol=1e-15,rtol=1e-13)
            with (Path(a)/state['wave_file']).open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):checkpointed_action(h,a,contract,ResourceLimits(60,2),8)

    def test_bloch_bath_basis_and_full_cell_propagation(self):
        from benchmarks.repair_bloch_validation import basis_and_born_check,full_cell_propagation_check
        from benchmarks.workflow_io import ROOT
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        self.assertTrue(basis_and_born_check(model)['passed'])
        with tempfile.TemporaryDirectory() as folder:self.assertTrue(full_cell_propagation_check(Path(folder))['passed'])
        self.assertTrue(basis_and_born_check(model,(4,4,2))['passed'])
        with tempfile.TemporaryDirectory() as folder:self.assertTrue(full_cell_propagation_check(Path(folder),4,(.25,-.25))['passed'])

    def test_bloch_channel_prefix_resume_equivalence(self):
        from benchmarks.repair_9a_bloch import calculate
        from benchmarks.workflow_io import ROOT,ResourceLimits
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        settings={'grid_unit':5,'slices':17,'thickness_repeats':2,'phonon_z':2,'temperature_k':0,'padding_angstrom':12,'elastic_background':True}
        class Stop:
            calls=0
            def check(self,**kwargs):
                self.calls+=1
                if self.calls==3:raise TimeoutError('controlled Bloch channel stop')
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            partial=calculate(model,settings,Path(a),Stop(),3)
            self.assertEqual(partial['completed_modes'],6)
            resumed=calculate(model,settings,Path(a),ResourceLimits(60,2),4)
            fresh=calculate(model,settings,Path(b),ResourceLimits(60,2),4)
            np.testing.assert_allclose(np.load(Path(resumed['checkpoint']).parent/'loss.npy'),np.load(Path(fresh['checkpoint']).parent/'loss.npy'),rtol=1e-12,atol=1e-17)

    def test_direct_bloch_covariance_matches_finite_bath_without_dense_storage(self):
        from yoshioka.bloch_covariance import covariance_from_groups
        from yoshioka.bloch_source import bath_groups
        from yoshioka.electron_phonon import common_diagonal_covariance
        from yoshioka.lattice_dynamics import RealSpaceForceConstants,finite_periodic_modes
        from benchmarks.workflow_io import ROOT
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        for reps in ((2,2,2),(4,4,2),(4,4,4)):
            _,groups=bath_groups(model,reps,2)
            actual,diag=covariance_from_groups(groups,4)
            reference,_=common_diagonal_covariance(finite_periodic_modes(model,reps)['displacements_m'])
            self.assertLess(np.linalg.norm(actual-reference)/np.linalg.norm(reference),1e-12)
            self.assertFalse(diag['dense_bath_dynamical_matrix_constructed'])

    def test_zero_slice_checkpoint_survives_repeated_short_budgets(self):
        from benchmarks.repair_material_checkpoint_init import initialize
        from yoshioka.coordinate_channel_action import CoordinateMaterialAction
        from yoshioka.checkpointed_channel_action import checkpointed_action
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        from benchmarks.workflow_io import ROOT,ResourceLimits
        import json
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        h=CoordinateMaterialAction(model,2,3,33)
        class Stop:
            calls=0
            def check(self):
                self.calls+=1
                if self.calls==3:raise TimeoutError('before first slice commit')
        with tempfile.TemporaryDirectory() as a:
            config={'output':a,'memory_limit_gib':2,'settings':{'modes':2,'p':2,'n':3,'slices':33,'order':24}}
            state=initialize(config);contract=state['contract']
            for _ in range(2):
                _,partial=checkpointed_action(h,Path(a)/'checkpoint',contract,Stop(),16)
                self.assertEqual(partial['completed_slices'],0)
            actual,complete=checkpointed_action(h,Path(a)/'checkpoint',contract,ResourceLimits(60,2),16)
            expected,_=h.propagate()
            self.assertEqual(complete['completed_slices'],33)
            np.testing.assert_allclose(actual,expected,atol=1e-15,rtol=1e-13)

    def test_windowed_four_worker_bloch_source_matches_serial_reference(self):
        from yoshioka.bloch_source import complex_depth_field,source_exit_bloch,bath_groups
        from yoshioka.bloch_source_windowed import complex_depth_field as windowed,source_exit_bloch as parallel
        from yoshioka.depth_fields import depth_provider,incident_depth
        from yoshioka.source_fields_gemm import depth_field
        from yoshioka.electron_phonon import common_diagonal_covariance
        from yoshioka.lattice_dynamics import RealSpaceForceConstants,finite_periodic_modes
        from benchmarks.workflow_io import ROOT
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        positions,groups=bath_groups(model,(2,2,2),2);group=groups[-2];positions[:,2]+=12e-10
        cov,_=common_diagonal_covariance(finite_periodic_modes(model,(2,2,2))['displacements_m'])
        grid=TransverseGrid(model.cell_m[0,0],model.cell_m[1,1],17,17);steps=129;length=2*model.cell_m[2,2]+24e-10
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);d=group['displacements_m'][:,:,:3];shift=group['bloch_shift_m_inv'];omega=group['omega_rad_s'][:3]
            a=complex_depth_field(positions,cov,grid,length,steps,folder/'a.npy',d,shift)
            b=windowed(positions,cov,grid,length,steps,folder/'b.npy',d,shift)
            self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-13)
            bg,_=depth_field(positions,cov,grid,length,steps,folder/'bg.npy');_,inc,_=incident_depth(grid,60000*elementary_charge,bg,length,folder/'inc.npy')
            bg_at=depth_provider(bg)
            expected=source_exit_bloch(grid,60000*elementary_charge,omega,length,steps,depth_provider(a),lambda i:inc[i],lambda i:bg_at(i)[0],shift)
            actual=parallel(grid,60000*elementary_charge,omega,length,steps,depth_provider(b),lambda i:inc[i],lambda i:bg_at(i)[0],shift)
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-12)

    def test_windowed_real_field_matches_checked_reference(self):
        from yoshioka.source_fields_gemm import depth_field as reference
        from yoshioka.windowed_fields import depth_field as windowed
        grid=TransverseGrid(6e-10,5e-10,33,33);positions=np.array([[1e-10,2e-10,8e-10],[4e-10,3e-10,1.2e-9]])
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);a,_=reference(positions,np.eye(3)*2e-23,grid,2e-9,257,folder/'a.npy')
            b,diagnostic=windowed(positions,np.eye(3)*2e-23,grid,2e-9,257,folder/'b.npy')
            self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-13)
            self.assertLess(diagnostic['maximum_imaginary_residual'],1e-12)

    def test_joint_b7_source_prefix_resume_equivalence(self):
        import json,subprocess,sys,os
        from benchmarks.workflow_io import ROOT
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            config={'mesh':2,'grid':16,'samples':64,'padding_angstrom':12,'runtime_limit_seconds':30,'memory_limit_gib':2,'FWHM_mev':2}
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
            paths=[]
            for folder in (a,b):
                p=Path(folder)/'config.json';p.write_text(json.dumps({**config,'output':folder}));paths.append(p)
            for p,count in ((paths[0],2),(paths[0],0),(paths[1],0)):
                subprocess.run([sys.executable,'-m','benchmarks.repair_b7_joint','--config',str(p),'--max-new-q',str(count)],cwd=ROOT,env=env,check=True,capture_output=True)
            da=next(Path(a).glob('*/state.json')).parent;db=next(Path(b).glob('*/state.json')).parent
            for name in ('weights','momentum','halves','forward'):
                np.testing.assert_allclose(np.load(da/(name+'.npy')),np.load(db/(name+'.npy')),rtol=1e-12,atol=1e-24)

    def test_fourier_depth_layout_matches_independent_space_fields_and_propagation(self):
        from yoshioka.fourier_depth_fields import complex_depth_field as fourier, fourier_depth_provider, depth_field
        from yoshioka.bloch_source import complex_depth_field as reference, source_exit_bloch, bath_groups
        from yoshioka.bloch_source_windowed import source_exit_bloch as parallel
        from yoshioka.source_fields_gemm import depth_field as reference_background
        from yoshioka.bloch_covariance import covariance_from_groups
        from yoshioka.depth_fields import depth_provider, incident_depth
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        from benchmarks.workflow_io import ROOT
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        positions,groups=bath_groups(model,(2,2,2),2);positions[:,2]+=12e-10
        cov,_=covariance_from_groups(groups,4);group=groups[-2]
        grid=TransverseGrid(model.cell_m[0,0],model.cell_m[1,1],17,17);steps=129;length=2*model.cell_m[2,2]+24e-10
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);d=group['displacements_m'][:,:,:3];shift=group['bloch_shift_m_inv'];omega=group['omega_rad_s'][:3]
            expected=reference(positions,cov,grid,length,steps,folder/'reference.npy',d,shift)
            mapped=fourier(positions,cov,grid,length,steps,folder/'source.npy',d,shift)
            actual=fourier_depth_provider(mapped,grid)
            for i in range(steps):np.testing.assert_allclose(actual(i),expected[i],rtol=2e-12,atol=1e-33)
            bg,diag=depth_field(positions,cov,grid,length,steps,folder/'bg.npy')
            old,_=reference_background(positions,cov,grid,length,steps,folder/'old.npy')
            self.assertLess(np.linalg.norm(bg-old)/np.linalg.norm(old),1e-13)
            self.assertLess(diag['global_relative_anti_Hermitian_error_before_projection'],1e-12)
            _,inc,_=incident_depth(grid,60000*elementary_charge,bg,length,folder/'inc.npy')
            bg_at=depth_provider(bg)
            a=source_exit_bloch(grid,60000*elementary_charge,omega,length,steps,depth_provider(expected),lambda i:inc[i],lambda i:bg_at(i)[0],shift)
            b=parallel(grid,60000*elementary_charge,omega,length,steps,actual,lambda i:inc[i],lambda i:bg_at(i)[0],shift)
            self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-12)

    def test_fourier_depth_channel_prefix_resume_equivalence(self):
        from benchmarks.repair_9a_fourier_depth import calculate
        from benchmarks.workflow_io import ROOT,ResourceLimits
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        settings={'grid_unit':5,'slices':17,'thickness_repeats':2,'phonon_xy':2,'phonon_z':2,'temperature_k':0,'padding_angstrom':12,'elastic_background':True}
        class Stop(ResourceLimits):
            calls=0
            def check(self,**kwargs):
                self.calls+=1
                if self.calls==3:raise TimeoutError('controlled Fourier depth channel stop')
                return super().check(**kwargs)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            partial=calculate(model,settings,Path(a),Stop(60,2),3)
            self.assertEqual(partial['completed_modes'],3)
            resumed=calculate(model,settings,Path(a),ResourceLimits(60,2),4)
            fresh=calculate(model,settings,Path(b),ResourceLimits(60,2),4)
            np.testing.assert_allclose(np.load(Path(resumed['checkpoint']).parent/'loss.npy'),np.load(Path(fresh['checkpoint']).parent/'loss.npy'),rtol=1e-12,atol=1e-17)

    def test_coordinate_nonlinear_action_matches_fock_and_no_alias_shifts(self):
        from yoshioka.coordinate_channel_action import CoordinateMaterialAction
        from yoshioka.reciprocal_material import ReciprocalMaterialHamiltonian
        from yoshioka.lattice_dynamics import RealSpaceForceConstants
        from benchmarks.workflow_io import ROOT
        model=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
        a=CoordinateMaterialAction(model,3,5,65,quadrature_order=24)
        b=ReciprocalMaterialHamiltonian(model,3,5,65)
        rng=np.random.default_rng(341);wave=rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
        for index in (0,19,31,40,64):
            actual=a.apply(wave,index);reference=b.apply(wave,index)
            self.assertLess(np.linalg.norm(actual-reference)/np.linalg.norm(reference),2e-13)
        self.assertLess(a.gram_error,1e-12)

    def test_fixed_projector_residual_is_not_norm_drift(self):
        from yoshioka.reciprocal_material import projected_coupling_residual
        matrix=np.array([[0.,.3],[.3,1.]])
        retained=np.array([True,False]);wave=np.array([1.,0.])
        residual=projected_coupling_residual(lambda x:matrix@x,wave,retained)
        np.testing.assert_allclose(residual,[0.,.3])
        # PHP evolves this state with exact unit norm; leakage is still nonzero.
        self.assertGreater(np.linalg.norm(residual),0)

    def test_user_normalization_supports_conditional_absolute_comparison(self):
        from yoshioka.comparison_metadata import comparison_availability,REQUIREMENTS
        metadata={key:{'status':'VERIFIED'} for key in REQUIREMENTS['absolute_intensity']}
        metadata['incident_normalization']={'status':'USER_SUPPLIED'}
        result=comparison_availability(metadata)['absolute_intensity']
        self.assertEqual(result['status'],'AVAILABLE_CONDITIONALLY')
        self.assertEqual(result['conditional_on'],['incident_normalization'])
        metadata['momentum_axes']={'status':'MISSING'}
        result=comparison_availability(metadata)['absolute_intensity']
        self.assertEqual(result['missing'],['momentum_axes'])

    def test_envelope_sampling_detects_slow_correlations(self):
        from benchmarks.repair_thermal import envelope_effective_samples
        rng=np.random.default_rng(791);white=rng.normal(size=2000)
        slow=np.convolve(white,np.ones(40)/40,mode='same')
        self.assertLess(envelope_effective_samples(slow)['effective_samples'],envelope_effective_samples(white)['effective_samples']/5)

    def test_checked_md_resume_and_fresh_force_consistency(self):
        from yoshioka.molecular_dynamics_checked import checkpointed_md
        from yoshioka.mod_forces import ModifiedTersoffForces
        from benchmarks.workflow_io import ROOT,ResourceLimits
        from ase.io import read
        unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
        settings={'temperature_k':1000,'repetitions':[2,2,2],'seed':723571,'timestep_fs':.5,'sample_stride':10,
            'equilibration_steps':200,'production_steps':6000,'thermostat_time_fs':100}
        class Stop:
            calls=0
            def check(self,**kwargs):
                self.calls+=1
                if self.calls==4:raise TimeoutError('at committed MD restart')
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            folder,partial=checkpointed_md(unit,ROOT/'Si.tersoff.mod',settings,Path(a),Stop())
            self.assertEqual(partial['completed_frames'],200)
            folder,complete=checkpointed_md(unit,ROOT/'Si.tersoff.mod',settings,Path(a),ResourceLimits(60,2))
            other,_=checkpointed_md(unit,ROOT/'Si.tersoff.mod',settings,Path(b),ResourceLimits(60,2))
            for name in ('displacements_m','velocities_m_s','forces_n'):
                actual=np.load(folder/(name+'.npy'));expected=np.load(other/(name+'.npy'))
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-8)
            atoms=unit.repeat((2,2,2));u=np.load(folder/'displacements_m.npy');f=np.load(folder/'forces_n.npy')
            with ModifiedTersoffForces(atoms,ROOT/'Si.tersoff.mod') as calculator:
                for i in (199,200,201,399,400):
                    fresh=calculator.evaluate(atoms.positions+u[i]*1e10)['forces_ev_per_angstrom']*elementary_charge/1e-10
                    self.assertLess(np.linalg.norm(fresh-f[i])/np.linalg.norm(f[i]),1e-10)

    def test_local_source_quadrature_against_independent_dense_exponential(self):
        from benchmarks.repair_source_local_audit import independent_control
        rows=independent_control()
        self.assertGreater(rows[0]['complex_error'],1e-6)
        self.assertLess(rows[-1]['complex_error'],1e-13)


if __name__=='__main__': unittest.main()
