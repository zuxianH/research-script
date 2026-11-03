"""Benchmark 6: Si one-phonon transition potentials and longitudinal Born sources."""

import argparse
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants

from yoshioka.electron_phonon import (
    FirstOrderSourcePropagator,LobatoSilicon,atomic_transition_fourier,common_diagonal_covariance,
    full_space_born,numerical_transition_source,transverse_modes,
)
from yoshioka.free_propagation import TransverseGrid
from yoshioka.lattice_dynamics import RealSpaceForceConstants,finite_periodic_modes
from .born_reference import finite_window_born,single_atom_fourier_audit
from .eels_observables import DETECTORS,PEAK_GROUPS,compare_observables,summarize
from .workflow_io import ROOT,ResourceLimits,artifact,environment,fingerprint,json_write,previous_statuses,sha256


@dataclass(frozen=True)
class BornConfig:
    material_manifest: str = "benchmark_results/si_lattice_dynamics/manifest.json"
    phonon_repetitions: tuple = (2,2,2)
    kinetic_energy_ev: float = 60000
    temperature_k: float = 0
    transverse_grids: tuple = (32,64,128,192)
    longitudinal_samples: tuple = (256,512,1024,2048)
    padding_angstrom: tuple = (2,4,6,8)
    selected_grid: int = 128
    selected_longitudinal_samples: int = 2048
    selected_padding_angstrom: float = 8
    maximum_angle_mrad: float = 20
    g_values: tuple = (0,.25,.5,1)
    amplitude_relative_tolerance: float = 1e-4
    intensity_relative_tolerance: float = .01
    refinement_relative_tolerance: float = .01
    weak_probability_scale: float = 1e-18
    fwhm_mev: float = 2
    energy_step_mev: float = .05
    runtime_limit_seconds: float = 1200
    memory_limit_gib: float = 4

    def __post_init__(self):
        if len(self.phonon_repetitions)!=3 or any(not isinstance(value,int) or value<1 for value in self.phonon_repetitions):
            raise ValueError('phonon_repetitions must contain three positive integers')
        for name in ('kinetic_energy_ev','amplitude_relative_tolerance','intensity_relative_tolerance','refinement_relative_tolerance','weak_probability_scale','fwhm_mev','energy_step_mev','runtime_limit_seconds','memory_limit_gib'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(f'{name} must be positive and finite')
        if self.maximum_angle_mrad!=20:raise ValueError('the predeclared detector/momentum region is fixed at 20 mrad')
        if self.temperature_k!=0:raise ValueError('Benchmark 6 is the T=0 source calculation')
        if not self.transverse_grids or any(not isinstance(n,int) or n<8 for n in self.transverse_grids):raise ValueError('transverse grids must be integers >=8')
        if not self.longitudinal_samples or any(not isinstance(n,int) or n<8 or n%2 for n in self.longitudinal_samples):raise ValueError('longitudinal samples must be even integers >=8')
        if not self.padding_angstrom or any(not np.isfinite(value) or value<=0 for value in self.padding_angstrom):raise ValueError('padding must be positive and finite')
        if self.selected_grid not in self.transverse_grids or self.selected_longitudinal_samples not in self.longitudinal_samples or self.selected_padding_angstrom not in self.padding_angstrom:
            raise ValueError('selected settings must appear in their refinement sequences')
        if 0 not in self.g_values or 1 not in self.g_values or any(not np.isfinite(g) or g<0 for g in self.g_values):raise ValueError('g_values must be finite nonnegative and include zero and one')


def load_material(manifest_path):
    path=Path(manifest_path)
    if not path.is_absolute():path=ROOT/path
    manifest=json.loads(path.read_text())
    for value in manifest['artifacts'].values():
        target=path.parent/value['path']
        if not target.is_file() or sha256(target)!=value['sha256']:
            raise ValueError(f"material artifact checksum mismatch: {target}")
    model=RealSpaceForceConstants.load(path.parent/manifest['artifacts']['force_constants']['path'])
    return model,manifest,path


def compute_case(model,modes,potential,config,grid_count,samples,padding_angstrom,output=None,displacements=None):
    energy=config.kinetic_energy_ev*constants.elementary_charge
    grid=TransverseGrid(lx_m=config.phonon_repetitions[0]*model.cell_m[0,0],ly_m=config.phonon_repetitions[1]*model.cell_m[1,1],nx=grid_count,ny=grid_count)
    positions=modes['positions_m'].copy();positions[:,2]+=padding_angstrom*1e-10
    length=config.phonon_repetitions[2]*model.cell_m[2,2]+2*padding_angstrom*1e-10
    displacement=modes['displacements_m'] if displacements is None else displacements
    covariance,diagnostic=common_diagonal_covariance(modes['displacements_m'])
    qxy,indices,angles=transverse_modes(grid,energy,config.maximum_angle_mrad)
    key=f"n{grid_count}_z{samples}_pad{padding_angstrom:g}"
    started=time.monotonic()
    z,source=numerical_transition_source(potential,positions,displacement,covariance,qxy,grid,length,samples)
    propagator=FirstOrderSourcePropagator(grid,energy,modes['omega_rad_s'],indices)
    amplitudes,history=propagator.propagate(source,length,record_history=True)
    reference=full_space_born(potential,positions,displacement,covariance,qxy,modes['omega_rad_s'],energy,grid.lx_m*grid.ly_m,length)
    egrid=np.arange(-10,100+config.energy_step_mev/2,config.energy_step_mev)
    observables,spectra,mode_weights=summarize(amplitudes,modes['omega_rad_s'],angles,egrid,config.fwhm_mev,qxy)
    reference_observables,reference_spectra,reference_modes=summarize(reference,modes['omega_rad_s'],angles,egrid,config.fwhm_mev,qxy)
    errors=compare_observables(observables,reference_observables,config.weak_probability_scale)
    mode_error=[]
    for index,(numerical,exact) in enumerate(zip(mode_weights['full_0_20_mrad'],reference_modes['full_0_20_mrad'])):
        absolute=abs(numerical-exact)
        mode_error.append({'mode':index,'energy_mev':float(modes['omega_rad_s'][index]*constants.hbar/(1e-3*constants.elementary_charge)),
                           'numerical_weight':float(numerical),'reference_weight':float(exact),'absolute_error':float(absolute),
                           'relative_error':float(absolute/exact) if exact>0 else None,
                           'acceptance_error':float(absolute/exact) if exact>=config.weak_probability_scale else float(absolute/config.weak_probability_scale)})
    amp_error=float(np.linalg.norm(amplitudes-reference))
    row={'key':key,'grid':grid_count,'longitudinal_samples':samples,'padding_angstrom':padding_angstrom,
         'dx_angstrom':grid.dx_m*1e10,'dy_angstrom':grid.dy_m*1e10,'longitudinal_sample_spacing_angstrom':length/samples*1e10,
         'propagation_step_angstrom':2*length/samples*1e10,'interaction_length_angstrom':length*1e10,
         'retained_electron_Fourier_modes':len(qxy),'phonon_channels':len(modes['omega_rad_s']),
         'amplitude_absolute_vector_error':amp_error,'amplitude_relative_vector_error':amp_error/float(np.linalg.norm(reference)),
         'reference_amplitude_norm':float(np.linalg.norm(reference)),'numerical_inelastic_weight':float(np.sum(abs(amplitudes)**2)),
         'observables':observables,'reference_observables':reference_observables,'observable_errors':errors,'mode_errors':mode_error,
         'maximum_mode_intensity_error':max(value['acceptance_error'] for value in mode_error),'covariance_diagnostic':diagnostic,
         'elapsed_seconds':time.monotonic()-started,'source_array_mebibytes':source.nbytes/1024**2}
    if output is not None:
        arrays={'amplitudes':amplitudes,'born_full_space_amplitudes':reference,'qxy_m_inv':qxy,'fft_flat_indices':indices,'angles_mrad':angles,
                'omega_rad_s':modes['omega_rad_s'],'energy_grid_mev':egrid,'source_norm_history':history,
                **{f'spectrum_{name}_per_mev':value for name,value in spectra.items()},
                **{f'reference_spectrum_{name}_per_mev':value for name,value in reference_spectra.items()}}
        np.savez_compressed(output/'arrays'/f'{key}.npz',**arrays)
        row['artifact']=f'arrays/{key}.npz'
    return row,(grid,positions,length,covariance,qxy,indices,angles,z,source,amplitudes,reference,propagator)


def _controls(model,modes,potential,config,nominal):
    grid,positions,length,cov,qxy,indices,angles,z,source,amplitudes,reference,propagator=nominal
    energy=config.kinetic_energy_ev*constants.elementary_charge
    finite,quadrature=finite_window_born(potential,positions,modes['displacements_m'],cov,qxy,modes['omega_rad_s'],energy,grid.lx_m*grid.ly_m,length)
    finite_error=float(np.linalg.norm(amplitudes-finite)/np.linalg.norm(finite))
    finite_full=float(np.linalg.norm(finite-reference)/np.linalg.norm(reference))
    # Independent direct G_tilde comparison at several actual z Fourier modes.
    kz=2*np.pi*np.fft.fftfreq(source.shape[-1],length/source.shape[-1])
    source_transform=np.fft.fft(source,axis=-1)*(grid.lx_m*grid.ly_m*length/source.shape[-1])
    coupling_errors=[]
    for index in (0,1,2,source.shape[-1]-1):
        vectors=np.column_stack((qxy,np.full(len(qxy),kz[index])))
        atomic=atomic_transition_fourier(potential,positions,modes['displacements_m'],cov,vectors)
        coupling_errors.append(float(np.linalg.norm(source_transform[:,:,index]-atomic)/np.linalg.norm(atomic)))
    g_scaling=[]
    p1=float(np.sum(abs(amplitudes)**2))
    for g in config.g_values:
        value,_=propagator.propagate(source,length,g)
        probability=float(np.sum(abs(value)**2))
        g_scaling.append({'g':g,'probability':probability,'quadratic_expected':p1*g*g,'relative_error':abs(probability-p1*g*g)/(p1*g*g) if g else 0,'amplitude_norm':float(np.linalg.norm(value))})
    # Rotate an exactly degenerate subspace and reconstruct its source again.
    energies=modes['omega_rad_s']*constants.hbar/(1e-3*constants.elementary_charge)
    group=None
    for index,value in enumerate(energies):
        chosen=np.flatnonzero(abs(energies-value)<1e-8)
        if len(chosen)>=2:group=chosen;break
    rng=np.random.default_rng(14031)
    random=rng.normal(size=(len(group),len(group)))+1j*rng.normal(size=(len(group),len(group)))
    unitary=np.linalg.qr(random)[0]
    rotated=modes['displacements_m'].copy().astype(complex)
    rotated[:,:,group]=modes['displacements_m'][:,:,group]@unitary
    _,rotated_source=numerical_transition_source(potential,positions,rotated,cov,qxy,grid,length,source.shape[-1])
    rotated_amplitudes,_=propagator.propagate(rotated_source,length)
    original_group=np.sum(abs(amplitudes[group])**2,axis=0)
    rotated_group=np.sum(abs(rotated_amplitudes[group])**2,axis=0)
    subspace_error=float(np.linalg.norm(original_group-rotated_group)/np.linalg.norm(original_group))
    # Pure longitudinal single-atom control makes the Qz sign observable.
    single_positions=np.array([[grid.lx_m*.31,grid.ly_m*.43,length*.47]])
    d=np.zeros((1,3,1),dtype=complex);d[0,2,0]=4e-12
    omega=np.array([60e-3*constants.elementary_charge/constants.hbar])
    q0=np.zeros((1,2));index0=np.array([0])
    _,single_source=numerical_transition_source(potential,single_positions,d,cov,q0,grid,length,source.shape[-1])
    single_prop=FirstOrderSourcePropagator(grid,energy,omega,index0)
    single,_=single_prop.propagate(single_source,length)
    single_reference=full_space_born(potential,single_positions,d,cov,q0,omega,energy,grid.lx_m*grid.ly_m,length)
    enlarged=TransverseGrid(lx_m=2*grid.lx_m,ly_m=grid.ly_m,nx=2*grid.nx,ny=grid.ny)
    _,large_source=numerical_transition_source(potential,single_positions,d,cov,q0,enlarged,length,source.shape[-1])
    large,_=FirstOrderSourcePropagator(enlarged,energy,omega,index0).propagate(large_source,length)
    # Opposite displacements at a common z cancel exactly in the forward source.
    cancel_positions=np.vstack((single_positions,single_positions+[grid.lx_m/3,0,0]))
    cancel_d=np.vstack((d,-d))
    _,cancel_source=numerical_transition_source(potential,cancel_positions,cancel_d,cov,q0,grid,length,source.shape[-1])
    cancel,_=single_prop.propagate(cancel_source,length)
    return {'finite_window_quadrature':quadrature,'finite_window_relative_amplitude_error':finite_error,
            'finite_window_vs_full_space_relative_amplitude_difference':finite_full,'coupling_fourier_relative_errors':coupling_errors,
            'g_scaling':g_scaling,'degenerate_subspace':{'indices':group.tolist(),'unitary_rotation_relative_intensity_error':subspace_error},
            'longitudinal_single_atom':{'numerical_amplitude_re_im':[single[0,0].real,single[0,0].imag],
                                        'reference_amplitude_re_im':[single_reference[0,0].real,single_reference[0,0].imag],
                                        'absolute_amplitude_error':float(abs(single-single_reference).max()),
                                        'relative_amplitude_error':float(np.linalg.norm(single-single_reference)/np.linalg.norm(single_reference)),
                                        'reference_probability':float(np.sum(abs(single_reference)**2)),
                                        'opposite_polarization_cancellation_probability':float(np.sum(abs(cancel)**2)),
                                        'double_area_probability_ratio':float(np.sum(abs(large)**2)/np.sum(abs(single)**2)),
                                        'expected_double_area_probability_ratio':.25},
            'finite_window_amplitudes':finite}


def run_benchmark(config:BornConfig,output):
    if config.temperature_k!=0:raise ValueError('Benchmark 6 starts at T=0; thermal propagation is a separate controlled extension')
    output=Path(output).resolve();(output/'arrays').mkdir(parents=True,exist_ok=True)
    model,material,material_path=load_material(config.material_manifest)
    modes=finite_periodic_modes(model,config.phonon_repetitions)
    potential=LobatoSilicon.from_abtem_table()
    limits=ResourceLimits(config.runtime_limit_seconds,config.memory_limit_gib,100)
    constants_data=json.loads(json.dumps(material['constants']))
    for name in ('electron mass','speed of light in vacuum'):
        value,units,uncertainty=constants.physical_constants[name]
        constants_data['values'][name]={'value':value,'units':units,'standard_uncertainty':uncertainty}
    results={'benchmark':'Benchmark 6: plane-wave Si one-phonon first-Born channels','status':'RUNNING','parameters':asdict(config),
             'material_status':material['status'],'material_manifest_sha256':sha256(material_path),'environment':environment(),'constants':constants_data,
             'prior_benchmark_statuses_preserved':previous_statuses(),'single_atom_audit':single_atom_fourier_audit(potential),'cases':[],
             'model_specification':{'atomic_positions_m':modes['positions_m'].tolist(),'specimen_thickness_m':config.phonon_repetitions[2]*model.cell_m[2,2],
                                    'phonon_model':f'bulk-periodic full {config.phonon_repetitions} oriented-cell model; no free surfaces','physical_modes':len(modes['omega_rad_s']),
                                    'removed_modes':'only the three known rigid translations','electron_potential':'Lobato Si, h=-e V; MOD used only for phonons',
                                    'illumination':f'{config.kinetic_energy_ev:g} eV plane wave along supplied z=[110], incident chi0=1/sqrt(A)',
                                    'propagation':'undepleted first-order inelastic source, common carrier and velocity; no elastic dynamical scattering',
                                    'boundaries':'periodic transverse, finite atomic segment along z with independently converged vacuum padding',
                                    'observables':{'detectors':DETECTORS,'peak_groups_mev':{name:[low,None if np.isinf(high) else high] for name,(low,high) in PEAK_GROUPS.items()},
                                                   'momentum_region':'full disk theta<=20 mrad; both qx half-regions and exact q=0 retained','energy_window_mev':[-10,100],
                                                   'energy_resolution_fwhm_mev':config.fwhm_mev,'incident_probability':1}},
             'Lobato_table':{'path':potential.source_path,'sha256':sha256(potential.source_path),'a_angstrom':potential.a_angstrom.tolist(),'b_angstrom2':potential.b_angstrom2.tolist()},
             'optional_complete_propagator':{'status':'NOT_PERFORMED','reason':'No validated more complete coupled-channel integrator is available; phonon Hamiltonian is retained in this source solver.'}}
    np.savez_compressed(output/'modes.npz',**{name:value for name,value in modes.items() if isinstance(value,np.ndarray)})
    setups=[(n,config.selected_longitudinal_samples,config.selected_padding_angstrom,'transverse') for n in config.transverse_grids]
    setups += [(config.selected_grid,n,config.selected_padding_angstrom,'longitudinal') for n in config.longitudinal_samples]
    setups += [(config.selected_grid,config.selected_longitudinal_samples,pad,'padding') for pad in config.padding_angstrom]
    cached={};nominal_data=None
    for n,samples,padding,category in setups:
        key=(n,samples,padding)
        limits.check(estimated_bytes=len(modes['omega_rad_s'])*100*samples*16*4)
        if key not in cached:
            row,data=compute_case(model,modes,potential,config,n,samples,padding,output)
            cached[key]=row
            if key==(config.selected_grid,config.selected_longitudinal_samples,config.selected_padding_angstrom):nominal_data=data
        row={**cached[key],'category':category};results['cases'].append(row)
        json_write(output/'checkpoint.json',results)
    selected_key=(config.selected_grid,config.selected_longitudinal_samples,config.selected_padding_angstrom)
    selected=cached[selected_key];results['selected_case']=selected
    controls=_controls(model,modes,potential,config,nominal_data)
    finite=controls.pop('finite_window_amplitudes')
    np.savez_compressed(output/'finite_window_reference.npz',amplitudes=finite)
    results['controls']=controls
    refinements={}
    for category in ('transverse','longitudinal','padding'):
        rows=[row for row in results['cases'] if row['category']==category]
        comparisons=[]
        for coarse,fine in zip(rows[:-1],rows[1:]):
            comparisons.append({'coarse':coarse['key'],'fine':fine['key'],**compare_observables(coarse['observables'],fine['observables'],config.weak_probability_scale)})
        refinements[category]=comparisons
    results['refinement_comparisons']=refinements
    results['checks']={
        'atomic_potential_Fourier_units_and_sign':results['single_atom_audit']['maximum_relative_error']<1e-10,
        'complex_amplitudes_vs_independent_finite_window_Born':controls['finite_window_relative_amplitude_error']<config.amplitude_relative_tolerance,
        'full_space_formula_padding_verified':controls['finite_window_vs_full_space_relative_amplitude_difference']<config.amplitude_relative_tolerance,
        'independent_coupling_Fourier_validation':max(controls['coupling_fourier_relative_errors'])<config.amplitude_relative_tolerance,
        'detector_and_peak_group_weights':selected['observable_errors']['maximum_acceptance_error']<config.intensity_relative_tolerance,
        'mode_weights':selected['maximum_mode_intensity_error']<config.intensity_relative_tolerance,
        'g_zero_source':next(row['probability'] for row in controls['g_scaling'] if row['g']==0)==0,
        'quadratic_g_scaling':max(row['relative_error'] for row in controls['g_scaling'])<config.intensity_relative_tolerance,
        'degenerate_subspace_invariance':controls['degenerate_subspace']['unitary_rotation_relative_intensity_error']<1e-10,
        'longitudinal_coupling_sign':controls['longitudinal_single_atom']['relative_amplitude_error']<config.amplitude_relative_tolerance,
        'incident_area_normalization':abs(controls['longitudinal_single_atom']['double_area_probability_ratio']/.25-1)<1e-10,
        'polarization_cancellation':controls['longitudinal_single_atom']['opposite_polarization_cancellation_probability']<1e-25,
        'no_T_zero_gain_lines':bool(np.all(modes['omega_rad_s']>0)),
        'spatial_integration_padding_refinement':all(values[-1]['maximum_acceptance_error']<config.refinement_relative_tolerance for values in refinements.values()),
    }
    results['numerical_status']='PASS' if all(results['checks'].values()) else 'FAIL'
    results['status']=results['numerical_status'] if material['status']=='PASS' else 'PROVISIONAL'
    results['resource_costs']=limits.costs()
    from .born_outputs import write_outputs
    results['figures']=write_outputs(results,output)
    json_write(output/'results.json',results)
    manifest={'schema_version':1,'benchmark':6,'status':results['status'],'numerical_status':results['numerical_status'],'material_status':material['status'],
              'parameters':asdict(config),'model_specification':results['model_specification'],'constants':constants_data,'checks':results['checks'],
              'artifacts':{'results':artifact(output/'results.json',output,'SI unless labelled','computed amplitudes, controls, and convergence'),
                           'modes':artifact(output/'modes.npz',output,'m, kg, rad/s; normalized eigenvectors','full declared periodic phonon model'),
                           'selected_channels':artifact(output/'arrays'/f"{selected['key']}.npz",output,'dimensionless amplitudes and probabilities; densities meV^-1','actual propagated source channels'),
                           'finite_window_reference':artifact(output/'finite_window_reference.npz',output,'dimensionless amplitudes','independent adaptive Born quadrature')}}
    json_write(output/'manifest.json',manifest)
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'benchmark_results/one_phonon_born')
    args=parser.parse_args(argv);values={} if args.config is None else json.loads(args.config.read_text())
    for name in ('phonon_repetitions','transverse_grids','longitudinal_samples','padding_angstrom','g_values'):
        if name in values:values[name]=tuple(values[name])
    results=run_benchmark(BornConfig(**values),args.output)
    print(results['benchmark'],results['status']);print(json.dumps(results['checks'],indent=2));print('Results:',args.output.resolve())
    return 0 if results['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
