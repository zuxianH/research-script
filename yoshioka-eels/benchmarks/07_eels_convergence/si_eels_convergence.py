"""Benchmark 7: resumable, fixed-model numerical convergence of phonon EELS."""

import argparse
from dataclasses import asdict,dataclass
import json
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge

from yoshioka.lattice_dynamics import covariance_mesh
from .bulk_eels import checkpointed_case,primitive_material
from .convergence_provenance import reusable_provenance
from .eels_observables import DETECTORS,absolute_spectrum,compare_observables,spectrum_change
from .one_phonon_born import load_material
from .workflow_io import ROOT,ResourceLimits,artifact,environment,fingerprint,json_write,previous_statuses,sha256


@dataclass(frozen=True)
class ConvergenceConfig:
    material_manifest: str = 'benchmark_results/si_lattice_dynamics/manifest.json'
    born_manifest: str = 'benchmark_results/one_phonon_born/manifest.json'
    electron_grids: tuple = (64,128,192)
    longitudinal_samples: tuple = (512,1024,2048)
    padding_angstrom: tuple = (4,6,8,10)
    phonon_meshes: tuple = (4,8,12,16,24,32)
    selected_grid: int = 192
    selected_samples: int = 2048
    selected_padding_angstrom: float = 10
    joint_grid: int = 256
    joint_samples: int = 4096
    joint_padding_angstrom: float = 12
    energy_steps_mev: tuple = (.2,.1,.05,.025)
    fwhm_mev: float = 2
    spectrum_L1_tolerance: float = .01
    observable_relative_tolerance: float = .01
    covariance_relative_tolerance: float = .01
    source_amplitude_tolerance: float = 1e-4
    weak_probability_scale: float = 1e-18
    required_successive_comparisons: int = 2
    runtime_limit_seconds: float = 1800
    memory_limit_gib: float = 2
    maximum_refinements: int = 30

    def __post_init__(self):
        if self.fwhm_mev!=2:raise ValueError('this declared convergence study holds Gaussian FWHM fixed at 2 meV')
        for name in ('spectrum_L1_tolerance','observable_relative_tolerance','covariance_relative_tolerance','source_amplitude_tolerance','weak_probability_scale','runtime_limit_seconds','memory_limit_gib'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(f'{name} must be positive and finite')
        for name in ('electron_grids','longitudinal_samples','phonon_meshes'):
            values=getattr(self,name)
            if len(values)<3 or any(not isinstance(value,int) or value<2 for value in values) or sorted(set(values))!=list(values):raise ValueError(f'{name} requires at least three increasing integers')
        if any(value%2 for value in self.longitudinal_samples) or self.joint_samples%2:raise ValueError('source sample counts must be even')
        if not self.padding_angstrom or any(value<=0 for value in self.padding_angstrom):raise ValueError('positive padding is required')
        if self.selected_grid not in self.electron_grids or self.selected_samples not in self.longitudinal_samples or self.selected_padding_angstrom not in self.padding_angstrom:raise ValueError('selected electron settings must be included in the sequences')
        if self.required_successive_comparisons<1 or self.maximum_refinements<1:raise ValueError('positive comparison/refinement limits required')


def case_comparison(coarse,fine,config):
    changes=compare_observables(coarse['observables'],fine['observables'],config.weak_probability_scale)
    with np.load(coarse['artifact']) as data:first={name:data[f'spectrum_{name}'] for name in DETECTORS}
    with np.load(fine['artifact']) as data:
        second={name:data[f'spectrum_{name}'] for name in DETECTORS};energy=data['energy_grid_mev']
    l1=spectrum_change(first,second,energy)
    peak_rows=[]
    for name,value in fine['peak_positions_mev'].items():
        before=coarse['peak_positions_mev'][name]
        peak_rows.append({'peak':name,'coarse_mev':before,'fine_mev':value,'absolute_change_mev':None if before is None or value is None else abs(value-before),
                          'missing_significant_peak':(before is None)!=(value is None)})
    resolved=[row for row in peak_rows if row['absolute_change_mev'] is not None]
    worst_peak=max(resolved,key=lambda row:row['absolute_change_mev']) if resolved else None
    peak_change=0.0 if worst_peak is None else worst_peak['absolute_change_mev']
    passed=changes['maximum_acceptance_error']<config.observable_relative_tolerance and l1['maximum_relative_L1']<config.spectrum_L1_tolerance and peak_change<config.fwhm_mev/10 and not any(row['missing_significant_peak'] for row in peak_rows)
    return {'coarse_settings':coarse['settings'],'fine_settings':fine['settings'],'observable_changes':changes,'spectral_changes':l1,
            'peak_changes':peak_rows,'maximum_peak_shift_mev':peak_change,'worst_peak':None if worst_peak is None else worst_peak['peak'],'passed':passed}


def _sequence_status(comparisons,required):
    return 'PASS' if len(comparisons)>=required and all(row['passed'] for row in comparisons[-required:]) else 'INCOMPLETE'


def _energy_study(case,config):
    with np.load(case['artifact']) as data:
        energies=data['energies_mev']*1e-3*elementary_charge;weights=data['mode_detector_weights']
    previous=None;comparisons=[]
    for step in config.energy_steps_mev:
        grid=np.arange(-10,100+step/2,step)
        spectra={name:absolute_spectrum(energies,weights[:,j],grid,config.fwhm_mev) for j,name in enumerate(DETECTORS)}
        if previous is not None:
            interpolated={name:np.interp(grid,previous[1],previous[2][name]) for name in DETECTORS}
            change=spectrum_change(interpolated,spectra,grid)
            comparisons.append({'coarse_step_mev':previous[0],'fine_step_mev':step,**change,'passed':change['maximum_relative_L1']<config.spectrum_L1_tolerance})
        previous=(step,grid,spectra)
    return comparisons


def _recommendations(results,config):
    completed=[row for row in results['cases'] if row['status']=='COMPLETE']
    nominal=[row for row in completed if row['settings']['grid']==config.selected_grid and row['settings']['samples']==config.selected_samples and row['settings']['padding_angstrom']==config.selected_padding_angstrom]
    finest=max(nominal,key=lambda row:row['settings']['mesh']) if nominal else None
    spec=results['model_specification']
    recommendation={'accepted_complete_calculation':results['accepted_settings'],
                    'electron_settings_supported_at_fixed_primitive_mesh':config.phonon_meshes[0] if results['category_statuses']['electron_spatial_and_integration']=='PASS' else None,
                    'electron_grid':config.selected_grid,'longitudinal_samples':config.selected_samples,'padding_angstrom_per_side':config.selected_padding_angstrom,
                    'transverse_spacings_angstrom':[value*1e10/config.selected_grid for value in spec['transverse_lengths_m']],
                    'source_sample_spacing_angstrom':(spec['specimen_thickness_m']*1e10+2*config.selected_padding_angstrom)/config.selected_samples,
                    'propagation_step_angstrom':2*(spec['specimen_thickness_m']*1e10+2*config.selected_padding_angstrom)/config.selected_samples,
                    'joint_electron_check_at_finest_mesh':results.get('joint_at_fixed_phonon_sampling',{}).get('passed',False),
                    'phonon_sampling_accepted':results['category_statuses']['phonon_sampling']=='PASS'}
    if finest is not None and not recommendation['phonon_sampling_accepted']:
        next_mesh=int(2*np.ceil(1.5*finest['settings']['mesh']/2))
        recommendation['next_phonon_mesh_to_test']=next_mesh
        recommendation['estimated_nominal_source_seconds']=finest['elapsed_seconds']/finest['q_points']*next_mesh**3
        recommendation['cost_estimate_basis']='linear number of independent q sectors using measured finest-mesh source time; excludes eigensystem, spectra, report, and cache reload costs; no extrapolated convergence claim'
        recommendation['action']='extend phonon_meshes in a copied config and resume in bounded invocations; keep the 2 meV FWHM, all modes, and all predeclared observables'
    return recommendation


def run_benchmark(config,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    model,material,material_path=load_material(config.material_manifest)
    born_path=ROOT/config.born_manifest
    born=json.loads(born_path.read_text())
    for value in born['artifacts'].values():
        path=born_path.parent/value['path']
        if sha256(path)!=value['sha256']:raise ValueError(f'Benchmark 6 artifact checksum mismatch: {path}')
    with np.load(born_path.parent/born['artifacts']['modes']['path']) as data:positions=data['positions_m'].copy()
    repetitions=tuple(born['parameters']['phonon_repetitions'])
    primitive,primitive_record=primitive_material(config.material_manifest,model,output)
    covariance_path=material_path.parent/material['artifacts']['covariances']['path']
    with np.load(covariance_path) as data:covariance=np.mean(data['covariance_m2'][0],axis=0)
    covariance=np.diag(np.diag(covariance))
    limits=ResourceLimits(config.runtime_limit_seconds,config.memory_limit_gib,config.maximum_refinements)
    spec={'positions_m':positions.tolist(),'cell_m':model.cell_m.tolist(),'electron_specimen_repetitions':repetitions,
          'specimen_thickness_m':repetitions[2]*model.cell_m[2,2],'transverse_lengths_m':[repetitions[0]*model.cell_m[0,0],repetitions[1]*model.cell_m[1,1]],
          'force_constant_model':{'manifest':str(material_path),'sha256':sha256(material_path),'status':material['status'],'force_supercell':material['force_constant_supercell_repetitions'],'fitted_range_angstrom':4},
          'electron_interaction':'Si Lobato, h=-eV; project SciPy constants','incident_energy_ev':60000,'temperature_k':0,'incident_probability':1,
          'illumination':'plane wave chi0=1/sqrt(A), supplied z=[110]','detectors_mrad':born['model_specification']['observables']['detectors'],
          'momentum_region':'fixed full disk 0..20 mrad, all 8x8 bins over [-20,20]^2; qx half regions and exact q=0 retained',
          'energy_window_mev':[-10,100],'peak_groups':born['model_specification']['observables']['peak_groups_mev'],'Gaussian_FWHM_mev':2,
          'phonons':'bulk primitive-cell harmonic BZ quadrature, all six branches except three exact Gamma translations',
          'propagation':'undepleted pure first-Born source with phonon Hamiltonian; numerical free-propagation/Simpson steps, equivalent prefix scan',
          'boundaries':'fixed transverse electron cell; explicit outgoing Bloch shift -q_phonon modulo reciprocal electron cell; finite z specimen and vacuum',
          'covariance_policy':'hold independently converged Benchmark 5 covariance fixed in electron and q-integral sequences; independently converge primitive covariances against it',
          'covariance_m2':covariance.tolist(),'primitive_basis':primitive_record,'constants':born['constants']}
    json_write(output/'model_specification.json',spec)
    provenance,cache_record=reusable_provenance(spec,material,material_path,born,born_path,output)
    results={'benchmark':'Benchmark 7: controlled one-phonon Si EELS convergence','status':'RUNNING','parameters':asdict(config),'model_specification':spec,
             'prior_benchmark_statuses_preserved':previous_statuses(),'environment':environment(),'cache_provenance':cache_record,'cases':[],'comparisons':{},'covariance_study':[],
             'calculation_source_sha256':{name:sha256(ROOT/name) for name in ('yoshioka/lattice_dynamics.py','yoshioka/electron_phonon.py','yoshioka/free_propagation.py','benchmarks/bulk_eels.py','benchmarks/eels_observables.py','benchmarks/si_eels_convergence.py','benchmarks/convergence_provenance.py')},
             'channel_truncation':{'status':'NOT_APPLICABLE','reason':'fixed harmonic one-phonon response, no actual coupled-channel occupation cutoff or p sweep'},
             'remaining_physical_approximations':['harmonic lattice','one creation event at T=0','single inelastic interaction','undepleted incident plane wave','no elastic dynamical scattering','bulk-periodic phonons with finite electron segment'],
             'thermal_300K_extension':{'status':'NOT_PERFORMED','reason':'primary task is T=0 numerical control; 300 K is not claimed before that control is established'},
             'resource_stop':None}
    def checkpoint():
        results['resource_costs']=limits.costs();json_write(output/'checkpoint.json',results)
    checkpoint()
    # Material/covariance established before electron numerical refinement.
    stopped=False
    for mesh in config.phonon_meshes:
        try:
            limits.check(estimated_bytes=mesh**3*4096+100*1024**2)
            cov,diagnostic=covariance_mesh(primitive,mesh,(0,))
        except (TimeoutError,MemoryError) as exc:
            stopped=True;results['resource_stop']=str(exc);break
        value=np.mean(cov[0],axis=0)
        previous=results['covariance_study'][-1]['covariance_m2'] if results['covariance_study'] else None
        results['covariance_study'].append({'mesh':mesh,'q_points':mesh**3,'covariance_m2':value.tolist(),
                                            'relative_change_from_previous':None if previous is None else float(np.linalg.norm(value-np.array(previous))/np.linalg.norm(value)),
                                            'relative_difference_from_independent_B5_covariance':float(np.linalg.norm(value-covariance)/np.linalg.norm(covariance)),**diagnostic})
    cases_by_settings={}
    def calculate(settings,category):
        key=fingerprint(settings)
        if key in cases_by_settings:return cases_by_settings[key]
        try:
            row,computed=checkpointed_case(primitive,positions,model.cell_m,repetitions,covariance,settings,output,limits,provenance)
        except (TimeoutError,MemoryError) as exc:
            row={'status':'INCOMPLETE','settings':settings,'stop_reason':str(exc)};computed=False
        row['category']=category;row['reused_completed_cache']=not computed
        if row.get('artifact'):
            row['artifact_sha256']=sha256(row['artifact'])
        results['cases'].append(row);checkpoint()
        if row['status']!='COMPLETE':
            results['resource_stop']=row.get('stop_reason','incomplete case');return None
        cases_by_settings[key]=row
        print(f"completed {category}: q={settings['mesh']}³ grid={settings['grid']} z={settings['samples']} pad={settings['padding_angstrom']:g} Å; P={row['observables']['full_0_20_mrad']:.8e}; amplitude error={row['complex_amplitude_relative_error_vs_Born']:.2e}",flush=True)
        return row
    electron_base={'mesh':config.phonon_meshes[0],'grid':config.selected_grid,'samples':config.selected_samples,'padding_angstrom':config.selected_padding_angstrom}
    electron_sequences={}
    for category,key,values in (('electron_grid','grid',config.electron_grids),('longitudinal','samples',config.longitudinal_samples),('padding','padding_angstrom',config.padding_angstrom)):
        if stopped:break
        sequence=[]
        for value in values:
            row=calculate({**electron_base,key:value},category)
            if row is None:stopped=True;break
            sequence.append(row)
        electron_sequences[category]=sequence
        results['comparisons'][category]=[case_comparison(a,b,config) for a,b in zip(sequence[:-1],sequence[1:])]
        if stopped:break
    phonon_cases=[]
    if not stopped:
        for mesh in config.phonon_meshes:
            row=calculate({**electron_base,'mesh':mesh},'phonon_sampling')
            if row is None:stopped=True;break
            phonon_cases.append(row)
            comparisons=[case_comparison(a,b,config) for a,b in zip(phonon_cases[:-1],phonon_cases[1:])]
            results['comparisons']['phonon_sampling']=comparisons
            checkpoint()
            if _sequence_status(comparisons,config.required_successive_comparisons)=='PASS':break
    joint=None
    if phonon_cases and not stopped:
        finest=phonon_cases[-1]
        joint=calculate({**finest['settings'],'grid':config.joint_grid,'samples':config.joint_samples,'padding_angstrom':config.joint_padding_angstrom},'joint_refinement')
        if joint is not None:
            results['joint_at_fixed_phonon_sampling']=case_comparison(finest,joint,config)
            if len(phonon_cases)>=2:
                results['joint_electron_and_phonon_comparison']=case_comparison(phonon_cases[-2],joint,config)
                cancellation=[]
                for name,value in joint['observables'].items():
                    coarse=phonon_cases[-2]['observables'][name];middle=finest['observables'][name]
                    denominator=abs(middle-coarse)+abs(value-middle)
                    cancellation.append({'observable':name,'separate_phonon_change':middle-coarse,'separate_electron_change':value-middle,'joint_change':value-coarse,
                                         'joint_to_sum_absolute_ratio':abs(value-coarse)/denominator if denominator>0 else None})
                results['joint_error_cancellation_diagnostics']=cancellation
    complete=[row for row in results['cases'] if row['status']=='COMPLETE']
    best=joint or (phonon_cases[-1] if phonon_cases else complete[-1] if complete else None)
    results['energy_sampling_comparisons']=[] if best is None else _energy_study(best,config)
    electron_status='PASS' if all(_sequence_status(results['comparisons'].get(name,[]),config.required_successive_comparisons)=='PASS' for name in ('electron_grid','longitudinal','padding')) else 'INCOMPLETE'
    phonon_status=_sequence_status(results['comparisons'].get('phonon_sampling',[]),config.required_successive_comparisons)
    max_mesh=phonon_cases[-1]['settings']['mesh'] if phonon_cases else None
    covariance_rows=[row for row in results['covariance_study'] if max_mesh is not None and row['mesh'][0]<=max_mesh]
    covariance_status='PASS' if len(covariance_rows)>=3 and all(row['relative_change_from_previous']<config.covariance_relative_tolerance and row['relative_difference_from_independent_B5_covariance']<config.covariance_relative_tolerance for row in covariance_rows[-2:]) else 'INCOMPLETE'
    energy_status=_sequence_status(results['energy_sampling_comparisons'],config.required_successive_comparisons)
    joint_pass=results.get('joint_at_fixed_phonon_sampling',{}).get('passed',False) and results.get('joint_electron_and_phonon_comparison',{}).get('passed',False)
    upstream=material['status']=='PASS' and born['numerical_status']=='PASS'
    source_accuracy=bool(best is not None and best['complex_amplitude_relative_error_vs_Born']<config.source_amplitude_tolerance)
    results['category_statuses']={'material_input_validation':'PASS' if upstream else 'PROVISIONAL',
                                'electron_spatial_and_integration':electron_status,'phonon_sampling':phonon_status,'covariance_sampling':covariance_status,
                                'energy_sampling':energy_status,'joint_refinement':'PASS' if joint_pass else 'INCOMPLETE',
                                'observable_convergence':'PASS' if electron_status==phonon_status==covariance_status==energy_status=='PASS' and joint_pass else 'INCOMPLETE',
                                'source_accuracy_vs_Born':'PASS' if source_accuracy else 'INCOMPLETE','channel_truncation':'NOT_APPLICABLE_FIXED_ONE_PHONON',
                                'multiphonon_and_repeated_interactions':'NOT_ASSESSED'}
    required=('material_input_validation','electron_spatial_and_integration','phonon_sampling','covariance_sampling','energy_sampling','joint_refinement','observable_convergence','source_accuracy_vs_Born')
    results['status']='PASS' if all(results['category_statuses'][name]=='PASS' for name in required) else 'INCOMPLETE'
    results['best_computed_settings']=None if best is None else best['settings']
    results['accepted_settings']=best['settings'] if results['status']=='PASS' else None
    results['recommendations']=_recommendations(results,config)
    results['conclusion']='Numerically converged within the harmonic one-phonon, single-inelastic-interaction approximation.' if results['status']=='PASS' else 'Numerical convergence is incomplete within the configured refinement/resource limits; all computed errors and uncompleted categories are retained.'
    results['resource_costs']=limits.costs();checkpoint()
    results['resource_costs']['unique_case_source_seconds']=sum({fingerprint(row['settings']):row.get('elapsed_seconds',0) for row in results['cases']}.values())
    results['resource_costs']['runtime_limit_scope']='numerical refinement/source work checked before each q sector; completed arrays are finalized into spectra and reports after a stop'
    from .convergence_outputs import write_outputs
    results['figures']=write_outputs(results,output)
    json_write(output/'results.json',results)
    json_write(output/'manifest.json',{'schema_version':1,'benchmark':7,'status':results['status'],'category_statuses':results['category_statuses'],'accepted_settings':results['accepted_settings'],
                                     'best_computed_settings':results['best_computed_settings'],'conclusion':results['conclusion'],
                                     'recommendations':results['recommendations'],
                                     'artifacts':{'results':artifact(output/'results.json',output,'documented per field','complete convergence table, provenance, and limits'),
                                                  'primitive_force_constants':artifact(output/'primitive_force_constants.npz',output,'m, kg, N/m','exact primitive representation of the validated material model'),
                                                  'model_specification':artifact(output/'model_specification.json',output,'SI except explicitly labelled','fixed physical model and observables')}})
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'benchmark_results/si_eels_convergence')
    parser.add_argument('--runtime-seconds',type=float);parser.add_argument('--memory-gib',type=float);parser.add_argument('--max-refinements',type=int)
    args=parser.parse_args(argv);values={} if args.config is None else json.loads(args.config.read_text())
    for name in ('electron_grids','longitudinal_samples','padding_angstrom','phonon_meshes','energy_steps_mev'):
        if name in values:values[name]=tuple(values[name])
    for arg,name in ((args.runtime_seconds,'runtime_limit_seconds'),(args.memory_gib,'memory_limit_gib'),(args.max_refinements,'maximum_refinements')):
        if arg is not None:values[name]=arg
    results=run_benchmark(ConvergenceConfig(**values),args.output)
    print(results['benchmark'],results['status']);print(json.dumps(results['category_statuses'],indent=2));print(results['conclusion'])
    return 0 if results['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
