"""Prospective crossed field/propagation depth audit of two frozen Si nodes."""
import argparse,json,time,shutil,hashlib,resource,mmap,importlib
from pathlib import Path
import numpy as np
from scipy import constants
from .workflow_io import ROOT,ResourceLimits,json_write,environment
from .complete_leading_eels import finite_json,KERNELS
from .approximate_eels import PRIMITIVE,ORIENTED,geometry,DETECTORS
_sampling=importlib.import_module('benchmarks.14_two_phonon_eels.pair_sampling_audit')
raw_vector=_sampling.raw_vector;WINDOW_NAMES=_sampling.WINDOW_NAMES
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.bulk_thermal_source import mode_group,occupations
from yoshioka.leading_two_phonon import leading_pair_trace
from yoshioka.two_phonon import masks_for_shift
from yoshioka.pair_planar_averages import AveragedPairPlanarFields
from yoshioka.pair_depth_controls import (contact_aligned_mesh,subdivide_mesh,pack_regular,
    coarsen_regular_table,FrozenPlanarTable,propagate_table_pair)

OLD=ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1'
DEFAULT=ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_continuation_20261102_v1/depth_error_audit'
EXTRA=('yoshioka/pair_depth_controls.py','yoshioka/pair_planar_fields.py','yoshioka/pair_planar_averages.py',
 'yoshioka/pair_galerkin.py','yoshioka/pair_planar_propagation.py','yoshioka/free_propagation.py',
 'yoshioka/elastic_propagation.py','yoshioka/detector_cells.py','benchmarks/approximate_eels.py',
 'benchmarks/two_phonon_analysis.py','benchmarks/14_two_phonon_eels/pair_sampling_audit.py','benchmarks/workflow_io.py',
 'benchmarks/14_two_phonon_eels/pair_depth_diagnostics.py','tests/test_pair_depth_controls.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(path,value):json_write(path,finite_json(value))


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    cancellation=json.loads((OLD/'localization_panel2.json').read_text())['pairs'][1]
    representative=json.loads((OLD/'representative_panel8.json').read_text())['pairs'][0]
    panel=[{**cancellation,'role':'known cancellation node5/109'},
           {**representative,'role':'first value-independent representative node0/69'}]
    potential=LobatoSilicon.from_abtem_table()
    inputs=(PRIMITIVE,ORIENTED,ROOT/'Si.tersoff.mod',OLD/'benchmark.json',OLD/'representative_panel8.json',OLD/'localization_panel2.json',Path(potential.source_path))
    sources=tuple(dict.fromkeys((*KERNELS,*EXTRA)))
    contract=dict(version=1,physical_benchmark='Si4_T300_60keV_staticTaylor_complete_changed_pair',
      retained=33,product=65,field_base_nominal_intervals=384,field_refinement=2,
      propagation_design='P0 = fine field mesh; P1 = its exact bisection; both field levels share the identical P mesh',
      regular_representation='Exact analytic regular interval means. Coarse means are width-weighted fine means; independent additivity tested.',
      contacts='Exact separate integrated V2zz atomic-plane jumps, once at each plane; all field and propagation meshes contain the exact planes.',
      local_substeps=2,optional_local_action_control=4,temperature_k=300,energy_ev=60000,
      incident_probability=1,gaussian_fwhm_mev=2,detectors=DETECTORS,window_names=WINDOW_NAMES,weak_absolute_floor=1e-18,
      source_hashes={p:digest(ROOT/p) for p in sources},
      input_hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs},panel=panel,environment=environment(),
      guards=dict(seconds_per_table_or_case=600,memory_gib_per_worker=4,max_scientific_workers=1,BLAS_OpenMP_MKL_threads=1),
      scope='Crossed diagnostic for two frozen sources only; no integrated BZ acceptance, no infinite-depth bound and no corrected production estimate.')
    contract=finite_json(contract);path=output/'contract.json'
    if path.exists():
        if json.loads(path.read_text())!=contract:raise ValueError('changed prospective depth contract; preserve run and use a new directory')
    else:
        write(path,contract)
        for name in contract['source_hashes']:
            target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        write(output/'panel.json',{'pairs':panel,'selection':'Known cancellation node plus first prior value-independent representative, fixed before evaluating this crossed design.'})
    return contract


def checked_contract(output):
    record=json.loads((output/'contract.json').read_text())
    for key in ('source_hashes','input_hashes'):
        for name,expected in record[key].items():
            if digest(ROOT/name)!=expected:raise ValueError('depth contract source/input changed: '+name)
    if environment()!=record['environment']:raise ValueError('numerical package environment changed')
    for row in record['panel']:
        if digest(row['source_path'])!=row['source_sha256']:raise ValueError('frozen source node changed')
    return record


def geometry_and_modes(record,node):
    model=RealSpaceForceConstants.load(PRIMITIVE);oriented=RealSpaceForceConstants.load(ORIENTED)
    positions,reference,length=geometry(oriented,4,12)
    product=TransverseGrid(oriented.cell_m[0,0],oriented.cell_m[1,1],record['product'],record['product'])
    source=record['panel'][node]
    with np.load(source['source_path']) as raw:q=raw['q_fractional'];branches=raw['branches'];inverse=raw['inverse_marginals']
    omega=[];shifts=[];displacements=[]
    for side in range(2):
        om,d,s,labels=mode_group(model,positions,[product.lx_m,product.ly_m],q[side],1.,reference)
        selected=np.flatnonzero(labels==branches[side])
        if len(selected)!=1:raise ValueError('unique frozen physical branch missing')
        omega.append(om[selected[0]]);shifts.append(s);displacements.append(d[:,:,selected[0]])
    return product,positions,length,np.array(omega),np.array(shifts),np.stack(displacements,axis=-1),q,branches,inverse


def build_tables(output,node):
    record=checked_contract(output);folder=output/f'node{node}';folder.mkdir(exist_ok=True)
    statepath=folder/'tables.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
    if state.get('status')=='COMPLETE':
        for name,expected in state['files'].items():
            if digest(folder/name)!=expected:raise ValueError('frozen field table changed')
        return state
    limits=ResourceLimits(600,4,100000);before=time.monotonic()
    product,positions,length,omega,shifts,d,q,branches,inverse=geometry_and_modes(record,node)
    fields=AveragedPairPlanarFields(positions,product,length,d,shifts,np.zeros((3,3)))
    coarse=contact_aligned_mesh(length,record['field_base_nominal_intervals'],fields.contact_depths_m);fine=subdivide_mesh(coarse,2)
    tablepath=folder/'field_fine.npy';shape=(len(fine)-1,7,product.ny,product.nx)
    limits.check(estimated_bytes=2*int(np.prod(shape))*16+128*1024**2)
    if state:
        if state['contract_sha256']!=digest(output/'contract.json') or digest(tablepath)!=state['fine_table_sha256']:raise ValueError('partial table changed')
        values=np.load(tablepath,mmap_mode='r+');start=state['next_cell']
    else:
        if tablepath.exists():raise ValueError('uncommitted table exists; preserve it in a new case')
        values=np.lib.format.open_memmap(tablepath,mode='w+',dtype=complex,shape=shape);start=0
    def commit(count,status):
        values.flush()
        write(statepath,dict(status=status,contract_sha256=digest(output/'contract.json'),next_cell=count,
          fine_table_sha256=digest(tablepath),cost_this_invocation=limits.costs(),source_node=record['panel'][node]))
    try:
        for i in range(start,len(fine)-1):
            if i%8==0:limits.check()
            values[i]=pack_regular(fields.averaged_fields(fine[i],fine[i+1]))
            if (i+1)%32==0:
                commit(i+1,'BUILDING_FIXED_TABLE');values._mmap.madvise(mmap.MADV_DONTNEED)
        commit(len(fine)-1,'FINE_TABLE_COMPLETE')
        merged=coarsen_regular_table(values,fine,coarse);np.save(folder/'field_coarse.npy',merged)
        # A direct coarse analytic evaluation verifies algebraic coarsening on
        # three preselected interval indices, independent of observed values.
        selected=sorted(set((0,len(coarse)//2,len(coarse)-2)));checks=[]
        for i in selected:
            expected=pack_regular(fields.averaged_fields(coarse[i],coarse[i+1]))
            checks.append(float(np.linalg.norm(merged[i]-expected)/np.linalg.norm(expected)))
        if max(checks)>2e-11:raise ValueError('analytic interval-mean coarsening verification failed')
        np.savez_compressed(folder/'field_geometry.npz',coarse_nodes_m=coarse,fine_nodes_m=fine,
          contact_depths_m=fields.contact_depths_m,contact_fourier=fields.quadratic_contact_fourier,
          omega_rad_s=omega,emission_shifts_m_inv=shifts,displacements_m=d,positions_m=positions,
          q_fractional=q,branches=branches,inverse_marginals=inverse,
          cell_lengths_m=[product.lx_m,product.ly_m],length_m=length)
        state=dict(status='COMPLETE',contract_sha256=digest(output/'contract.json'),
          files={name:digest(folder/name) for name in ('field_fine.npy','field_coarse.npy','field_geometry.npz')},
          coarse_cells=len(coarse)-1,fine_cells=len(fine)-1,coarse_to_direct_analytic_relative_checks=checks,
          field_table_preparation_seconds=time.monotonic()-before,cost_this_invocation=limits.costs(),source_node=record['panel'][node],
          physical_profiles='Exact regular analytic interval means; h_z second-derivative Dirac contacts stored separately.')
        write(statepath,state);print('TABLE',node,state['field_table_preparation_seconds'],flush=True)
    except (TimeoutError,MemoryError) as error:
        # The last committed 32-cell prefix is reusable and hash guarded.
        previous=json.loads(statepath.read_text()) if statepath.exists() else {}
        previous.update(status='PARTIAL_RESOURCE_LIMIT',reason=str(error),cost_this_invocation=limits.costs())
        write(statepath,previous);print('TABLE_PARTIAL',node,str(error),flush=True)
    finally:values._mmap.close()
    return state


def run_case(output,node,field_level,prop_level,substeps=2):
    record=checked_contract(output);folder=output/f'node{node}';tables=json.loads((folder/'tables.json').read_text())
    if tables['status']!='COMPLETE':raise ValueError('field tables are not complete')
    for name,expected in tables['files'].items():
        if digest(folder/name)!=expected:raise ValueError('fixed field table changed')
    name=f'F{field_level}_P{prop_level}_R{substeps}';case=folder/name;case.mkdir(exist_ok=True)
    identity=dict(contract_sha256=digest(output/'contract.json'),table_record_sha256=digest(folder/'tables.json'),
      node=node,field_level=field_level,propagation_level=prop_level,local_substeps=substeps)
    path=case/'result.json'
    if path.exists():
        done=json.loads(path.read_text())
        if done['identity']!=identity:raise ValueError('case identity changed')
        if done['status']=='COMPLETE':
            if digest(case/'amplitudes.npz')!=done['artifact_sha256']:raise ValueError('completed depth amplitudes changed')
            return done
    limits=ResourceLimits(600,4,100000);start_time=time.monotonic()
    raw=np.load(folder/'field_geometry.npz');n=record['retained'];m=record['product'];lx,ly=raw['cell_lengths_m']
    g=TransverseGrid(lx,ly,n,n);p=TransverseGrid(lx,ly,m,m)
    values=np.load(folder/('field_coarse.npy' if field_level==0 else 'field_fine.npy'),mmap_mode='r')
    table=FrozenPlanarTable(raw['coarse_nodes_m'] if field_level==0 else raw['fine_nodes_m'],values,raw['contact_depths_m'],raw['contact_fourier'])
    prop=subdivide_mesh(raw['fine_nodes_m'],2**prop_level);start=0;initial=None
    info=case/'checkpoint.json';checkpoint=case/'checkpoint.npz'
    if info.exists():
        saved=json.loads(info.read_text())
        if saved['identity']!=identity or digest(checkpoint)!=saved['artifact_sha256']:raise ValueError('partial propagation identity/array changed')
        with np.load(checkpoint) as a:initial=a['coefficients'].copy();start=int(a['next_interval'])
    def save(index,wave):
        temp=case/'checkpoint.tmp.npz';np.savez_compressed(temp,coefficients=wave,next_interval=index);temp.replace(checkpoint)
        write(info,dict(identity=identity,next_interval=index,artifact_sha256=digest(checkpoint),cost_this_invocation=limits.costs()))
    try:
        result=propagate_table_pair(g,p,raw['omega_rad_s'],raw['emission_shifts_m_inv'],table,prop,substeps,limits,start,initial,save)
        trace=leading_pair_trace(result['direct_amplitudes'],result['sequential_amplitudes'],raw['omega_rad_s'],occupations(raw['omega_rad_s'],300),g,result['final_shifts'],raw['inverse_marginals'])
        result.update(trace);result.update(omega_rad_s=raw['omega_rad_s'],q_fractional=raw['q_fractional'],branches=raw['branches'],inverse_marginals=raw['inverse_marginals'],cell_lengths_m=raw['cell_lengths_m'])
        for key in ('weights','direct_weights','sequential_weights','interference_weights'):result[key+'_windows']=raw_vector(result['energies_mev'],result[key])
        artifact=case/'amplitudes.npz';np.savez_compressed(artifact,**result)
        done=dict(status='COMPLETE',identity=identity,artifact_sha256=digest(artifact),cost_this_invocation=limits.costs(),
          elapsed_seconds_this_invocation=time.monotonic()-start_time,propagation_intervals=len(prop)-1,field_cells=len(table.nodes)-1,
          contact_planes_applied=result['contact_planes_applied'],maximum_local_diagonal_phase=result['maximum_local_diagonal_phase'],
          maximum_elastic_norm_drift_this_invocation=result['maximum_elastic_norm_drift_this_invocation'])
        write(path,done);print('CASE',node,name,done['elapsed_seconds_this_invocation'],flush=True)
    except (TimeoutError,MemoryError) as error:
        done=dict(status='PARTIAL_RESOURCE_LIMIT',identity=identity,reason=str(error),cost_this_invocation=limits.costs(),saved_prefix=str(info));write(path,done)
        print('CASE_PARTIAL',node,name,str(error),flush=True)
    finally:values._mmap.close();raw.close()
    return done


def compare(output):
    record=checked_contract(output);rows=[];incomplete=[];raw_differences={}
    for node in range(len(record['panel'])):
        cases={};hashes={}
        for f,p in ((0,0),(0,1),(1,0),(1,1)):
            folder=output/f'node{node}'/f'F{f}_P{p}_R2';meta=folder/'result.json'
            if not meta.exists() or json.loads(meta.read_text())['status']!='COMPLETE':incomplete.append(str(folder.relative_to(output)));continue
            expected=json.loads(meta.read_text())['artifact_sha256']
            if digest(folder/'amplitudes.npz')!=expected:raise ValueError('crossed result changed')
            with np.load(folder/'amplitudes.npz') as raw:cases[f,p]={k:raw[k].copy() for k in raw.files}
            hashes[f'F{f}_P{p}']=expected
        if len(cases)!=4:continue
        ref=cases[1,1];g=TransverseGrid(*ref['cell_lengths_m'],record['retained'],record['retained'])
        for f in (0,1):
            if not np.array_equal(cases[f,0]['field_table_nodes_m'],cases[f,1]['field_table_nodes_m']):raise ValueError('propagation comparison changed field representation')
        for p in (0,1):
            if not np.array_equal(cases[0,p]['propagation_nodes_m'],cases[1,p]['propagation_nodes_m']):raise ValueError('field comparison changed propagation mesh')
        terms={'field_at_coarse_propagation':{(1,0):1,(0,0):-1},'field_at_fine_propagation':{(1,1):1,(0,1):-1},
          'propagation_at_coarse_field':{(0,1):1,(0,0):-1},'propagation_at_fine_field':{(1,1):1,(1,0):-1},
          'field_propagation_interaction':{(1,1):1,(1,0):-1,(0,1):-1,(0,0):1}}
        effects={}
        for label,coeff in terms.items():
            entry={'amplitudes':{},'window_intensity_differences':{}}
            for component in ('direct','sequential','total'):
                key=component+'_amplitudes';delta=sum(value*cases[index][key] for index,value in coeff.items());reference=ref[key]
                mask_norm=[];mask_relative=[]
                for sector,shift in enumerate(ref['final_shifts']):
                    masks=masks_for_shift(g,shift);dn=np.sqrt(masks@(abs(delta[sector].ravel())**2));rn=np.sqrt(masks@(abs(reference[sector].ravel())**2))
                    mask_norm.append(dn);mask_relative.append(np.divide(dn,rn,out=np.full(4,np.nan),where=rn>0))
                entry['amplitudes'][component]=dict(absolute_complex_l2=float(np.linalg.norm(delta)),
                  relative_complex_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference)),sign_detector_absolute_l2=mask_norm,sign_detector_relative_l2=mask_relative)
                raw_differences[f'node{node}_{label}_{component}_complex']=delta
            for component in ('weights','direct_weights','sequential_weights','interference_weights'):
                delta=sum(value*cases[index][component+'_windows'] for index,value in coeff.items());reference=ref[component+'_windows']
                entry['window_intensity_differences'][component]=dict(signed_absolute_difference=delta,
                  relative_difference_above_existing_floor=np.divide(delta,reference,out=np.full_like(delta,np.nan),where=abs(reference)>1e-18),
                  absolute_difference_near_zero_preserved=True)
                raw_differences[f'node{node}_{label}_{component}_windows']=delta
            effects[label]=entry
        action_folder=output/f'node{node}'/'F1_P1_R4';action=None
        if (action_folder/'result.json').exists() and json.loads((action_folder/'result.json').read_text())['status']=='COMPLETE':
            with np.load(action_folder/'amplitudes.npz') as a:
                action=dict(record=str(action_folder.relative_to(output)),artifact_sha256=digest(action_folder/'amplitudes.npz'),
                  component_absolute_complex_l2={k:float(np.linalg.norm(a[k+'_amplitudes']-ref[k+'_amplitudes'])) for k in ('direct','sequential','total')},
                  window_signed_probability_difference=a['weights_windows']-ref['weights_windows'])
        fine_cancel=(ref['direct_weights_windows']+ref['sequential_weights_windows'])/np.maximum(ref['weights_windows'],1e-300)
        rows.append(dict(node=node,selection=record['panel'][node],case_artifact_hashes=hashes,effects=effects,local_action_control=action,
          fine_reference_windows=ref['weights_windows'],fine_reference_component_cancellation=fine_cancel,
          fine_case='F1_P1_R2',contacts_applied=int(ref['contact_planes_applied']),propagation_intervals=int(ref['actual_intervals']),
          scope='Deterministic selected-node crossed differences; interaction is a finite mixed difference, not a rigorous error bound.'))
    if raw_differences:np.savez_compressed(output/'crossed_differences.npz',**raw_differences)
    result=dict(status='COMPLETE_CROSSED_DIAGNOSTIC' if len(rows)==2 else 'INCOMPLETE_CROSSED_DIAGNOSTIC',
      rows=rows,incomplete_cases=incomplete,window_names=WINDOW_NAMES,detectors=DETECTORS,
      contract_sha256=digest(output/'contract.json'),source_sha256=digest(__file__),
      acceptance='No global numerical or BZ convergence inferred; field/propagation controls use identical physical inputs, explicit contacts and all56 windows.')
    write(output/'crossed_comparison.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--build-table',type=int,choices=(0,1));group.add_argument('--case',nargs=3,type=int,metavar=('NODE','FIELD','PROP'));group.add_argument('--compare',action='store_true')
    parser.add_argument('--substeps',type=int,choices=(2,4),default=2);args=parser.parse_args()
    if args.prepare:prepare(args.output)
    elif args.build_table is not None:build_tables(args.output,args.build_table)
    elif args.case:
        if any(value not in (0,1) for value in args.case):parser.error('NODE, FIELD and PROP must each be0or1')
        run_case(args.output,*args.case,args.substeps)
    else:compare(args.output)


if __name__=='__main__':main()
