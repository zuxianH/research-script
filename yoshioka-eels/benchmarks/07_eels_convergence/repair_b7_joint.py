"""Independent full-q numerical-source joint gate, streamed and resumable.

This is deliberately separate from selected-sector accelerator validation. A
partial prefix is never called a converged joint result or renormalized.
"""
import argparse,json,time,mmap
from pathlib import Path
import numpy as np
from scipy import constants

from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.integration_measures import uniform_bz_nodes
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.electron_phonon import LobatoSilicon,numerical_transition_source,FirstOrderSourcePropagator,transverse_modes
from yoshioka.longitudinal_integration import numerical_source_contraction
from .bulk_eels import primitive_atom_mapping,displacement_for_q
from .born_quadrature import born_batch
from .detector_cubature import geometry_table
from .workflow_io import ROOT,ResourceLimits,fingerprint,json_write,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='benchmarks/configs/repair_b7_joint_20261102_v2.json')
    parser.add_argument('--max-new-q',type=int,default=512,help='0 means bounded by configured runtime only')
    parser.add_argument('--seed-checkpoint',help='copy a validated legacy prefix into a NEW versioned output; never mutate the old checkpoint')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    spec=json.loads((ROOT/'benchmark_results/si_eels_convergence/results.json').read_text())['model_specification']
    primitive=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_eels_convergence/primitive_force_constants.npz')
    n=config['mesh'];count=n**3;grid=TransverseGrid(*spec['transverse_lengths_m'],config['grid'],config['grid'])
    pad=config['padding_angstrom']*1e-10;length=spec['specimen_thickness_m']+2*pad
    energy=60000*constants.elementary_charge;k=electron_wavenumber(energy)
    backend=config.get('source_backend','scan')
    if backend not in ('scan','contracted'):raise ValueError('source_backend must be scan or contracted')
    source={f:sha256(ROOT/f) for f in ('benchmarks/repair_b7_joint.py','yoshioka/electron_phonon.py','benchmarks/detector_cubature.py','yoshioka/longitudinal_integration.py','benchmarks/bulk_eels.py','benchmarks/born_quadrature.py','yoshioka/free_propagation.py','yoshioka/integration_measures.py')}
    contract={'parameters':{k:v for k,v in config.items() if k not in ('output','runtime_limit_seconds','seed_runner_snapshot')},'source':source,'specification':spec,
              'physical_inputs':{'primitive_force_constants':sha256(ROOT/'benchmark_results/si_eels_convergence/primitive_force_constants.npz')}}
    digest=fingerprint(contract);folder=output/digest[:20];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'state.json';state=json.loads(path.read_text()) if path.exists() else {'digest':digest,'contract':contract,'completed_q':0,'total_q':count,'cumulative_seconds':0.,'status':'INCOMPLETE','squared_amplitude_error':0.,'reference_squared_norm':0.}
    if state['digest']!=digest:raise ValueError('joint checkpoint identity mismatch')
    shapes={'energies':(count,6),'weights':(count,6,4),'momentum':(count,8,8),'halves':(count,2),'forward':(count,),'physical':(count,6)}
    arrays={}
    for name,shape in shapes.items():
        file=folder/(name+'.npy');arrays[name]=np.load(file,mmap_mode='r+') if file.exists() else np.lib.format.open_memmap(file,mode='w+',dtype=float,shape=shape)
        if arrays[name].shape!=shape or arrays[name].dtype!=np.float64:raise ValueError('joint checkpoint array mismatch')
    if args.seed_checkpoint and state['completed_q']==0:
        original_path=Path(args.seed_checkpoint);original=json.loads(original_path.read_text());old_contract=original['contract']
        old_parameters=old_contract['parameters'];new_parameters={k:v for k,v in contract['parameters'].items() if k!='source_backend'}
        if old_parameters!=new_parameters or old_contract['specification']!=spec:raise ValueError('legacy checkpoint physical/numerical settings differ')
        if original_path.resolve()==path.resolve():raise ValueError('seed must be copied into a new checkpoint')
        snapshot=ROOT/config['seed_runner_snapshot']
        for name,expected in old_contract['source'].items():
            candidate=snapshot if name=='benchmarks/repair_b7_joint.py' else ROOT/name
            if sha256(candidate)!=expected:raise ValueError('legacy source provenance mismatch: '+name)
        before=original_path.parents[2]/'protected_before.json';protected=json.loads(before.read_text())['files']
        physical_name='benchmark_results/si_eels_convergence/primitive_force_constants.npz'
        if protected[physical_name]['sha256']!=contract['physical_inputs']['primitive_force_constants']:
            raise ValueError('legacy force-constant bytes changed; prefix cannot be reused')
        completed=int(original['completed_q'])
        if not 0<=completed<=count or original['total_q']!=count:raise ValueError('legacy prefix count is invalid')
        copied={}
        for name,shape in shapes.items():
            old=np.load(original_path.parent/(name+'.npy'),mmap_mode='r')
            if old.shape!=shape or old.dtype!=np.float64:raise ValueError('legacy array shape/dtype mismatch')
            prefix=old[:completed]
            if not np.all(np.isfinite(prefix)):raise ValueError('legacy committed prefix contains nonfinite entries')
            arrays[name][:completed]=prefix;arrays[name].flush()
            import hashlib
            copied[name]=hashlib.sha256(prefix.tobytes()).hexdigest()
        if np.any(arrays['weights'][:completed]<0):raise ValueError('legacy probability prefix is negative')
        state.update(completed_q=completed,cumulative_seconds=original['cumulative_seconds'],
                     squared_amplitude_error=original['squared_amplitude_error'],reference_squared_norm=original['reference_squared_norm'],
                     inherited_prefix={'state':str(original_path),'state_sha256':sha256(original_path),'completed_q':completed,
                     'array_prefix_sha256':copied,'original_runner_snapshot':str(snapshot),'original_runner_sha256':sha256(snapshot),
                     'input_audit':str(before),'input_audit_sha256':sha256(before),
                     'equivalence':'new terminal contraction tested against original scan and ordinary source stepper; same numerical kernels, stencils, endpoint convention, grid and Simpson rule'})
        json_write(path,state)
    spacing,offset,bounds,nx,det,pixels,halves,geometric=geometry_table(primitive,spec,n,[0,0,0])
    angle=20+np.linalg.norm(spacing)/2+1e-9
    positions=np.array(spec['positions_m']);padded=positions.copy();padded[:,2]+=pad
    covariance=np.array(spec['covariance_m2']);basis,_=primitive_atom_mapping(primitive,positions)
    relation=np.rint(primitive.reciprocal_m_inv[:,:2]/(2*np.pi/np.array(spec['transverse_lengths_m'])))
    limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'],100)
    potential=LobatoSilicon.from_abtem_table();kernels={};resumed=state['completed_q'];started=time.monotonic();last=started
    try:
        for iq in range(resumed,count):
            limits.check()
            if args.max_new_q and iq-resumed>=args.max_new_q:break
            q,qw=uniform_bz_nodes(n,iq,iq+1,[0,0,0]);omega,v=primitive.modes(q);omega=omega[0];v=v[0]
            physical=np.ones(6,bool)
            if iq==0:
                t=np.tile(np.eye(3),(len(primitive.positions_m),1))*np.repeat(np.sqrt(primitive.masses_kg),3)[:,None];t/=np.linalg.norm(t,axis=0)
                physical[np.argsort(np.sum(abs(v.conj().T@t)**2,axis=1))[-3:]]=False
            if np.any(omega[physical]<=0):raise ValueError('unstable physical source mode')
            shift=(-q[0]@relation)%1.;shift[shift>.5]-=1.;shift*=2*np.pi/np.array(spec['transverse_lengths_m'])
            qxy,index,theta=transverse_modes(grid,energy,angle,shift)
            d=displacement_for_q(primitive,positions,basis,q[0],omega[physical],v[:,physical],count)
            propagator=FirstOrderSourcePropagator(grid,energy,omega[physical],index,shift)
            if backend=='contracted':
                amplitude=numerical_source_contraction(potential,padded,d,covariance,qxy,propagator,length,config['samples'],kernels)
            else:
                _,source_j=numerical_transition_source(potential,padded,d,covariance,qxy,grid,length,config['samples'],kernels)
                amplitude=propagator.propagate_scan(source_j,length)
            energies,fast,fastq,_,_=born_batch(primitive,positions,covariance,spec['transverse_lengths_m'],length,pad,q,qw,energy,angle)
            near=np.argmin(np.linalg.norm(qxy[:,None]-fastq[0,None],axis=-1),axis=1)
            reference=fast[0,physical][:,near]
            state['squared_amplitude_error']+=float(np.sum(abs(amplitude-reference)**2));state['reference_squared_norm']+=float(np.sum(abs(reference)**2))
            probability=np.zeros((6,len(qxy)));probability[physical]=abs(amplitude)**2
            angular=qxy/k*1000;coordinate=np.rint(angular/spacing-offset).astype(int)
            indices=(coordinate[:,1]+bounds[1])*nx+coordinate[:,0]+bounds[0]
            arrays['energies'][iq]=energies[0];arrays['physical'][iq]=physical
            arrays['weights'][iq]=probability@det[indices]
            arrays['momentum'][iq]=(np.sum(probability,axis=0)@pixels[indices]).reshape(8,8)
            arrays['halves'][iq]=np.sum(probability,axis=0)@halves[indices]
            arrays['forward'][iq]=np.sum(probability[:,np.linalg.norm(qxy,axis=1)<1e-5])
            state['completed_q']=iq+1
            if (iq+1)%32==0:
                for a in arrays.values():a.flush();a._mmap.madvise(mmap.MADV_DONTNEED)
                state['cumulative_seconds']+=time.monotonic()-last;json_write(path,state);last=time.monotonic()
    except (TimeoutError,MemoryError) as exc:state['stop_reason']=str(exc)
    for a in arrays.values():a.flush()
    state['cumulative_seconds']+=time.monotonic()-last;state['latest_invocation_seconds']=time.monotonic()-started
    state['resumed_from_q']=resumed;state['status']='COMPLETE' if state['completed_q']==count else 'INCOMPLETE'
    json_write(path,state)
    new_count=state['completed_q']-resumed
    measured=state['latest_invocation_seconds']/max(new_count,1)
    result={'status':state['status'],'checkpoint':str(path),'completed_q':state['completed_q'],'total_q':count,
            'partial_prefix_complex_relative_error':float(np.sqrt(state['squared_amplitude_error']/max(state['reference_squared_norm'],1e-300))),
            'measured_seconds_per_q':measured,'estimated_total_source_seconds':measured*count,
            'source_backend':backend,'resumed_from_q':resumed,'completed_new_q':new_count,
            'estimated_remaining_seconds':measured*(count-state['completed_q']),
            'estimate_note':'pilot source work only; includes cold transverse kernel work; not project wall time or an observed complete result',
            'partial_weight_not_renormalized':float(np.sum(arrays['weights'][:state['completed_q'],:,0])),
            'global_acceptance_evaluated':False,'source_method':'independent inverse-FFT potential, finite fourth derivatives and actual Simpson/free source propagation; all six branches minus three Gamma rigid modes',
            'unchanged_acceptance':{'full_q_required':True,'intensity_and_spectral_L1':.01,'peak_shift_mev':.2,'complex_amplitude_relative':1e-4},
            'geometry':geometric,'resource_costs':limits.costs()}
    json_write(output/'results.json',result);print(result,flush=True)


if __name__=='__main__':main()
