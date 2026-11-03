#!/usr/bin/env python3
"""Bounded local SNAP pilot. Run one explicit stage/case per invocation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'benchmarks/90_workflow'))
import numpy as np
from scipy import constants
from ase.io import read, write
from workflow_io import ResourceLimits
from yoshioka.provenance import json_write, sha256, fingerprint
from yoshioka.snap_forces import SNAPForces, snap_metadata, central_snap_force_constants
from yoshioka.mod_forces import complete_force_constants, correct_sum_rules, force_constant_residuals
from yoshioka.lattice_dynamics import real_space_from_supercell, RealSpaceForceConstants, finite_periodic_modes

DEFAULT_OUTPUT = Path('/home/zuxian/Documents/PhD/Yoshioka/benchmark_results/15_snap_two_mode/si110_snap_T10_p2_v1')


def input_paths(out):
    return out/'inputs/SNAPotential.snapcoeff', out/'inputs/SNAPotential.snapparam'


def equilibrium(out, args, limits):
    if (out/'phonons/equilibrium.json').exists(): raise FileExistsError('equilibrium already recorded')
    coeff, param = input_paths(out)
    unit = read(out/'inputs/Si_110_temp300.lmp', format='lammps-data', atom_style='atomic')
    unit.set_masses(np.full(len(unit),28.085))
    metadata = snap_metadata(coeff,param)
    cfg = json.loads((out/'inputs/config.json').read_text())
    for name in ('coefficients_sha256','parameters_sha256'):
        if metadata[name] != cfg[name]: raise ValueError('supplied SNAP hash mismatch: '+name)
    with SNAPForces(unit,coeff,param) as calc:
        initial = calc.evaluate()
        relaxed, final = calc.relax_zero_pressure()
        metadata['lammps_version'] = calc.lammps_version
    write(out/'phonons/equilibrium.extxyz',relaxed)
    write(out/'phonons/equilibrium.lmp',relaxed,format='lammps-data',atom_style='atomic',masses=True)
    np.savez(out/'phonons/equilibrium.npz',cell_angstrom=np.asarray(relaxed.cell),positions_angstrom=relaxed.positions,mass_u=relaxed.get_masses())
    # Test a nonzero force, not only symmetry-forced zeros.
    test = relaxed.repeat((2,3,3))
    positions = test.positions.copy(); positions[0,0] += .013
    with SNAPForces(test,coeff,param) as calc:
        displaced = calc.evaluate(positions)
        eps = 1e-5
        plus, minus = positions.copy(), positions.copy()
        plus[0,0] += eps; minus[0,0] -= eps
        derivative = -(calc.evaluate(plus)['energy_ev']-calc.evaluate(minus)['energy_ev'])/(2*eps)
    result = dict(metadata=metadata,initial=initial,final=final,
                  force_energy_derivative_ev_per_angstrom=derivative,
                  force_ev_per_angstrom=float(displaced['forces_ev_per_angstrom'][0,0]),
                  derivative_absolute_error=float(abs(derivative-displaced['forces_ev_per_angstrom'][0,0])),
                  approximation='Static zero-pressure orthorhombic equilibrium volume at 10 K; no thermal or zero-point expansion',
                  isotope_convention='Uniform mass 28.085 u, not isotope disorder',
                  cell_angstrom=np.asarray(relaxed.cell),positions_angstrom=relaxed.positions)
    result['passed'] = (np.max(abs(final['forces_ev_per_angstrom']))<1e-6 and np.max(abs(final['pressure_bar_voigt']))<.1 and result['derivative_absolute_error']<1e-5)
    json_write(out/'phonons/equilibrium.json',result)
    print(json.dumps(result,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)),flush=True)
    if not result['passed']: raise ValueError('static equilibrium/force check failed')


def force_constants(out,args,limits):
    unit = read(out/'phonons/equilibrium.extxyz')
    reps = tuple(args.fc_repetitions)
    tag = 'fc_'+'x'.join(map(str,reps))+'_d'+format(args.displacement,'.4f')
    folder = out/'phonons'/tag
    folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists(): raise FileExistsError('completed FC case exists')
    coeff,param = input_paths(out)
    columns,supercell,cells = central_snap_force_constants(unit,reps,args.displacement,coeff,param,limits)
    raw = complete_force_constants(columns,reps)
    corrected = correct_sum_rules(raw)
    np.savez_compressed(folder/'force_constants.npz',columns_raw_ev_a2=columns,raw_ev_a2=raw,corrected_ev_a2=corrected,cells=cells)
    record = dict(repetitions=reps,atom_count=len(supercell),displacement_angstrom=args.displacement,
                  raw_residuals=force_constant_residuals(raw),corrected_residuals=force_constant_residuals(corrected),
                  correction_relative_norm=float(np.linalg.norm(corrected-raw)/np.linalg.norm(raw)))
    json_write(folder/'corrections.json',record)
    model,range_record = real_space_from_supercell(unit,reps,corrected)
    model.save(folder/'model.npz')
    q = np.array(list(np.ndindex(*reps)))/np.asarray(reps)
    omega,vectors = model.modes(q)
    np.savez_compressed(folder/'commensurate_modes.npz',q_fractional=q,omega_signed_rad_s=omega)
    gamma,gv = model.modes([0,0,0])
    np.savez(folder/'gamma.npz',omega_signed_rad_s=gamma,eigenvectors=gv)
    finite = finite_periodic_modes(model,(1,1,4))
    np.savez_compressed(folder/'finite_periodic.npz',**finite)
    record.update(range=range_record,gamma_mev=gamma*constants.hbar/(constants.e*1e-3),
                  minimum_signed_mesh_mev=float(omega.min()*constants.hbar/(constants.e*1e-3)),
                  finite_minimum_mev=float(finite['omega_rad_s'].min()*constants.hbar/(constants.e*1e-3)),
                  status='COMPUTED; convergence requires independent comparisons')
    json_write(folder/'result.json',record)
    print(json.dumps(record,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)),flush=True)


def build_bath(out,args,limits):
    if (out/'phonons/bath.npz').exists(): raise FileExistsError('finite bath already recorded')
    from yoshioka.finite_snap_bath import finite_optical_bath
    from yoshioka.coupled_channels import channel_basis,thermal_channel_weights
    model=RealSpaceForceConstants.load(out/'phonons/fc_4x6x6_d0.0050/model.npz')
    bath,evidence=finite_optical_bath(model)
    np.savez_compressed(out/'phonons/bath.npz',**{k:v for k,v in vars(bath).items() if k!='labels'},**{k:v for k,v in evidence.items() if k!='inactive_covariance_m2'})
    quantum=constants.hbar*bath.omega_rad_s
    ratios=np.exp(-quantum/(10*constants.k))
    weights=thermal_channel_weights(channel_basis(2,2),quantum,10.)
    # 1-prod(1-r) loses tiny optical thermal mass to cancellation. This
    # equivalent expression retains the nonzero omitted probability at 10 K.
    omitted_vacuum=float(ratios.sum()-ratios.prod())
    record=dict(model_id=json.loads((out/'inputs/config.json').read_text())['model_id'],
                labels=bath.labels,quanta_mev=quantum/(1e-3*constants.e),
                thermal_ratios=ratios,thermal_weights_p2=weights,
                initial_states_executed=[[0,0]],omitted_initial_probability=omitted_vacuum,
                vacuum_weight=float(np.prod(1-ratios)),
                absolute_error_bound_any_probability_observable=omitted_vacuum,
                specimen_atoms=len(bath.positions_m),specimen_thickness_nm=float(bath.cell_m[2,2]*1e9),
                mode_gram=evidence['gram'],covariance_reconstruction_relative=float(np.linalg.norm(evidence['full_covariance_m2']-evidence['active_covariance_m2']-bath.inactive_covariance_m2)/np.linalg.norm(evidence['full_covariance_m2'])),
                boundary_conditions=evidence['boundary_conditions'],covariance_accounting=evidence['covariance_accounting'],
                omitted_rigid_translations=3,two_phonon_basis_note=evidence['two_phonon_basis_note'])
    json_write(out/'phonons/bath.json',record)
    print(json.dumps(record,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)),flush=True)


def load_bath(out):
    from yoshioka.finite_snap_bath import FiniteOpticalBath
    with np.load(out/'phonons/bath.npz') as a:
        bath=FiniteOpticalBath(**{k:a[k] for k in FiniteOpticalBath.__dataclass_fields__ if k!='labels'},labels=json.loads((out/'phonons/bath.json').read_text())['labels'])
    model=RealSpaceForceConstants.load(out/'phonons/fc_4x6x6_d0.0050/model.npz')
    return model,bath


def propagate(out,args,limits):
    from yoshioka.projected_material_action import ProjectedMaterialAction,field_preflight
    from yoshioka.checkpointed_channel_action import checkpointed_action
    from yoshioka.channel_spectra import transition_lines,broaden_transition_lines,angular_detector_masks
    from yoshioka.free_propagation import electron_wavenumber
    from yoshioka.coupled_channels import thermal_channel_weights
    model,bath=load_bath(out)
    name=f'p{args.p}_N{args.n}_F{args.fields}_S{args.steps}_D{args.dimension}'+('_'+args.case_tag if args.case_tag else '')
    folder=out/'cases'/name;folder.mkdir(exist_ok=True)
    if (folder/'exit.npz').exists(): raise FileExistsError('completed case is immutable; inspect its existing outputs')
    preflight=field_preflight(len(bath.positions_m),args.n,args.fields,args.steps,(args.p+1)*(args.p+2)//2,args.dimension)
    json_write(folder/'preflight.json',preflight)
    limits.check(preflight['estimated_peak_rss_bytes'])
    h=ProjectedMaterialAction(model,args.p,args.n,args.steps,args.fields,out/'fields',bath=bath,working_dimension=args.dimension,limits=limits)
    quantum=constants.hbar*bath.omega_rad_s
    thermal=thermal_channel_weights(h.basis,quantum,10.)
    ratios=np.exp(-quantum/(10*constants.k));omitted=float(ratios.sum()-ratios.prod())
    if omitted>1e-12: raise ValueError('vacuum-only initial-state execution exceeds frozen omitted-mass bound')
    initial=np.zeros(h.shape,complex);initial[0,args.n//2,args.n//2]=1
    sources=source_record(out)
    contract={'model':json.loads((out/'inputs/config.json').read_text())['model_id'],
              'case':name,'field_digest':h.field_digest,'bath_sha256':sha256(out/'phonons/bath.npz'),
              'source_sha256':sources['numerical_source_sha256'],'config_sha256':sha256(out/'inputs/config.json'),
              'basis':h.basis.tolist(),'initial_index':0,'initial_energy_j':0.,
              'thermal_weights':thermal.tolist(),'omitted_initial_probability':omitted}
    wave,state=checkpointed_action(h,folder/'checkpoint',contract,limits,checkpoint_slices=16,initial_wave=initial)
    json_write(folder/'manifest.json',{'contract':contract,'source':sources,'state':state,'preflight':preflight,'resources':limits.costs()})
    print(json.dumps({'case':name,'state':state,'resources':limits.costs()},default=str),flush=True)
    if state['status']!='COMPLETE': return
    angle,masks=angular_detector_masks(h.qxy,electron_wavenumber(h.energy_j))
    lines=transition_lines(wave[None],[0],h.basis,quantum,[thermal[0]],masks)
    spectrum=broaden_transition_lines(lines,np.linspace(-180,180,7201),2.)
    np.savez_compressed(folder/'exit.npz',exit_coefficients=wave[None],initial_indices=[0],basis=h.basis,
                        qxy_m_inv=h.qxy,angles_mrad=angle,detector_masks=masks,mode_quanta_j=quantum,
                        thermal_weights=thermal,executed_initial_weights=[thermal[0]],omitted_initial_probability=omitted)
    np.savez_compressed(folder/'spectra.npz',**lines,**spectrum)
    import csv
    with (folder/'channels.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['initial_a','initial_b','final_a','final_b','loss_mev','returning_state','detector_mrad','absolute_probability','fraction_of_inclusive_detector'])
        totals=lines['probability'].sum(axis=(0,1))
        for j,label in enumerate(h.basis):
            for d,detector in enumerate(['0-20','0-5','5-10','10-20']):
                value=lines['probability'][0,j,d]
                writer.writerow([0,0,*label,lines['energy_loss_mev'][0,j],bool(lines['returning_state'][0,j]),detector,value,value/totals[d]])
    columns=[spectrum['energy_mev']];headers=['energy_mev']
    for d,detector in enumerate(['0-20','0-5','5-10','10-20']):
        for kind in ('inclusive','changed','returning'):
            columns.append(spectrum[kind+'_per_mev'][d]);headers.append(detector+'_'+kind+'_per_mev')
        for j,label in enumerate(h.basis):
            columns.append(spectrum['components_per_mev'][0,j,d]);headers.append(detector+'_'+''.join(map(str,label))+'_per_mev')
    np.savetxt(folder/'spectra.csv',np.column_stack(columns),delimiter=',',header=','.join(headers),comments='')


def focused_checks(out,args,limits):
    from yoshioka.projected_material_action import ProjectedMaterialAction,phase_blocks,field_preflight
    from yoshioka.reciprocal_material import ReciprocalMaterialHamiltonian
    from yoshioka.coordinate_channel_action import CoordinateMaterialAction
    from yoshioka.coupled_channels import oscillator_phase_block,channel_basis
    model,bath=load_bath(out)
    result={}
    hs=ProjectedMaterialAction(model,2,3,17,17,out/'checks/fields',bath=bath,limits=limits)
    hf=ProjectedMaterialAction(model,2,3,17,17,out/'checks/fields',bath=bath,storage='full',limits=limits)
    old=ReciprocalMaterialHamiltonian(model,2,3,17,bath=bath,working_dimension=32)
    permutation=[list(map(tuple,old.basis)).index(tuple(row)) for row in hs.basis]
    rng=np.random.default_rng(918);wave=rng.normal(size=hs.shape)+1j*rng.normal(size=hs.shape);wave/=np.linalg.norm(wave)
    oldwave=wave[np.argsort(permutation)]
    index=7
    streamed=hs.apply(wave,index);full=hf.apply(wave,index);direct=old.apply(oldwave,index)[permutation]
    result['mapped_full_action_relative']=float(np.linalg.norm(streamed-full)/np.linalg.norm(full))
    result['independent_reciprocal_action_relative']=float(np.linalg.norm(streamed-direct)/np.linalg.norm(direct))
    for order in (24,32):
        control=CoordinateMaterialAction(model,2,3,17,bath=bath,quadrature_order=order)
        acted=control.apply(oldwave,index)[permutation]
        result[f'GH{order}_complex_action_relative']=float(np.linalg.norm(acted-streamed)/np.linalg.norm(streamed))
    # Maximum N33 transfer, not an artificially small oscillator argument.
    Q=np.array([2*np.pi*32/bath.cell_m[0,0],2*np.pi*32/bath.cell_m[1,1]])
    xi=Q@bath.displacement_m[0,:2,:]
    blocks={}
    for d in (32,48): blocks['D'+str(d)]=oscillator_phase_block(hs.basis,xi,working_dimension=d)
    for order in (24,32): blocks['GH'+str(order)]=oscillator_phase_block(hs.basis,xi,quadrature_order=order)
    result['maximum_N33_test_xi']=xi.tolist()
    result['matrix_elements']={key:float(np.linalg.norm(value-blocks['D48'])) for key,value in blocks.items()}
    np.savez(out/'checks/independent_action.npz',wave=wave,mapped=streamed,full=full,reciprocal=direct,xi=xi,**blocks)
    result['passed']=all(result[k]<1e-10 for k in ('mapped_full_action_relative','independent_reciprocal_action_relative','GH24_complex_action_relative','GH32_complex_action_relative')) and max(result['matrix_elements'].values())<1e-10
    result['preflights']=[field_preflight(16,n,f,f,6) for n,f in ((33,769),(65,1537),(97,3073))]
    json_write(out/'checks/action.json',result)
    print(json.dumps(result),flush=True)
    if not result['passed']: raise ValueError('independent nonlinear action checks failed')


def summarize(out,args,limits):
    from analysis import summarize_results
    summarize_results(out)


def source_record(out):
    files = sorted([*ROOT.glob('yoshioka/*.py'),*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'config.json'])
    hashes = {str(f.relative_to(ROOT)):sha256(f) for f in files}
    digest = fingerprint(hashes)
    dest = out/'provenance'/'source'/digest
    if not dest.exists():
        for f in files:
            target = dest/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    record = dict(source_sha256=digest,numerical_source_sha256=fingerprint({k:v for k,v in hashes.items() if k.startswith('yoshioka/') or k.endswith('/run.py') or k.endswith('/config.json')}),files=hashes,
                  revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                  changes=subprocess.check_output(['git','-C',str(ROOT),'status','--short'],text=True),
                  python=sys.version,executable=sys.executable,
                  packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','ase','lammps','abtem','phonopy','psutil')},
                  threads={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    json_write(dest/'manifest.json',record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['equilibrium','fc','bath','propagate','checks','summarize'])
    parser.add_argument('--case-tag',default='')
    parser.add_argument('--p',type=int,default=2)
    parser.add_argument('--n',type=int,default=33)
    parser.add_argument('--fields',type=int,default=769)
    parser.add_argument('--steps',type=int,default=769)
    parser.add_argument('--dimension',type=int,default=32)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--fc-repetitions',type=int,nargs=3,default=[4,6,6])
    parser.add_argument('--displacement',type=float,default=.005)
    args = parser.parse_args(); out=args.output
    limits=ResourceLimits(seconds=570,memory_gib=4)
    def alarm(*_): raise TimeoutError('590 second hard invocation alarm')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(590)
    stamp=time.strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid())
    log = out/'logs'/f'{stamp}_{args.stage}.json'
    record={'stage':args.stage,'arguments':vars(args),'source':source_record(out),'status':'RUNNING'}
    json_write(log,record)
    try:
        {'equilibrium':equilibrium,'fc':force_constants,'bath':build_bath,'propagate':propagate,'checks':focused_checks,'summarize':summarize}[args.stage](out,args,limits)
        record['status']='COMPLETE'
    except BaseException as exc:
        record.update(status='FAILED',exception=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        record['resources']=limits.costs();json_write(log,record)
        print(json.dumps({'invocation':str(log),'status':record['status'],'resources':record['resources']}),flush=True)

if __name__=='__main__': main()
