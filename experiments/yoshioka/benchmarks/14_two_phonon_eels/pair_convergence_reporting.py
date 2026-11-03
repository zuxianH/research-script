"""Verified numerical-case inventory and scientific figures; no propagation.

Repeated mode-pair calculations at different discretizations are numerical
refinements, not new independent BZ production. Partial panels remain partial.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import constants

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.workflow_io import ROOT
from yoshioka.free_propagation import TransverseGrid,electron_wavenumber
from yoshioka.dynamical_source import full_transverse_vectors
from yoshioka.electron_phonon import LobatoSilicon
from benchmarks.two_phonon_analysis import WINDOWS
from benchmarks.approximate_eels import DETECTORS

DEFAULT=ROOT/'benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(a):
    if isinstance(a,np.ndarray):return clean(a.tolist())
    if isinstance(a,np.generic):return clean(a.item())
    if isinstance(a,float) and not np.isfinite(a):return None
    if isinstance(a,dict):return {str(k):clean(v) for k,v in a.items()}
    if isinstance(a,(list,tuple)):return [clean(v) for v in a]
    return a


def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(data),indent=2,sort_keys=True)+'\n');tmp.replace(path)


class Inputs:
    def __init__(self):self.files={}
    def read(self,path):
        path=Path(path);self.files[str(path.resolve())]=sha(path);return json.loads(path.read_text())
    def pin(self,path):
        path=Path(path);digest=sha(path);self.files[str(path.resolve())]=digest;return digest


def source_index(output):
    paths=set()
    for folder in [output/'source_versions',output/'electron_audit/source_snapshots',output/'starting_source']:
        if folder.exists():paths.update(p for p in folder.rglob('*.py') if p.is_file())
    for folder in [ROOT/'yoshioka',ROOT/'benchmarks',ROOT/'benchmarks/14_two_phonon_eels']:
        paths.update(p for p in folder.glob('*.py') if p.is_file())
    index={}
    for p in paths:index.setdefault(sha(p),[]).append(str(p.resolve()))
    return index


def source_coverage(contract,settings,index,inputs):
    expected=dict(contract.get('kernels',{}));expected.update(contract.get('planar_kernels',{}))
    if contract.get('background_setup_sha256'):expected['background_setup_entrypoint']=contract['background_setup_sha256']
    if contract.get('entrypoint_sha256'):expected['runner_entrypoint']=contract['entrypoint_sha256']
    elif settings.get('runner_creation_sha256'):expected['runner_creation_from_numerical_record']=settings['runner_creation_sha256']
    result={}
    for name,digest in expected.items():
        available=sorted(set(index.get(digest,[])))
        # Prefer preserved archives to live source when both are available.
        available.sort(key=lambda p:('source_versions' not in p and 'source_snapshots' not in p,p))
        for p in available[:1]:inputs.pin(p)
        result[name]=dict(expected_sha256=digest,verified_available_sources=available,
            source_snapshot_does_not_imply_creation_time_capture=True,
            preserved_archive_available=any('source_versions' in p or 'source_snapshots' in p or 'starting_source' in p for p in available),
            current_source_matches=bool((ROOT/name).is_file() and sha(ROOT/name)==digest))
    return result


def kinetic_phase(grid,word_shifts,final_shifts,step_m):
    q=full_transverse_vectors(grid);shifts=np.concatenate([word_shifts,final_shifts])
    squared=np.sum((q[None]+shifts[:,None])**2,axis=-1)
    at=np.unravel_index(np.argmax(squared),squared.shape)
    return dict(maximum_full_step_phase_rad=float(step_m*squared[at]/(2*electron_wavenumber(60000*constants.elementary_charge))),
        maximum_half_step_phase_rad=float(step_m*squared[at]/(4*electron_wavenumber(60000*constants.elementary_charge))),
        channel_index=int(at[0]),reciprocal_flat_index=int(at[1]),
        interpretation='Largest kinetic phase of a retained physical beam plus its actual word/final Bloch shift. This is a diagnostic; no new acceptance threshold is imposed.')


def load_cases(output,inputs,index):
    cases=[];raw_by_case={};source_metadata={};parameter=LobatoSilicon.from_abtem_table();table=inputs.pin(parameter.source_path)
    for folder in sorted((output/'electron_cases').iterdir()):
        statepath=folder/'state.json'
        if not statepath.exists():continue
        state=inputs.read(statepath);contract=state['contract'];panelpath=Path(contract['panel'])
        if inputs.pin(panelpath)!=contract['panel_sha256']:raise ValueError('Changed frozen panel '+folder.name)
        panel=json.loads(panelpath.read_text());panel=panel.get('pairs',panel.get('panel',panel));requested=len(panel)
        if table!=contract['lobato_file_sha256']:raise ValueError('Changed physical Lobato table')
        settings=inputs.read(folder/'numerical_settings.json') if (folder/'numerical_settings.json').exists() else {}
        for name,digest in state.get('setup_sha256',{}).items():
            if inputs.pin(folder/name)!=digest:raise ValueError('Changed guarded case setup '+folder.name)
        if state.get('numerical_settings_sha256') and state['numerical_settings_sha256']!=inputs.pin(folder/'numerical_settings.json'):
            raise ValueError('Changed guarded numerical settings')
        config=inputs.read(folder/'config.json') if (folder/'config.json').exists() else None
        derived=dict(retained_N=contract['retained'],product_M=contract['product'],
            reciprocal_transfer_halfwidth=(contract['product']-1)//2,
            exact_projected_product_requirement_M_ge_2N_minus1=contract['product']>=2*contract['retained']-1,
            retained_inverse_DFT_spacing_angstrom=np.asarray(settings.get('cell_lengths_angstrom',[]))/contract['retained'],
            interpretation='Physical reciprocal spacing is2pi/L at every N. Retained inverse-DFT spacing and product real-space spacing are separate; N changes retained cutoff and intermediate support.')
        for name in ['criteria.json','inputs.json','covariance/T300_mesh32.json','background_only.json']:
            if (folder/name).exists():inputs.read(folder/name)
        physical={}
        for name,digest in contract.get('input_sha256',{}).items():
            actual=inputs.pin(ROOT/name)
            if actual!=digest:raise ValueError('Changed physical benchmark input '+name)
            physical[name]=actual
        invocations=[dict(file=str(p),**inputs.read(p)) for p in sorted(folder.glob('invocation*.json'))]
        rows=[];arrays=[];source_ids=[]
        for name,digest in sorted(state['files'].items()):
            path=folder/name
            if inputs.pin(path)!=digest:raise ValueError('Changed committed pair array '+str(path))
            with np.load(path) as z:
                source=str(z['source_path']);sourcehash=str(z['source_sha256'])
                if inputs.pin(source)!=sourcehash:raise ValueError('Changed original mode-pair source')
                if sourcehash not in source_metadata:
                    with np.load(source) as original:
                        source_metadata[sourcehash]={k:original[k].copy() for k in ['q_fractional','branches','inverse_marginals','energies_mev']}
                for key,expected in source_metadata[sourcehash].items():
                    if not np.array_equal(z[key],expected):raise ValueError('Changed physical source identity '+key)
                for key in ['direct_amplitudes','sequential_amplitudes']:
                    if z[key].shape!=(4,contract['retained'],contract['retained']):raise ValueError('Lost sign/component amplitude')
                if 'total_amplitudes' in z and not np.allclose(z['total_amplitudes'],z['direct_amplitudes']+z['sequential_amplitudes'],rtol=1e-13,atol=1e-28):raise ValueError('Coherent total amplitude mismatch')
                scale=max(float(np.max(z['direct_weights']+z['sequential_weights'])),1e-300)
                if np.max(abs(z['weights']-z['direct_weights']-z['sequential_weights']-z['interference_weights']))>2e-12*scale:raise ValueError('Lost signed intensity component identity')
                identity=hashlib.sha256(z['q_fractional'].tobytes()+z['branches'].tobytes()).hexdigest()
                step=settings.get('depth_step_angstrom',0)*1e-10
                actual_step=step
                if 'depth_nodes_m' in z:actual_step=float(np.max(np.diff(z['depth_nodes_m'])))
                grid=TransverseGrid(*(np.asarray(settings['cell_lengths_angstrom'])*1e-10),contract['retained'],contract['retained'])
                phase=kinetic_phase(grid,z['word_shifts'],z['final_shifts'],actual_step)
                loss=z['energies_mev']>0
                component={k:np.sum(z[k]*loss[:,None],axis=0) for k in ['direct_weights','sequential_weights','interference_weights','weights']}
                row=dict(file=str(path),sha256=digest,source_path=source,source_sha256=sourcehash,
                    physical_q_branch_identity=identity,q_fractional=z['q_fractional'],branches=z['branches'],inverse_marginals=z['inverse_marginals'],
                    all_four_direct_and_sequential_complex_signs_preserved=True,explicit_total_amplitudes_preserved='total_amplitudes' in z,
                    coherent_total_available=True,word_coefficients_preserved='word_coefficients' in z,
                    measured_wall_seconds=float(z.get('measured_wall_seconds',z['elapsed_seconds'])),internal_elapsed_seconds=float(z['elapsed_seconds']),
                    peak_process_rss_mib=float(z.get('peak_process_rss_mib',np.nan)),maximum_actual_interval_angstrom=actual_step*1e10,
                    actual_intervals=int(z['actual_intervals']) if 'actual_intervals' in z else contract['steps'],
                    maximum_local_diagonal_phase=float(z['maximum_local_diagonal_phase']) if 'maximum_local_diagonal_phase' in z else None,
                    kinetic_phase=phase,component_raw_loss=component,
                    mapped_field_page_eviction_steps=int(z['mapped_field_page_eviction_steps']) if 'mapped_field_page_eviction_steps' in z else 0)
                rows.append(row);source_ids.append(identity)
                arrays.append(dict(sourcehash=sourcehash,source=source,energies=z['energies_mev'].copy(),weights=z['weights'].copy(),components=component))
        raw_by_case[folder.name]=arrays
        if len(rows)!=state['completed_pairs']:raise ValueError('Committed count does not match arrays')
        complete=len(rows)==requested
        status='COMPLETE_DECLARED_DIAGNOSTIC_PANEL' if complete else 'PARTIAL_DIAGNOSTIC_PANEL'
        if not rows:status='NO_COMMITTED_EVALUATIONS'
        last=invocations[-1]['status'] if invocations else None
        if last=='PARTIAL_RESOURCE_LIMIT':status+=';RESOURCE_LIMIT_RECORDED'
        if not invocations and not complete:status+=';NO_TERMINAL_INVOCATION_AT_SNAPSHOT'
        coverage=source_coverage(contract,settings,index,inputs)
        cases.append(dict(case=folder.name,folder=str(folder),status=status,creation_guard_version=int(contract.get('guard_version',1)),
            contract=contract,numerical_settings=settings,derived_numerical_settings=derived,config=config,completed_numerical_evaluations=len(rows),declared_panel_pairs=requested,
            independent_bz_production_pairs=0,unique_physical_q_branch_sources=len(set(source_ids)),
            all_committed_array_hashes_verified=True,source_coverage=coverage,
            missing_source_hash_archives=[k for k,v in coverage.items() if not v['verified_available_sources']],
            guard_scope=dict(explicit_physical_input_hashes=bool(contract.get('input_sha256')),explicit_detector_definitions='detectors' in contract,
                explicit_runner_entrypoint_hash='entrypoint_sha256' in contract,local_setup_hashes=bool(state.get('setup_sha256')),
                numerical_record_hash=bool(state.get('numerical_settings_sha256')),
                serialized_contract_container_canonicalization=contract.get('guard_version',1)>=4,
                earliest_guard_limitation='Absent creation-time fields remain absent; current input audits and source archives are retrospective evidence, not backdated contracts.'),
            rows=rows,invocations=invocations,costs=dict(summed_committed_pair_wall_seconds=sum(r['measured_wall_seconds'] for r in rows),
                recorded_initial_preparation_seconds=settings.get('product_preparation_seconds'),
                summed_reported_invocation_elapsed_seconds=sum(i.get('elapsed_seconds',0) for i in invocations),
                failed_resource_attempt_seconds=sum(i.get('elapsed_seconds',0) for i in invocations if i['status']=='PARTIAL_RESOURCE_LIMIT'),
                maximum_recorded_process_rss_mib=max([r['peak_process_rss_mib'] for r in rows]+[i.get('peak_process_rss_mib',0) for i in invocations],default=None),
                scope='Committed-pair wall, preparation, and invocation elapsed are distinct overlapping measurements; they are not added into a fabricated elapsed total. Failed attempts and preparation are retained.')))
    return cases,raw_by_case


def build_commands(cases,output):
    entries=[]
    for c in cases:
        k=c['contract'];argv=['.venv/bin/python','-m','benchmarks.pair_electron_diagnostics','--output',str(output/'electron_cases'),'--case',c['case'],'--panel',k['panel'],
            '--retained',str(k['retained']),'--product',str(k['product']),'--steps',str(k['steps']),'--substeps',str(k['local_substeps'])]
        for key,flag in [('born','--born'),('legacy','--legacy')]:
            if k.get(key):argv.append(flag)
        if not k['require_dealiased']:argv.append('--circular')
        if k.get('field_halfwidth') is not None:argv+=['--field-halfwidth',str(k['field_halfwidth'])]
        if k.get('representation'):
            argv.append('--planar')
            if 'interval means' in k['representation']:argv.append('--cell-average')
        if k.get('setup_strategy'):argv.append('--background-only')
        seconds=next((i.get('configured_runtime_seconds') for i in reversed(c['invocations']) if i.get('configured_runtime_seconds')),600)
        argv+=['--seconds',str(seconds)]
        current=(k.get('guard_version',1)>=4 and k.get('entrypoint_sha256')==sha(ROOT/'benchmarks/14_two_phonon_eels/pair_electron_diagnostics.py'))
        current &= all(v['current_source_matches'] for name,v in c['source_coverage'].items() if name not in ['runner_entrypoint','background_setup_entrypoint'])
        if k.get('background_setup_sha256'):current &= k['background_setup_sha256']==sha(ROOT/'benchmarks/14_two_phonon_eels/pair_background_setup.py')
        legacy_hash='14d5dbd1d5b22ce415afedeb130dca67fb14f0b66433b99a980363c000508948'
        compatible=(k.get('guard_version')==3 and k.get('entrypoint_sha256')==legacy_hash and (output/'resume_legacy_guard.py').exists())
        compatible &= all(v['current_source_matches'] for name,v in c['source_coverage'].items() if name not in ['runner_entrypoint','background_setup_entrypoint'])
        if k.get('background_setup_sha256'):compatible &= k['background_setup_sha256']==sha(ROOT/'benchmarks/14_two_phonon_eels/pair_background_setup.py')
        fresh=argv.copy();fresh[fresh.index('--case')+1]=c['case']+'_fresh_current_guard'
        entries.append(dict(case=c['case'],working_directory=str(ROOT),environment={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(ROOT)},
            reconstructed_from_creation_contract_not_shell_history=True,
            exact_case_resume_argv=argv if current else None,current_runner_can_resume_existing_contract=bool(current),
            recorded_guard3_compatibility_resume_argv=([argv[0],str(output/'resume_legacy_guard.py'),*argv[3:]] if compatible else None),
            compatibility_scope='Only exact14dguard3caller with recorded tuple-to-list metadata normalization; detector values and strict original physical/source/raw guards retained. Legacy state is never rewritten.',
            fresh_current_source_replay_argv=fresh,
            source_versions_required={name:v['expected_sha256'] for name,v in c['source_coverage'].items()},
            warning='Historical guard/backend contracts must reject changed source. A fresh replay uses a new identifier and is not authorized convergence production; settings reconstructed here do not recover missing original shell/max-new arguments.'))
    return dict(refresh_reporting_argv=['.venv/bin/python','benchmarks/14_two_phonon_eels/pair_convergence_reporting.py','--output',str(output)],
        commands=entries,execution='Commands are records only; this reporting utility never executes propagation.')


def raw_windows(energies,weights):
    out=[]
    for name,(lo,hi) in WINDOWS.items():
        select=(energies>=lo)&(energies<hi)
        if name=='all_loss':select=energies>0
        if name=='all_gain':select=energies<0
        out.append(np.sum(weights*select[:,None],axis=0))
    return np.asarray(out)


def comparisons_from_cases(cases,raw):
    pairs=[('diag_n17_m35_z769','diag_n25_m49_z769'),('diag_n25_m49_z769','diag_n33_m65_z769'),
        ('diag_n33_m65_z769','diag_n49_m99_z769'),('diag_n49_m99_z769','loc_n65_m135_z769_g2'),
        ('rep_n33_m65_z769_g2','rep_n49_m99_z769_g2'),
        ('diag_n49_m99_z769','loc_n49_m99_z1537_g2'),('diag_n49_m99_z769','loc_n49_m99_z769_rk2_g2'),
        ('born_diag_n33_m65_z769','born_diag_n65_m135_z769_g3'),
        ('stress_n33_m65_z769_rk1','stress_n33_m65_z769_rk4'),
        ('avg_diag_n17_m35_z769','diag_n17_m35_z769'),('avg_diag_n33_m65_z1537_streamed','diag_n33_m65_z1537_g2'),
        ('smoke_n17_m35_z769','smoke_n17_m35_z769_evicted'),('loc_n49_m99_z1537_g2','loc_n65_m135_z1537_streamed'),
        ('loc_n65_m135_z1537_streamed','loc_n97_m195_z1537_evicted'),('loc_n97_m195_z1537_evicted','loc_n97_m195_z3073_evicted'),
        ('diag_n33_m65_z769','diag_n33_m65_z1537_g2'),('avg_diag_n17_m35_z385','avg_diag_n17_m35_z769'),
        ('avg_diag_n17_m35_z769','avg_diag_n17_m35_z1537'),('avg_diag_n33_m65_z769','avg_diag_n33_m65_z1537_streamed')]
    lookup={c['case']:c for c in cases};result=[]
    for a,b in pairs:
        if a not in lookup or b not in lookup:continue
        left={r['sourcehash']:r for r in raw[a]};right={r['sourcehash']:r for r in raw[b]};common=left.keys()&right.keys();rows=[]
        for key in common:
            x,y=left[key],right[key]
            if not np.array_equal(x['energies'],y['energies']):raise ValueError('Unmatched comparison energy')
            lx=x['components']['weights'];ly=y['components']['weights'];delta=ly-lx
            wx=raw_windows(x['energies'],x['weights']);wy=raw_windows(y['energies'],y['weights']);wd=wy-wx
            wr=np.divide(wd,wy,out=np.full_like(wd,np.nan),where=abs(wy)>1e-18)
            finite=np.isfinite(wr);worst=None
            if finite.any():
                at=np.unravel_index(np.nanargmax(abs(wr)),wr.shape)
                worst=dict(window=list(WINDOWS)[at[0]],detector=DETECTORS[at[1]],signed_relative_change=wr[at],absolute_difference=wd[at],fine_raw_weight=wy[at])
            rows.append(dict(source_sha256=key,source_path=x['source'],coarse_loss=lx,fine_loss=ly,absolute_loss_difference=delta,
                signed_relative_loss_difference=np.divide(delta,ly,out=np.full_like(delta,np.nan),where=abs(ly)>1e-18),
                coarse_all_window_weights=wx,fine_all_window_weights=wy,all_window_differences=wd,signed_relative_all_window_changes=wr,
                near_zero_absolute_difference_rule=(abs(wy)>1e-18)|(abs(wd)<=1e-18),
                worst_finite_relative_window_change=worst,coarse_all_sign_weights=x['weights'],fine_all_sign_weights=y['weights']))
        result.append(dict(coarse=a,fine=b,matched_sources=len(rows),coarse_count=lookup[a]['completed_numerical_evaluations'],fine_count=lookup[b]['completed_numerical_evaluations'],
            individual_selected_pair_results=rows,coarse_costs=lookup[a]['costs'],fine_costs=lookup[b]['costs'],
            interpretation='Individual fixed-source diagnostic values only. No selected-node mean is labeled a bulk integral and different-N timings are not matched-target speedups.'))
    return result


def figures(output,cases,raw,inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.18})
    case_lookup={c['case']:c for c in cases};panel=inputs.read(output/'localization_panel2.json')['pairs'];ids=[x['source_sha256'] for x in panel]
    series=[('769 steps','-',[(17,'diag_n17_m35_z769'),(25,'diag_n25_m49_z769'),(33,'diag_n33_m65_z769'),(49,'diag_n49_m99_z769'),(65,'loc_n65_m135_z769_g2')]),
        ('1537 steps','--',[(33,'diag_n33_m65_z1537_g2'),(49,'loc_n49_m99_z1537_g2'),(65,'loc_n65_m135_z1537_streamed'),(97,'loc_n97_m195_z1537_evicted')]),
        ('3073 steps',':',[(97,'loc_n97_m195_z3073_evicted')])]
    components=[('direct_weights','Direct','#2878b5'),('sequential_weights','Sequential','#e28726'),('interference_weights','Signed interference','#419b73'),('weights','Coherent total','#222222')]
    fig,axes=plt.subplots(2,2,figsize=(12.2,7.5),sharex=True)
    for row,identity in enumerate(ids[:2]):
        for col,det in enumerate((0,1)):
            ax=axes[row,col];plotted=[]
            for label,style,points in series:
                for component,name,color in components:
                    xs=[];ys=[]
                    for n,case in points:
                        pair=next((r for r in raw.get(case,[]) if r['sourcehash']==identity),None)
                        if pair is not None:xs.append(n);ys.append(pair['components'][component][det])
                    if xs:
                        plotted.extend(ys);ax.plot(xs,ys,style,color=color,marker='o' if style=='-' else ('s' if style=='--' else '^'),markersize=4,lw=1.5,label=name if row==0 and col==0 and style=='-' else None)
            magnitude=np.abs(plotted);positive=magnitude[magnitude>0]
            lo=int(np.floor(np.log10(np.min(positive))))-1;hi=int(np.ceil(np.log10(np.max(positive))))
            first=lo+min(2,max(1,(hi-lo)//3))
            exponents=np.unique(np.rint(np.linspace(first,hi,3)).astype(int));ticks=10.**exponents
            ax.set_yscale('symlog',linthresh=10.**lo,linscale=.7)
            ax.set_yticks(np.r_[-ticks[::-1],0,ticks]);limit=max(max(magnitude)*1.6,10.**hi*1.2);ax.set_ylim(-limit,limit)
            ax.set_xticks([17,33,49,65,97]);ax.set_xlim(14,101)
            ax.set_title(('Dominant selected pair' if row==0 else 'Cancelling selected pair')+(' · 0–20 mrad' if det==0 else ' · 0–5 mrad'),fontsize=11)
            if col==0:ax.set_ylabel('Weighted raw loss per fixed pair')
            if row==1:ax.set_xlabel('Retained electron grid N (fixed physical cell)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.99))
    fig.text(.5,.02,'Solid: 769 depth steps · dashed: 1537 · dotted triangle: 3073. Signed interference is retained. Two diagnostic sources; no BZ average.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.89,bottom=.12,hspace=.29,wspace=.25);path=output/'pair_component_support.png';fig.savefig(path,dpi=170);plt.close(fig)
    figs=[path]
    audit=inputs.read(output/'electron_audit/results.json');validation=inputs.read(output/'electron_audit/planar_validation_results.json')
    fig,axes=plt.subplots(2,2,figsize=(12.2,8.2))
    records=validation['rigid_translation']['records'];vals=[r['identity_relative_l2'] for r in records]
    axes[0,0].bar(range(4),vals,color=['#2878b5','#419b73','#d45a52','#b97845']);axes[0,0].set_yscale('log');axes[0,0].set_ylim(3e-17,3)
    axes[0,0].set_xticks(range(4),['Planar\nprojected','Uniform\nprojected','Circular\nRK4','Circular\nlocal exact']);axes[0,0].set_ylabel('Relative complex identity residual');axes[0,0].set_title('Rigid-translation identity: projection matters')
    for i,v in enumerate(vals):axes[0,0].text(i,v*1.7,f'{v:.2g}',ha='center',fontsize=8)
    action=[audit['one_slice_potential_action'][0],*audit['one_slice_source_action']];vals=[100*r['circular_vs_linear_convolution_relative_error'] for r in action]
    axes[0,1].bar(range(3),vals,color=['#7b91ab','#e28726','#419b73']);axes[0,1].set_xticks(range(3),['Background','Linear source','Quadratic source']);axes[0,1].set_ylabel('Circular vs projected action error (%)');axes[0,1].set_ylim(0,max(vals)*1.3);axes[0,1].set_title('One-slice action test at N=97')
    for i,v in enumerate(vals):axes[0,1].text(i,v+.25,f'{v:.2f}%',ha='center',fontsize=9)
    weak=validation['actual_si_weak']['results'][0];vals=[weak['quadratic_regular_plus_contact_weak_relative_error'],weak['quadratic_without_contacts_relative_error']]
    axes[1,0].bar(range(2),vals,color=['#2878b5','#d45a52']);axes[1,0].set_yscale('log');axes[1,0].set_ylim(1e-15,2);axes[1,0].set_xticks(range(2),['Regular + exact contacts','Contacts omitted']);axes[1,0].set_ylabel('Quadratic weak-field relative error');axes[1,0].set_title('Independent actual-Si Fourier field check')
    for i,v in enumerate(vals):axes[1,0].text(i,v*1.7,f'{v:.2g}',ha='center',fontsize=9)
    ax=axes[1,1]
    for i,identity in enumerate(ids[:2]):
        values=[]
        for steps,case in [(385,'avg_diag_n17_m35_z385'),(769,'avg_diag_n17_m35_z769'),(1537,'avg_diag_n17_m35_z1537')]:
            pair=next((r for r in raw.get(case,[]) if r['sourcehash']==identity),None)
            if pair:values.append((steps,pair['components']['weights'][0]))
        if len(values)>=2:
            ref=values[-1][1];ax.plot([a for a,b in values],[100*(b-ref)/ref for a,b in values],'-o',label='Dominant pair' if i==0 else 'Cancelling pair',markersize=4)
    ax.set_xlabel('Nominal depth steps (fixed N=17, M=35)');ax.set_ylabel('Primary loss change vs finest shown (%)');ax.set_title('Analytic interval means + exact contacts');ax.legend(frameon=False,fontsize=9)
    fig.text(.5,.018,'Identity and field tests have their declared reduced scopes. One-slice errors and two-pair depth changes are not full-spectrum error bounds.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.12,top=.94,hspace=.48,wspace=.28);path=output/'numerical_error_diagnosis.png';fig.savefig(path,dpi=170);plt.close(fig);figs.append(path)
    return figs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--no-figures',action='store_true');a=p.parse_args();started=time.monotonic();inputs=Inputs();index=source_index(a.output);cases,raw=load_cases(a.output,inputs,index)
    reconstruction=a.output/'source_versions/reconstructed_entrypoints.json'
    reconstruction_record=inputs.read(reconstruction) if reconstruction.exists() else None
    compatibility_script=a.output/'resume_legacy_guard.py'
    if compatibility_script.exists():inputs.pin(compatibility_script)
    compatibility_attempts=[inputs.read(p) for p in sorted((a.output/'guard_attempts').glob('compatibility_*.json'))]
    exercise_path=a.output/'guard_exercise.json';guard_exercise=inputs.read(exercise_path) if exercise_path.exists() else None
    if (a.output/'guard_roundtrip_test.log').exists():inputs.pin(a.output/'guard_roundtrip_test.log')
    qids={r['physical_q_branch_identity'] for c in cases for r in c['rows']};sids={r['source_sha256'] for c in cases for r in c['rows']};counts=Counter(c['creation_guard_version'] for c in cases)
    inv=dict(scope='New electron diagnostic cases only; finite nonlinear, thermal, B7 and historical bulk production remain separate.',
        numerical_evaluations=sum(c['completed_numerical_evaluations'] for c in cases),unique_physical_q_branch_sources=len(qids),unique_original_source_file_hashes=len(sids),
        new_independent_bz_production_pairs=0,cases=len(cases),guard_version_case_counts=counts,
        complete_declared_diagnostic_panels=sum(c['completed_numerical_evaluations']==c['declared_panel_pairs'] for c in cases),
        all_committed_arrays_four_sign_D_and_S_preserved=all(r['all_four_direct_and_sequential_complex_signs_preserved'] for c in cases for r in c['rows']),
        source_coverage_missing=[dict(case=c['case'],missing=c['missing_source_hash_archives']) for c in cases if c['missing_source_hash_archives']],
        case_inventory=cases,guard_exercise=guard_exercise,legacy_guard_compatibility_attempts=compatibility_attempts,reconstructed_entrypoint_record=reconstruction_record,source_sha256=sha(__file__))
    costs=dict(status='NO_THIN_BULK_ACCEPTANCE_FROM_DIAGNOSTIC_TIMINGS',
        total_committed_pair_wall_seconds=sum(c['costs']['summed_committed_pair_wall_seconds'] for c in cases),
        total_recorded_initial_preparation_seconds=sum(c['costs']['recorded_initial_preparation_seconds'] or 0 for c in cases),
        total_recorded_failed_resource_attempt_seconds=sum(c['costs']['failed_resource_attempt_seconds'] for c in cases),
        maximum_recorded_process_rss_mib=max(c['costs']['maximum_recorded_process_rss_mib'] or 0 for c in cases),
        window_names=list(WINDOWS),detectors=DETECTORS,absolute_floor=1e-18,
        comparisons=comparisons_from_cases(cases,raw),
        limitations='Timings use actual completed matched-source records. Changed retained support, product representation or depth is not a matched-target speedup. Preparatory and failed work is retained separately; parallel elapsed time is not inferred from summed worker times. All target acceptance remains controlled by the authoritative full acceptance calculation.')
    stream=a.output/'streaming_verification.json'
    if stream.exists():costs['matched_memory_implementation_validation']=inputs.read(stream)
    figs=figures(a.output,cases,raw,inputs) if not a.no_figures else []
    # Reuse the already independently checked detector-intersection evidence.
    panelanalysis=a.output/'sampling_audit/electron_panel_analysis/analysis.json'
    if panelanalysis.exists():
        d=inputs.read(panelanalysis);costs['verified_added_detector_weight_max']=max(float(np.max(row['components'][component]['added_fine_thermal_weight'])) for c in d['comparisons'] for row in c['rows'] for component in ['direct','sequential','total'])
        costs['added_beam_scope']='Recorded N17..N49 selected-panel comparisons: added output-beam detector intensity is exactly zero; common-beam amplitudes change. This does not exclude intermediate-beam support errors.'
    write(a.output/'case_inventory.json',inv);write(a.output/'cost_vs_accuracy.json',costs);write(a.output/'reproducible_commands.json',build_commands(cases,a.output))
    write(a.output/'reporting_input_manifest.json',dict(files=inputs.files,source_sha256=sha(__file__),source_archives_checked_by_actual_hash=True))
    lines=['# Measured numerical-case inventory','',
        f"{inv['numerical_evaluations']} committed numerical evaluations use {len(qids)} unique physical q/branch pairs ({len(sids)} source-file identities) across {len(cases)} cases. These are repeated diagnostic refinements and Born/control pilots; new independent BZ production=0. Every completed evaluation preserves all four direct and sequential complex sign amplitudes and the coherent total.",'',
        '| Case | Completed / declared | N / M | Depth / RK | Max kinetic full-step phase | Pair wall seconds | Peak RSS MiB |','|---|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        k=c['contract'];phase=max([r['kinetic_phase']['maximum_full_step_phase_rad'] for r in c['rows']],default=None)
        phase_label='not evaluated' if phase is None else f'{phase:.5g}'
        rss=c['costs']['maximum_recorded_process_rss_mib'];rss_label='not recorded' if rss is None else f'{rss:.1f}'
        lines.append(f"| {c['case']} | {c['completed_numerical_evaluations']} / {c['declared_panel_pairs']} | {k['retained']} / {k['product']} | {k['steps']} / {k['local_substeps']} | {phase_label} | {c['costs']['summed_committed_pair_wall_seconds']:.3f} | {rss_label} |")
    lines+=['',f"Summed committed pair work={costs['total_committed_pair_wall_seconds']:.3f}s; recorded initial preparation={costs['total_recorded_initial_preparation_seconds']:.3f}s; recorded resource-failed attempt time={costs['total_recorded_failed_resource_attempt_seconds']:.3f}s. These overlapping categories are not combined into an elapsed-time claim.",'',
        'Guard1–4 differences and exact source-archive matches are recorded per case. Missing creation-time hashes remain explicitly missing. Partial and zero-output resource failures remain in the inventory. Kinetic phases use physical reciprocal vectors, Bloch word shifts and actual maximum propagation intervals; changing N at fixed cell changes cutoff, not reciprocal spacing.','',
        '[Case inventory](case_inventory.json) · [Measured cost and sensitivity](cost_vs_accuracy.json) · [Reconstructed commands](reproducible_commands.json) · [Input hashes](reporting_input_manifest.json)','']
    (a.output/'reporting_digest.md').write_text('\n'.join(lines))
    write(a.output/'reporting_generation.json',dict(source_sha256=sha(__file__),seconds=time.monotonic()-started,figures=[dict(path=str(q),sha256=sha(q)) for q in figs],counts={k:inv[k] for k in ['numerical_evaluations','unique_physical_q_branch_sources','cases','new_independent_bz_production_pairs']},active_case_snapshot_only=True))
    print(json.dumps(clean(dict(counts=inv['numerical_evaluations'],unique_q_pairs=len(qids),cases=len(cases),missing_source_coverage=inv['source_coverage_missing'],seconds=time.monotonic()-started))),flush=True)

if __name__=='__main__':main()
