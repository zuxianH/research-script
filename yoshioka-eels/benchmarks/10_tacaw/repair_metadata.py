"""Scoped TACAW metadata discovery and independently enabled comparisons."""
import json
from pathlib import Path

import numpy as np
from scipy.constants import h,elementary_charge

from yoshioka.comparison_metadata import comparison_availability,REQUIREMENTS
from .workflow_io import ROOT,json_write,sha256


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2/B10';output.mkdir(parents=True,exist_ok=True)
    project=Path('/home/zuxian/Documents/PhD/HiPhive_TRPMD')
    sources={}
    for p in sorted(project.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.ipynb','.json','.log','.sh','.yaml','.yml','.npz'):
            sources[str(p)]={'sha256':sha256(p),'bytes':p.stat().st_size}
    user_path=ROOT/'benchmarks/configs/tacaw_user_metadata_20261102.json'
    user=json.loads(user_path.read_text())
    helper=project/'multiphonon_analysis.py'
    job=project/'simple_peak_results_300to900K/job.json'
    j=json.loads(job.read_text());settings=j['settings']
    sidecars=[]
    for T,b in ((300,9),(1000,4)):
        path=next((project/'simple_peak_results_300to900K').glob(f'Si_5x7x52_{T}K_beads{b}_Tersoff_*.npz'))
        with np.load(path,allow_pickle=False) as a:
            axis=a['energy'];step=float(np.mean(np.diff(axis)))
            derived=h/(600*settings['timestep_fs']*settings['stride']*1e-15)/(1e-3*elementary_charge)
            sidecars.append({'temperature_k':T,'path':str(path),'sha256':sha256(path),'axis_length':len(axis),
                'analysis_axis_min_max_mev':[float(axis[0]),float(axis[-1])],'step_mev':step,
                'difference_from_helper_formula_mev':abs(step-derived),
                'generation_axis_verification':False,'prior_analysis_smoothing_sigma_mev':settings['sigma_mev'],
                'background_curves_present':True,'used_for_primary_raw_array_comparison':False})
    rows={}
    def add(key,value,units,status,source,note=''):
        rows[key]={'value':value,'units':units,'status':status,'source':source,'note':note,
                   'comparisons':[name for name,keys in REQUIREMENTS.items() if key in keys]}
    add('array_identity','two supplied arrays exactly match HiPhive_TRPMD Tersoff arrays','SHA-256','VERIFIED',
        'benchmark_results/tacaw_20261102_v1/results.json')
    add('temperature',[300,1000],'K','USER_SUPPLIED','user and filenames','actual trajectory mean temperatures not independently verified')
    add('beads',[9,4],'count','DERIVED_FROM_VERIFIED','hash-matched counterpart filenames','labels establish claimed bead counts, not the dynamics estimator')
    add('electron_energy',60,'keV','USER_SUPPLIED',str(user_path))
    add('slice_count',208,'count','USER_SUPPLIED',str(user_path),'slice thickness and total thickness unknown')
    add('incident_normalization','integral over px,py,E equals one','dimensionless','USER_SUPPLIED',str(user_path),'coordinate measures and elastic handling remain unspecified')
    add('quantum_correction','Kubo already applied','none','USER_SUPPLIED',str(user_path),'no second correction applied')
    add('force_model_hint',user['force_model']['value'],'none','USER_SUPPLIED',str(user_path),'tentative; no parameter-hash match established')
    add('local_parameter_hash',sha256(ROOT/'Si.tersoff.mod'),'SHA-256','VERIFIED',str(ROOT/'Si.tersoff.mod'),'local file identity only, not verified TACAW force model')
    add('analysis_energy_axis',{'timestep_fs':settings['timestep_fs'],'stride':settings['stride'],'cadence_fs':settings['timestep_fs']*settings['stride'],
        'length':600,'FFT':'centered','zero_index':300,'step_mev':sidecars[0]['step_mev']},'meV','ANALYSIS_ASSUMPTION',str(job),
        'sidecar axis formula independently reproduced; generation cadence, padding and sign still unverified')
    add('analysis_detector',{'radius':20,'offset_px':0,'offset_py':0},'array pixels','ANALYSIS_ASSUMPTION',str(job),'no physical angle calibration')
    add('analysis_duration_if_unpadded',4.5,'ps','ANALYSIS_ASSUMPTION',str(helper),'N*cadence; not established physical segment duration if zero padded')
    add('analysis_axis_order','energy,y,x','none','ANALYSIS_ASSUMPTION',str(helper),'matches existing helper, not generation log')
    for key,units,note in (
        ('energy_axis','meV','generation cadence, physical duration, padding, zero and FFT ordering'),
        ('energy_sign','none','energy-loss versus gain sign from generating transform'),
        ('momentum_axes','m^-1 or angle','coordinates, origin, shifts, angular versus cycles convention'),
        ('density_or_bin_weight','density or bin weight','saved-array measure before downstream normalization'),
        ('elastic_handling','none','coherent subtraction, zero-loss inclusion or mask'),
        ('specimen_geometry','m and orientation','actual oriented cell, repeats, total thickness, surfaces and temperature geometry'),
        ('electron_potential','none','electron potential parametrization, projection, filters and grid'),
        ('force_parameter_hash','SHA-256','exact generating pair style and parameter-file hash'),
        ('native_window','ps and window','un-padded segment length, window, overlap, averaging and existing broadening'),
        ('dynamics','fs and ensemble','MD timestep, sampling cadence, thermostat, equilibration, replicas and estimator'),
        ('uncertainty','none','independent blocks or raw trajectory statistics'),
        ('generation_cost','s and bytes','trustworthy TACAW generation time and peak memory'),
        ('cost_hardware','none','generation hardware/software/threads'),
        ('matched_accuracy_scope','none','cost comparison requires common physical inputs and accepted accuracy')):
        add(key,None,units,'MISSING','not recovered in scoped project files',note)
    availability=comparison_availability(rows)
    results={'status':'METADATA_INCOMPLETE','metadata':rows,'availability':availability,'sidecar_analysis_checks':sidecars,
        'discovery_scope':[str(project),str(ROOT)],'scoped_analysis_inventory':sources,'user_metadata_sha256':sha256(user_path),
        'conditional_comparison_artifact':'benchmark_results/tacaw_material_20261102_v1/results.json',
        'correction_applied_this_run':False,'generation_cost_inferred':False,
        'limitation':'sidecar axes verify downstream analysis convention only; no generation provenance was recovered'}
    json_write(output/'results.json',results)
    table=['# TACAW metadata and independent comparison availability','','| Field | Value | Units | Status | Source | Depends on |','|---|---|---|---|---|---|']
    for k,row in rows.items():
        table.append('| '+ ' | '.join((k,str(row['value']).replace('|','/'),row['units'],row['status'],row['source'],', '.join(row['comparisons'])))+' |')
    table+=['','| Comparison | Availability | Remaining fields |','|---|---|---|']
    for k,v in availability.items():table.append(f"| {k} | {v['status']} | {', '.join(v['missing'])} |")
    table+=['','The existing conditional shape figures remain available. They use disclosed analysis axes and unmatched pixel/angular detectors. Runtime records are not required for these figures. No independently calibrated absolute or matched-model comparison is claimed. The sidecar spectra contain prior smoothing and background models; they are provenance evidence, not substituted raw data.']
    (output/'metadata_table.md').write_text('\n'.join(table)+'\n')
    missing=['# Remaining inputs only','','For calibrated spectra and matched physical comparisons:','']
    for key in ('energy_axis','energy_sign','momentum_axes','density_or_bin_weight','elastic_handling','specimen_geometry','electron_potential','force_parameter_hash','native_window','dynamics','uncertainty'):
        missing.append('- '+key+': '+rows[key]['note']+'.')
    missing+=['','For a separate cost comparison only:','- TACAW generation wall time/peak memory, hardware/software/threads, and a matched accuracy target. Missing cost logs do not block conditional spectral figures.','','Already recorded: 60 keV, 208 slices, unit px-py-E integral, Kubo already applied. These values are not requested again.']
    (output/'missing_inputs.md').write_text('\n'.join(missing)+'\n')


if __name__=='__main__':main()
