"""Bounded/resumable continuous-BZ leading two-phonon silicon EELS."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import constants

from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.two_phonon import pair_nodes, propagate_bulk_pair
from .approximate_eels import prepare_background, geometry, PRIMITIVE, ORIENTED, selected_covariance
from .workflow_io import ROOT, ResourceLimits, environment, fingerprint, json_write, sha256


DEFAULT = ROOT/'benchmark_results/two_phonon_eels_20261102_v1'
PREVIOUS = ROOT/'benchmark_results/approximate_eels_20261102_v1'
CONTROLLED = ROOT/'benchmark_results/controlled_eels_comparison_20261102_v1'
KERNELS = ('yoshioka/two_phonon.py', 'yoshioka/thermal_dyson_words.py',
    'yoshioka/bulk_thermal_source.py', 'yoshioka/fourier_depth_fields.py',
    'yoshioka/electron_phonon.py', 'yoshioka/dynamical_source.py',
    'yoshioka/free_propagation.py', 'yoshioka/elastic_propagation.py',
    'yoshioka/lattice_dynamics.py', 'benchmarks/approximate_eels.py')


def setup(output):
    if not (output/'protected_before.json').exists():
        raise ValueError('Hash pre-existing files before setup')
    records = []
    for mesh in (32, 64):
        source = PREVIOUS/'covariance'/f'T300_mesh{mesh}.json'
        before = json.loads((output/'protected_before.json').read_text())['files']
        if sha256(source) != before[str(source.relative_to(ROOT))]['sha256']:
            raise ValueError('historical covariance changed')
        row = json.loads(source.read_text())
        if row['force_constants_sha256'] != sha256(PRIMITIVE):
            raise ValueError('covariance force model mismatch')
        target = output/'covariance'/source.name
        target.parent.mkdir(exist_ok=True)
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError('copied covariance changed')
        if not target.exists(): shutil.copy2(source, target)
        records.append({'source': str(source.relative_to(ROOT)), 'target': str(target.relative_to(output)),
                        'sha256': sha256(source)})
    config = dict(selected_repeats=4, padding_angstrom=12, selected_covariance_mesh=32,
        temperature_k=300, energy_ev=60000, runtime_limit_seconds=600, memory_limit_gib=2,
        primary_grid=49, primary_steps=385, batches=8, seed_start=729001,
        gaussian_fwhm_mev=2, spectral_axis_mev=[-180, 180, .05],
        uniform_fraction=.5, importance_radius=.35)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text()) != config:
        raise ValueError('immutable configuration changed')
    json_write(output/'config.json', config)
    criteria = dict(primary_detector='full_0_20_mrad', numerical_shape_L1_max=.1,
        identifiable_feature_shift_mev_max=1., statistical_95_halfwidth_relative_max=.1,
        weak_integrated_weight_absolute_floor=1e-18,
        pair_shape_window_mev=[5, 160], historical_shape_window_mev=[5, 90],
        historical_feature_windows_mev={'acoustic':[10,40], 'middle':[40,60], 'optical':[60,85]},
        extra_feature_windows_mev={'sum_90_125':[90,125], 'sum_125_160':[125,160]},
        raw_windows_mev={'acoustic_0_40':[0,40], 'middle_40_65':[40,65],
                         'optical_65_90':[65,90], 'sum_90_125':[90,125], 'sum_125_160':[125,160]},
        limitations='Statistical intervals exclude deterministic discretization and linear-harmonic model errors. Undefined peaks remain unresolved. No full9B PASS.',
        sample_count_selection='Pilot at8 independent batches x8 pairs; freeze measured-cost production/refinement sequence before extension.',
        pilot_has_influenced_count_selection=True)
    if not (output/'criteria.json').exists(): json_write(output/'criteria.json', criteria)
    json_write(output/'reuse_provenance.json', {'covariances': records,
        'one_phonon_baseline': {'path': str((CONTROLLED/'fixed_r4_T300_spectra.npz').relative_to(ROOT)),
            'sha256': sha256(CONTROLLED/'fixed_r4_T300_spectra.npz')},
        'constants': json.loads((PREVIOUS/'constants_and_physics.json').read_text()),
        'scope': 'Historical inputs are read-only; new backgrounds have independent caches.'})
    json_write(output/'environment.json', environment())
    json_write(output/'initial_diagnosis.json', {
        'existing_solver': 'thermal_dyson_words exact local0/1/2 triangular action; independent dense controls exist',
        'remaining_gap': 'Existing material pilot has only primitive-Gamma optical triplet, not full BZ.',
        'representation': '21 complex electron waves for two selected modes, streamed source depths; no dense mode-pair Hamiltonian.',
        'integration': '6D BZ plus two uniform branch labels, pair importance product and1/2 unordered-final-state factor.',
        'statistics': 'Independent scrambled Sobol batches; exact per-sampled-pair intensity, no noisy amplitude square.',
        'model': 'Linear coupling through two ordered insertions; nonlinear multiphonon vertices and higher orders omitted.',
        'same_mode': 'Finite discrete identical-mode factors tested independently. Diagonal pair weights vanish in continuum double integral, while single-integral return paths survive.',
        'cheapest_test': 'Exact reduced discrete-bath trace and a64-pair production-geometry cost/variance pilot.'})


def contract(config, settings, proposal, output):
    potential = LobatoSilicon.from_abtem_table()
    return dict(settings=settings, proposal=proposal, temperature_k=config['temperature_k'],
        selected_repeats=config['selected_repeats'], padding_angstrom=config['padding_angstrom'],
        covariance=json.loads((output/'covariance'/f"T300_mesh{settings.get('covariance_mesh',32)}.json").read_text()),
        primitive_sha256=sha256(PRIMITIVE), oriented_sha256=sha256(ORIENTED),
        kernel_sha256={name:sha256(ROOT/name) for name in KERNELS},
        potential_parameters={'a':potential.a_angstrom.tolist(), 'b':potential.b_angstrom2.tolist()},
        hbar=constants.hbar, elementary_charge=constants.elementary_charge,
        energy_ev=config['energy_ev'], numpy=np.__version__,
        conventions='unit source BZ weights; branch/proposal factor in probability once;1/2 ordered pair; marginal average1/2; coherent exchange')


def prepare_importance(output):
    folder=output/'importance';folder.mkdir(exist_ok=True)
    source=PREVIOUS/'born/T300_mesh64/weights.npy'
    before=json.loads((output/'protected_before.json').read_text())['files']
    if before[str(source.relative_to(ROOT))]['sha256']!=sha256(source):
        raise ValueError('historical Born input changed')
    w=np.load(source,mmap_mode='r')
    mass=np.array(w[:,:6,0]+w[:,6:,0]);normalization=float(mass.sum());mass/=normalization
    target=folder/'mass.npy'
    if target.exists():
        if not np.array_equal(np.load(target),mass):raise ValueError('importance mass changed')
    else:np.save(target,mass)
    record={'mesh':64,'mass_sha256':sha256(target),'source':str(source),'source_sha256':sha256(source),
        'source_joint_weight_before_proposal_normalization':normalization,
        'definition':'Joint q-cell/branch emission+absorption Born weight in0-20mrad. Normalized only to define a sampling PDF; propagated outputs are not renormalized.',
        'continuous_cells':'Nearest64^3 nodal cell, uniform continuous position inside cell; dynamical matrix recomputed at actual q.',
        'mixture':{'uniform_joint':.2,'radial_joint':.4,'Born_joint':.4},'radial_radius':.35}
    path=folder/'proposal.json'
    if path.exists():
        if json.loads(path.read_text())!=record:raise ValueError('importance provenance changed')
    else:json_write(path,record)


def require_contract(state, expected):
    if state['contract'] != expected:
        raise ValueError('Incompatible pair cache: physical model, covariance, propagator or sampling changed. Use a new job directory.')


def ready_background(config,settings,output,limits):
    """Read completed compatible caches without updating their metadata.

    This permits independent batch processes to share immutable elastic fields.
    A new numerical geometry is prepared sequentially before parallel work.
    """
    from yoshioka.free_propagation import TransverseGrid
    model=RealSpaceForceConstants.load(ORIENTED)
    covariance=selected_covariance(config,output,300)
    grid=TransverseGrid(model.cell_m[0,0],model.cell_m[1,1],settings['grid_unit'],settings['grid_unit'])
    expected={'grid':grid.__dict__,'steps':settings['slices'],'repeats':config['selected_repeats'],'elastic':True,
        'temperature_k':300,'covariance_m2':covariance.tolist(),'padding_angstrom':config['padding_angstrom'],
        'force_sha256':sha256(ORIENTED),'field_sha256':sha256(ROOT/'yoshioka/fourier_depth_fields.py'),
        'incident_sha256':sha256(ROOT/'yoshioka/depth_fields.py')}
    folder=output/'background'/fingerprint(expected)[:20]
    if (folder/'state.json').exists():
        state=json.loads((folder/'state.json').read_text())
        if state['contract']!=expected:raise ValueError('elastic background contract mismatch')
        if state.get('background_complete') and state.get('incident_complete'):
            positions,reference,length=geometry(model,config['selected_repeats'],config['padding_angstrom'])
            return positions,reference,length,covariance,grid,np.load(folder/'background.npy',mmap_mode='r'),np.load(folder/'incident.npy',mmap_mode='r'),{**state,'latest_setup_seconds':0.,'immutable_reuse':True}
    return prepare_background(config,settings,output,limits)


def run(output, job='main', power=3, batches=8, grid=49, steps=385,
        radius=.35, uniform_fraction=.5, covariance_mesh=32, max_new=None, batch_start=0, proposal_kind='radial'):
    config = json.loads((output/'config.json').read_text())
    if config['temperature_k']!=300 or config['energy_ev']!=60000:
        raise ValueError('this controlled task requires300K/60keV')
    if json.loads((output/'environment.json').read_text())!=environment():
        raise ValueError('environment changed since cache creation')
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'])
    settings = dict(temperature_k=300, grid_unit=grid, slices=steps, covariance_mesh=covariance_mesh)
    proposal = dict(radius=radius, uniform_fraction=uniform_fraction, branch_probability=1/6,
                    seed_start=config['seed_start'], engine='10D scrambled Sobol')
    seed_start=config['seed_start']
    node_function=lambda power,seed:pair_nodes(power,seed,radius,uniform_fraction)
    if proposal_kind=='born':
        from yoshioka.phonon_pair_importance import BornPairProposal
        path=output/'importance/mass.npy'
        metadata=json.loads((output/'importance/proposal.json').read_text())
        if sha256(path)!=metadata['mass_sha256']:raise ValueError('importance table changed')
        sampler=BornPairProposal(np.load(path),metadata['mesh'])
        if radius!=.35 or uniform_fraction!=.5:
            raise ValueError('Born mixture uses its explicit immutable component weights')
        seed_start=829001
        proposal={**proposal,'kind':'Born table + radial + uniform joint q/branch mixture',
            'table':metadata,'sampler_sha256':sha256(ROOT/'yoshioka/phonon_pair_importance.py'),
            'seed_start':seed_start,'branch_probability':'actual joint q/branch density evaluated at every sample',
            'uniform_joint_fraction':.2,'radial_joint_fraction':.4,'Born_joint_fraction':.4}
        node_function=sampler.nodes
    elif proposal_kind!='radial':raise ValueError('unknown proposal')
    expected = contract(config, settings, proposal, output)
    root = output/'pairs'/job; root.mkdir(parents=True, exist_ok=True)
    batch_indices=range(batch_start,batch_start+batches)
    for seed_index in batch_indices:
        state_path = root/f'batch{seed_index:02d}'/'state.json'
        if state_path.exists(): require_contract(json.loads(state_path.read_text()), expected)
    # A finished resume is read-only: no field setup, no state timestamp changes.
    complete = True
    for seed_index in batch_indices:
        path = root/f'batch{seed_index:02d}'/'state.json'
        if not path.exists() or json.loads(path.read_text())['completed_pairs'] < 2**power:
            complete = False
    if complete:
        for seed_index in batch_indices:
            folder = root/f'batch{seed_index:02d}'
            state = json.loads((folder/'state.json').read_text())
            nodes, branch, weights = node_function(power, seed_start+seed_index)
            for i in range(2**power):
                path = folder/f'pair{i:06d}.npz'
                if sha256(path) != state['files'][path.name]: raise ValueError('committed pair changed')
                with np.load(path) as d:
                    if not np.array_equal(nodes[i],d['q_fractional']) or not np.array_equal(branch[i],d['branches']):
                        raise ValueError('nested sample sequence changed')
        print('COMPLETE; immutable resume verified', job, power, batches, flush=True)
        return
    # Verify every committed prefix before extending it, including its nodes.
    for seed_index in batch_indices:
        folder=root/f'batch{seed_index:02d}';state_path=folder/'state.json'
        if not state_path.exists():continue
        state=json.loads(state_path.read_text());nodes,branch,_=node_function(power,seed_start+seed_index)
        for i in range(min(state['completed_pairs'],2**power)):
            path=folder/f'pair{i:06d}.npz'
            if sha256(path)!=state['files'][path.name]:raise ValueError('committed pair changed before extension')
            with np.load(path) as d:
                if not np.array_equal(nodes[i],d['q_fractional']) or not np.array_equal(branch[i],d['branches']):
                    raise ValueError('nested sample prefix changed before extension')
    local_config = {**config, 'selected_covariance_mesh':covariance_mesh}
    setup_start=time.monotonic();cpu_start=time.process_time()
    positions, reference, length, covariance, transverse, background, incident, setup_state = ready_background(
        local_config, settings, output, limits)
    setup_seconds = time.monotonic()-setup_start
    model = RealSpaceForceConstants.load(PRIMITIVE)
    new = 0; status = 'COMPLETE'; started = time.monotonic()
    try:
        # Interleave batches so every bounded invocation improves independent uncertainty.
        for i in range(2**power):
            for seed_index in batch_indices:
                folder = root/f'batch{seed_index:02d}'; folder.mkdir(exist_ok=True)
                state_path = folder/'state.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {
                    'contract':expected, 'completed_pairs':0, 'cumulative_seconds':0., 'files':{}, 'status':'INCOMPLETE'}
                require_contract(state, expected)
                if i < state['completed_pairs']: continue
                if i != state['completed_pairs']: raise ValueError('noncontiguous pair checkpoint')
                if max_new is not None and new >= max_new: raise TimeoutError('bounded pair count reached')
                limits.check()
                nodes, branch, inverse = node_function(power, seed_start+seed_index)
                pair_start = time.monotonic()
                result = propagate_bulk_pair(model, positions, reference, covariance, transverse, length, steps,
                    nodes[i], branch[i], inverse[i], background, folder/'scratch', 300, limits,
                    save_amplitudes=(i==0 and seed_index==0))
                result.update(q_fractional=nodes[i], branches=branch[i], inverse_marginals=inverse[i])
                path = folder/f'pair{i:06d}.npz'
                temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, **result); temporary.replace(path)
                elapsed = time.monotonic()-pair_start
                state['files'][path.name] = sha256(path)
                state.update(completed_pairs=i+1, cumulative_seconds=state['cumulative_seconds']+elapsed,
                    status='COMPLETE' if i+1==2**power else 'INCOMPLETE', target_pairs_latest_request=2**power,
                    latest_pair_seconds=elapsed)
                json_write(state_path,state); new+=1
                print(job, seed_index, i+1, '/', 2**power, 'seconds',round(elapsed,3),flush=True)
    except TimeoutError as error:
        status = 'INCOMPLETE'; print(str(error),flush=True)
    records = output/'invocations'; records.mkdir(exist_ok=True)
    import os
    invocation=f'{time.time_ns()}_{os.getpid()}_{job}'
    json_write(records/f'{invocation}.json', dict(job=job, target_power=power, batches=batches,batch_start=batch_start,
        new_pairs=new,status=status,setup_seconds=setup_seconds, new_pair_loop_seconds=time.monotonic()-started,
        resources=limits.costs(), contract_fingerprint=fingerprint(expected),runner_sha256=sha256(__file__),
        spacings_m=[transverse.dx_m,transverse.dy_m,length/steps],thickness_nm=config['selected_repeats']*.38388827440614426,
        process_cpu_seconds_since_setup=time.process_time()-cpu_start))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    sub=p.add_subparsers(dest='stage',required=True)
    sub.add_parser('setup')
    sub.add_parser('importance-setup')
    r=sub.add_parser('run')
    r.add_argument('--job',default='main'); r.add_argument('--power',type=int,default=3)
    r.add_argument('--batches',type=int,default=8); r.add_argument('--grid',type=int,default=49)
    r.add_argument('--batch-start',type=int,default=0)
    r.add_argument('--steps',type=int,default=385); r.add_argument('--radius',type=float,default=.35)
    r.add_argument('--uniform-fraction',type=float,default=.5)
    r.add_argument('--covariance-mesh',type=int,default=32); r.add_argument('--max-new',type=int)
    r.add_argument('--proposal-kind',choices=('radial','born'),default='radial')
    args=vars(p.parse_args()); output=args.pop('output'); stage=args.pop('stage')
    if stage=='setup': setup(output)
    elif stage=='importance-setup':prepare_importance(output)
    else: run(output,**args)


if __name__=='__main__': main()
