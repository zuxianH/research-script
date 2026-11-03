"""Explicit compatible nested-Sobol prefix reuse for a justified source refinement."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from .controlled_eels import DEFAULT, physical_guard, check_source_contract
from .workflow_io import ROOT, json_write, sha256
from yoshioka.bulk_thermal_source import importance_nodes


def promote_prefix(config, case, folder, power):
    old_n = 2**case['power']; new_n = 2**power
    if power <= case['power']: raise ValueError('Refinement must increase q count')
    if check_source_contract(config, case, folder) != old_n: raise ValueError('Original prefix is incomplete')
    original = folder/'source'/case['name']; state = json.loads((original/'state.json').read_text())
    if state['status'] != 'COMPLETE': raise ValueError('Original prefix is not complete')
    new_case = {**case, 'name': case['name']+f'_q{new_n}', 'power': power}
    target = folder/'source'/new_case['name']
    if target.exists():
        check_source_contract(config, new_case, folder)
        return new_case
    q, inverse = importance_nodes(power, case['seed'])
    with np.load(original/'spectrum.npz') as d:
        np.testing.assert_array_equal(q[:old_n], d['q_fractional'])
        np.testing.assert_array_equal(inverse[:old_n], d['inverse_sampling_density'])
    target.mkdir(); source_hashes = {}
    for name in ('energies_mev', 'weights', 'born_weights'):
        path = original/(name+'.npy'); array = np.load(path)
        saved = np.zeros((new_n, *array.shape[1:]), dtype=array.dtype); saved[:old_n] = array
        np.save(target/(name+'.npy'), saved); source_hashes[name] = sha256(path)
    new_state = {'contract': copy.deepcopy(state['contract']), 'completed_q': old_n, 'total_q': new_n,
        'status': 'INCOMPLETE', 'cumulative_seconds': 0.,
        'prefix_import': {'source': str(original), 'q': old_n, 'source_array_sha256': source_hashes,
            'reused_computation_seconds': state['cumulative_seconds'],
            'normalization': 'Raw arrays are inverse-density-weighted integrands. No 1/N rescaling is applied during prefix import.'}}
    new_state['contract']['settings'] = new_case
    json_write(target/'state.json', new_state)
    return new_case


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default=DEFAULT)
    p.add_argument('--job', default='fitted_r8_T300'); p.add_argument('--power', type=int, default=7)
    args = p.parse_args(); output = ROOT/args.output; physical_guard(output)
    path = output/'configs'/f'{args.job}.json'; config = json.loads(path.read_text())
    if all(c['power'] == args.power for c in config['source_cases']):
        print('Existing refined sampling configuration preserved'); return
    archive = output/'source64_evidence'; archive.mkdir(exist_ok=True)
    for pth in output.glob('*'):
        if pth.is_file() and (pth.suffix == '.png' or pth.name in ('comparison.json', 'finest_numerical_checks.json', 'report.md', args.job+'_spectra.npz')):
            if not (archive/pth.name).exists(): shutil.copy2(pth, archive/pth.name)
    old_config = output/'configs'/f'{args.job}_q64.json'
    if old_config.exists(): raise ValueError('Preserve the original sampling config')
    shutil.copy2(path, old_config); folder = ROOT/config['output']
    refined = [promote_prefix(config, c, folder, args.power) for c in config['source_cases']]
    config['previous_source_cases'] = config['source_cases']; config['source_cases'] = refined
    config['sampling_refinement'] = {'previous_config': str(old_config.relative_to(ROOT)),
        'reason': 'Weak off-axis middle-feature dominant maximum changes by2.05meV at32→64q/batch; acoustic feature also becomes unidentifiable. Refine only the affected model, preserving all64q evidence.',
        'new_q_per_batch': 2**args.power, 'additional_q': sum(2**args.power-2**c['power'] for c in config['previous_source_cases']),
        'source_sha256': sha256(Path(__file__))}
    json_write(path, config); json_write(output/(args.job+'_sampling_refinement.json'), config['sampling_refinement'])
    print('Refined configuration:', args.job, 'q per batch:', 2**args.power, flush=True)


if __name__ == '__main__': main()
