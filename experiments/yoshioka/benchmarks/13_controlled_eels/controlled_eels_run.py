"""Resume bounded controlled-comparison stages with explicit progress checks."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from .controlled_eels import DEFAULT, JOBS, physical_guard, validate_covariances, check_source_contract
from .workflow_io import ROOT, json_write


def progress(output, job, stage):
    config = json.loads((output/'configs'/f'{job}.json').read_text()); folder = ROOT/config['output']
    if stage == 'born':
        paths = [folder/'born'/f'T{t}_mesh{m}'/'state.json' for t in config['temperatures_k'] for m in config['born_meshes']]
        total = sum(m**3 for t in config['temperatures_k'] for m in config['born_meshes'])
    elif stage in ('source', 'refine', 'covtest'):
        cases = config['source_cases'] if stage == 'source' else config['refinement_cases'] if stage == 'refine' else [config['covariance_case']]
        paths = [folder/'source'/c['name']/'state.json' for c in cases]; total = sum(2**c['power'] for c in cases)
    else: return None
    states = [json.loads(p.read_text()) if p.exists() else {} for p in paths]
    return {'completed_q': sum(s.get('completed_q', 0) for s in states), 'total_q': total,
        'complete': all(s.get('status') == 'COMPLETE' for s in states)}


def verify_complete(config, folder, stage):
    """Audit finished physical contracts without mutating archived histories."""
    from .approximate_eels import run_born
    from .workflow_io import ResourceLimits
    validate_covariances(config, folder)
    if stage == 'born':
        run_born(config, folder, ResourceLimits(600, 2))
        for temperature in config['temperatures_k']:
            for mesh in config['born_meshes']:
                state = json.loads((folder/'born'/f'T{temperature}_mesh{mesh}'/'state.json').read_text())
                if state['completed_q'] != mesh**3: raise ValueError('Incomplete Born count marked complete')
    else:
        cases = config['source_cases'] if stage == 'source' else config['refinement_cases'] if stage == 'refine' else [config['covariance_case']]
        if stage == 'covtest': config = {**config, 'selected_covariance_mesh': 64}
        for c in cases:
            if check_source_contract(config, c, folder) != 2**c['power']: raise ValueError('Incomplete source count marked complete')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=DEFAULT); p.add_argument('--job', required=True, choices=JOBS)
    p.add_argument('--stages', default='covariance,pilot,born,gamma,source,refine,covtest')
    p.add_argument('--attempts', type=int, default=6)
    args = p.parse_args(); output = ROOT/args.output
    path = output/f'pipeline_{args.job}.json'; records = json.loads(path.read_text()) if path.exists() else []
    for stage in args.stages.split(','):
        for attempt in range(args.attempts):
            before = progress(output, args.job, stage)
            if before is not None and before['complete']:
                physical_guard(output)
                config = json.loads((output/'configs'/f'{args.job}.json').read_text())
                verify_complete(config, ROOT/config['output'], stage)
                print('Complete prefixes audited without recomputation:', args.job, stage, before['completed_q'], flush=True)
                break
            # Even complete stages run once to audit their physical contracts;
            # they return without recomputing a completed source or Born prefix.
            index = len(records)+1; log = output/f'pipeline_{args.job}_{index}_{stage}.log'
            start = time.monotonic()
            command = [sys.executable, '-m', 'benchmarks.controlled_eels', '--output', str(output.relative_to(ROOT)), '--job', args.job, '--stage', stage]
            print('Starting', args.job, stage, before, flush=True)
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            after = progress(output, args.job, stage)
            record = {'stage': stage, 'command': command, 'returncode': result.returncode, 'before': before, 'after': after,
                'seconds': time.monotonic()-start, 'log': str(log.relative_to(output))}
            records.append(record); json_write(path, records)
            print('Finished', args.job, stage, after, 'seconds', round(record['seconds'], 2), flush=True)
            if result.returncode: raise RuntimeError('Stage failed: '+str(log))
            if after is None or after['complete']: break
            if before is not None and after['completed_q'] <= before['completed_q']:
                raise RuntimeError('No committed progress; inspect the stage log')
        else: raise RuntimeError('Bounded attempt count exhausted; prefix is resumable')


if __name__ == '__main__': main()
