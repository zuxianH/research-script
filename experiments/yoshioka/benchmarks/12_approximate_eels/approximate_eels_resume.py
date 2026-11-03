"""Bounded invocations with committed-q progress checks, never blind restarts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from .workflow_io import ROOT, json_write, sha256


def progress(config):
    output = ROOT/config['output']; completed = total = 0
    from .approximate_eels import verify_input_baseline, selected_covariance, primitive_path
    verify_input_baseline(output)
    for case in config['source_cases']:
        total += 2**case['power']
        path = output/'source'/case['name']/'state.json'
        if path.exists():
            state = json.loads(path.read_text()); contract = state['contract']
            expected = contract['settings'] == case and contract['repeats'] == config['selected_repeats']
            expected &= contract['padding_angstrom'] == config['padding_angstrom']
            expected &= contract['algorithm'] == sha256(ROOT/'yoshioka/bulk_thermal_source.py')
            expected &= contract['force'] == sha256(primitive_path(config, case['temperature_k']))
            expected &= contract['covariance'] == selected_covariance(config, output, case['temperature_k']).tolist()
            if not expected: raise ValueError('Completed-prefix physics changed; do not reuse this named case')
            completed += state['completed_q']
    return completed, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='benchmarks/configs/approximate_eels_production_20261102_v1.json')
    p.add_argument('--attempts', type=int, default=4)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text()); output = ROOT/config['output']
    guard = output/'cache_guard.json'
    if guard.exists():
        from .guarded_run import profile, validate_guard
        validate_guard(guard, profile('benchmarks.approximate_eels'))
    additional = output/'extended_cache_guard.json'
    if additional.exists():
        for name, digest in json.loads(additional.read_text())['source_sha256'].items():
            if sha256(ROOT/name) != digest:
                raise ValueError('Additional scientific source changed; preserve this output and use a new results directory')
    record_path = output/(Path(args.config).stem+'_invocations.json')
    records = json.loads(record_path.read_text()) if record_path.exists() else []
    for attempt in range(args.attempts):
        before, total = progress(config)
        if before == total:
            print('All requested source q points complete:', total, flush=True); return
        index = len(records)+1; log_path = output/f'{Path(args.config).stem}_invocation_{index}.log'
        print('Starting bounded invocation', index, 'committed', before, '/', total, flush=True)
        start = time.monotonic()
        with log_path.open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'benchmarks.approximate_eels', '--config', args.config, '--stage', 'source'],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        after, total = progress(config)
        records.append({'index': index, 'before_q': before, 'after_q': after, 'total_q': total,
                        'returncode': result.returncode, 'seconds': time.monotonic()-start, 'log': str(log_path)})
        json_write(record_path, records)
        print('Finished bounded invocation', index, 'committed', after, '/', total, flush=True)
        if result.returncode or after <= before:
            raise RuntimeError('No safe automatic continuation: inspect the recorded error or no-progress condition')
        if after == total:
            print('All requested source q points complete:', total, flush=True); return
    print('Bounded attempt count reached; committed q points remain resumable.', flush=True)


if __name__ == '__main__': main()
