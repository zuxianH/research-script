"""Prepare new-directory reproduction configurations without changing history."""
import argparse
import json
from pathlib import Path
import shlex
from .workflow_io import ROOT, json_write, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True, help='A new path beneath the project root')
    args = p.parse_args(); directory = (ROOT/args.directory).resolve()
    relative = directory.relative_to(ROOT)
    directory.mkdir(parents=True, exist_ok=False)
    reference = ROOT/'benchmark_results/approximate_eels_20261102_v1'
    # Copy only the declared methods/criteria and commands, never completed
    # numerical spectra, q arrays, source fields, or checkpoints.
    for name in ('methods.md', 'criteria.json', 'objective.txt', 'implementation_diagnostics.md'):
        (directory/name).write_bytes((reference/name).read_bytes())
    resume = (reference/'reproduce_and_resume.sh').read_text()
    configurations = {}
    for name in ('approximate_eels', 'approximate_eels_production', 'approximate_eels_refinement',
                 'approximate_eels_confirmation', 'approximate_eels_covariance', 'approximate_eels_ehm'):
        source = ROOT/'benchmarks/configs'/f'{name}_20261102_v1.json'
        config = json.loads(source.read_text()); config['output'] = str(relative/('ehm' if name.endswith('_ehm') else ''))
        if 'primitive_force_constants_by_temperature' in config:
            config['primitive_force_constants_by_temperature'] = {
                key: str(relative/'ehm'/Path(value).name) for key, value in config['primitive_force_constants_by_temperature'].items()}
        target = directory/source.name; json_write(target, config); configurations[name] = str(target.relative_to(ROOT))
        resume = resume.replace(str(source.relative_to(ROOT)), str(target.relative_to(ROOT)))
    resume = resume.replace(str(reference.relative_to(ROOT)), str(relative))
    (directory/'reproduce_and_resume.sh').write_text(resume)
    q = lambda name: shlex.quote(configurations[name])
    json_write(directory/'extended_cache_guard.json', {'source_sha256': {
        str(path.relative_to(ROOT)): sha256(path) for path in (ROOT/'benchmarks').glob('approximate_eels*.py')
        if path.name not in ('approximate_eels_report.py', 'approximate_eels_finalize.py')}, 'scope': 'prospective cold-run calculation-source guard'})
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail', f'cd {shlex.quote(str(ROOT))}',
        'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',
        f'.venv/bin/python -m benchmarks.approximate_eels_finalize --config {q("approximate_eels_production")} --preserve before',
        f'.venv/bin/python -m benchmarks.guarded_run benchmarks.approximate_eels --config {q("approximate_eels")} --audit-only',
        f'.venv/bin/python -m benchmarks.approximate_eels --config {q("approximate_eels")} --stage covariance',
        f'.venv/bin/python -m benchmarks.approximate_eels --config {q("approximate_eels")} --stage pilot',
        '# Each Born invocation is bounded. Repeat this command until all requested Born meshes are COMPLETE.',
        f'.venv/bin/python -m benchmarks.approximate_eels --config {q("approximate_eels")} --stage born',
        f'.venv/bin/python -m benchmarks.approximate_eels --config {q("approximate_eels")} --stage born',
        f'.venv/bin/python -m benchmarks.approximate_eels_ehm --config {q("approximate_eels_ehm")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_gamma_patch --config {q("approximate_eels")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_gamma_patch --config {q("approximate_eels")} --mesh 48',
        f'.venv/bin/python -m benchmarks.approximate_eels_gamma_patch --config {q("approximate_eels_ehm")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_resume --config {q("approximate_eels_production")} --attempts 4',
        f'.venv/bin/python -m benchmarks.approximate_eels_resume --config {q("approximate_eels_refinement")} --attempts 2',
        f'.venv/bin/python -m benchmarks.approximate_eels_resume --config {q("approximate_eels_confirmation")} --attempts 2',
        f'.venv/bin/python -m benchmarks.approximate_eels --config {q("approximate_eels_covariance")} --stage covariance',
        f'.venv/bin/python -m benchmarks.approximate_eels_resume --config {q("approximate_eels_covariance")} --attempts 2',
        f'.venv/bin/python -m benchmarks.approximate_eels_cost --config {q("approximate_eels_production")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_validation --config {q("approximate_eels")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_repeated --config {q("approximate_eels")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_analysis --config {q("approximate_eels_production")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_compare --config {q("approximate_eels_production")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_finalize --config {q("approximate_eels_production")} --regression',
        f'.venv/bin/python -m benchmarks.approximate_eels_finalize --config {q("approximate_eels_production")} --preserve after',
        f'.venv/bin/python -m benchmarks.approximate_eels_report --config {q("approximate_eels_production")}',
        f'.venv/bin/python -m benchmarks.approximate_eels_finalize --config {q("approximate_eels_production")}']
    (directory/'calculate.sh').write_text('\n'.join(commands)+'\n')
    json_write(directory/'setup.json', {'status': 'COLD_CONFIGURATIONS_PREPARED_ONLY', 'configurations': configurations,
        'physics_inputs': 'Existing immutable force-constant and corrected-MD-fit inputs; no completed EELS arrays copied.',
        'next': 'Run calculate.sh; inspect bounded checkpoints and repeat an incomplete stage. Copy the preservation/report commands from the main reproduction guide.'})
    print(directory/'calculate.sh')


if __name__ == '__main__': main()
