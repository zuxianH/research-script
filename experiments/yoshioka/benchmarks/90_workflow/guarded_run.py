"""Reproduction entry point with transitive source/constants cache guards.

The per-run inner checkpoints additionally validate arrays, physics and prefixes.
Changing a budget alone leaves this guard unchanged. Existing data are adopted
only by an explicit recorded --adopt-current-provenance action; this is not a
claim that a missing historical guard existed at calculation time.
"""

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

from scipy import constants

from .workflow_io import ROOT, fingerprint, json_write, sha256


def profile(module):
    paths = sorted((ROOT/'yoshioka').glob('*.py'))
    # All calculation modules are covered; reporting-only files are excluded so
    # producing figures does not invalidate a physically identical calculation.
    names = ('born_quadrature', 'channel_reference', 'eels_observables', 'abtem_reference',
             'reference_solutions', 'elastic_observables', 'elastic_refinement', 'workflow_io')
    paths += [ROOT/'benchmarks'/f'{name}.py' for name in names]
    paths += [ROOT/Path(*module.split('.')).with_suffix('.py')]
    values = {name: float(getattr(constants, name)) for name in
              ('hbar', 'h', 'elementary_charge', 'electron_mass', 'atomic_mass', 'Boltzmann', 'speed_of_light')}
    source = Path(inspect.getfile(constants))
    # The exported CODATA dictionary is hashed, not just a version label.
    actual = {key: list(value) for key, value in constants.physical_constants.items()}
    from yoshioka.electron_phonon import LobatoSilicon
    potential = LobatoSilicon.from_abtem_table()
    potential_values = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in potential.__dict__.items()}
    return {'format': 1, 'module': module, 'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        'constants_SI': values, 'constants_source': 'SciPy constants, CODATA 2022',
        'constants_module_sha256': sha256(source), 'physical_constants_dictionary_sha256': fingerprint(actual),
        'Lobato_Si_actual_table': potential_values,
        'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'ase', 'abtem', 'hiphive', 'lammps')},
        'precision': 'float64/complex128',
        'thread_environment': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'thread_note': 'Use the recorded single-thread environment to reproduce eigenvectors in degenerate subspaces and their exact cache hashes.'}


def validate_guard(path, current, adopt=False):
    digest = fingerprint(current)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['digest'] != digest:
            raise ValueError('Cache provenance changed: use a new versioned output directory; prior data are preserved.')
        return previous
    if not adopt and path.parent.exists() and any(path.parent.glob('cache/*/state.json')):
        raise ValueError('Existing unguarded cache needs an explicit provenance audit/adoption before resume.')
    record = {'digest': digest, 'profile': current, 'adopted_existing_cache': bool(adopt),
        'adoption_scope': 'current source/constants audit; original inner cache contracts and creation evidence remain authoritative'}
    json_write(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module')
    parser.add_argument('--config', required=True)
    parser.add_argument('--adopt-current-provenance', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args, rest = parser.parse_known_args()
    config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    current = profile(args.module)
    validate_guard(output/'cache_guard.json', current, args.adopt_current_provenance)
    snapshots = output/'guarded_source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for name, digest in current['source_sha256'].items():
        target = snapshots/name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        if sha256(target) != digest:
            raise ValueError('Source snapshot mismatch; preserve old run and use new output')
    if not args.audit_only:
        sys.argv = [args.module, '--config', args.config]+rest
        importlib.import_module(args.module).main()


if __name__ == '__main__':
    main()
