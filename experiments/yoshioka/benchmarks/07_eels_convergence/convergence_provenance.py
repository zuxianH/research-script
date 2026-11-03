"""Keep numerical cache identities stable across reporting-only corrections.

The complete manifests remain archived and reported. Reuse is allowed only when
the actual scientific inputs, validated statuses, physical model, and software
environment match. Calculation-source hashes are checked separately by bulk_eels.
"""

import copy
import json
from pathlib import Path

from .workflow_io import environment, fingerprint, json_write, sha256


def reusable_provenance(spec, material, material_path, born, born_path, output):
    physical = copy.deepcopy(spec)
    # File locations and hashes of report manifests are provenance, not physics.
    physical['force_constant_model'].pop('manifest', None)
    physical['force_constant_model'].pop('sha256', None)
    contract = {
        'physical_model': physical,
        'material_inputs': {name: material['artifacts'][name]['sha256'] for name in
                            ('equilibrium', 'force_constants', 'hiphive_potential', 'covariances')},
        'born_modes': born['artifacts']['modes']['sha256'],
        'material_validation': {'status': material['status'], 'checks': material['checks']},
        'born_validation': {'status': born['numerical_status'], 'checks': born['checks']},
        'environment': environment(),
    }
    directory = Path(output)/'provenance'
    directory.mkdir(parents=True, exist_ok=True)
    current = {'material_manifest': sha256(material_path), 'born_manifest': sha256(born_path)}
    snapshots = {}
    for name, path in (('material_manifest', material_path), ('born_manifest', born_path)):
        target = directory/f'{name}_{current[name]}.json'
        if target.exists() and sha256(target) != current[name]:
            raise ValueError(f'archived provenance checksum mismatch: {target}')
        if not target.exists():
            target.write_bytes(Path(path).read_bytes())
        snapshots[name] = str(target)
    path = directory/f'cache_identity_{fingerprint(contract)}.json'
    if path.exists():
        record = json.loads(path.read_text())
        if fingerprint(record['scientific_input_contract']) != fingerprint(contract):
            raise ValueError('numerical cache contract mismatch')
    else:
        # This is also the original Benchmark 7 cache identity, so existing
        # numerical arrays need no migration or alteration.
        record = {'scientific_input_contract': contract,
                  'identity': {'model_specification': fingerprint(spec), **current,
                               'environment': environment()},
                  'identity_manifest_snapshots': snapshots}
        json_write(path, record)
    return record['identity'], {'identity_record': str(path), 'sha256': sha256(path),
                                'scientific_input_contract_sha256': fingerprint(contract),
                                'current_validation_manifests': current,
                                'current_manifest_snapshots': snapshots,
                                'identity_manifest_snapshots': record['identity_manifest_snapshots']}
