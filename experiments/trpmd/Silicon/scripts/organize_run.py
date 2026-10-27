"""Group a finished/prepared run while preserving every original file's bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def category(name):
    if name.endswith(('.snapcoeff', '.snapparam')):
        return 'models'
    if name in ('RESTART', 'simulation.restart'):
        return 'restart'
    if name.endswith('.log') or name.startswith('log.'):
        return 'logs'
    if name == 'simulation.out' or name.startswith('simulation.pos_'):
        return 'outputs'
    if name in ('input.xml', 'lmp_Si_ipi.in') or name.startswith(('Si_', 'init_', '.continue_')):
        return 'inputs'
    return 'provenance'


def organize(root):
    manifest = root / 'provenance/layout.json'
    if manifest.exists():
        return
    originals = sorted(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in originals):
        raise ValueError('Expected a flat run folder containing regular files only')
    records = {path.name: {'path': f'{category(path.name)}/{path.name}',
                          'sha256': digest(path)} for path in originals}
    for name, record in records.items():
        destination = root / record['path']
        destination.parent.mkdir(exist_ok=True)
        (root / name).rename(destination)
    (root / 'provenance').mkdir(exist_ok=True)
    manifest.write_text(json.dumps({'original_files': records}, indent=2) + '\n')
    shutil.copy2(Path(__file__), root / 'provenance/organize_run.py')
    (root / 'scripts').mkdir(exist_ok=True)
    continuation = root / 'scripts/continue.sh'
    continuation.write_text('''#!/bin/bash
# Continue in a separate segment; this completed run stays unchanged.
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RUN_DIR/provenance/run_metadata.env"
SEGMENT="$("$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import shutil
import sys
import tempfile
root = Path(sys.argv[1])
parent = root / 'restart/continuations'
parent.mkdir(exist_ok=True)
segment = Path(tempfile.mkdtemp(prefix='segment_', dir=parent))
records = json.loads((root / 'provenance/layout.json').read_text())['original_files']
for name, record in records.items():
    if Path(record['path']).parts[0] not in ('logs', 'outputs'):
        shutil.copy2(root / record['path'], segment / name)
print(segment)
PY
)"
echo "Continuation segment: $SEGMENT"
status=0
bash "$SEGMENT/continue.sh" || status=$?
"$PYTHON_BIN" "$RUN_DIR/provenance/organize_run.py" "$SEGMENT"
exit "$status"
''')
    continuation.chmod(0o755)
    (root / 'README.md').write_text(f'''# {root.name}

```text
inputs/       original executed XML/LAMMPS inputs and initial structures
models/       SNAP potential
outputs/      simulation.out and bead trajectories
logs/         i-PI and LAMMPS logs
restart/      checkpoints and separate continuation segments
provenance/   original scripts, metadata, duplicate snapshots and file hashes
scripts/      continuation entry point
```

Files were moved without changing their contents. `provenance/layout.json`
maps original names to current paths and records their SHA-256 hashes.
Native inputs retain the relative paths used at execution time.

To add ten steps on an allocated compute node, from this run folder:

```bash
ADD_STEPS=10 bash scripts/continue.sh
```

Continuation stages the saved inputs/checkpoint in a new folder under
`restart/continuations/`; the original outputs and checkpoints stay unchanged.
The continuation folder is organized the same way after the engines stop.
Runtime paths are recorded in `provenance/run_metadata.env`.
''')
    assert all(digest(root / record['path']) == record['sha256']
               for record in records.values())
    print(f'Organized {len(records)} files without changing their contents: {root}')


if __name__ == '__main__':
    organize(Path(sys.argv[1]).resolve())
