"""Check fresh 32-atom, 100-step smoke runs; print a JSON evidence report."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


def check(path):
    manifest = path / 'provenance/layout.json'
    records = json.loads(manifest.read_text())['original_files'] if manifest.exists() else {}

    def locate(name):
        return path / records[name]['path'] if records else path / name

    for name, record in records.items():
        assert hashlib.sha256(locate(name).read_bytes()).hexdigest() == record['sha256'], name
    settings = ET.parse(locate('input.xml')).getroot()
    count = int(settings.find('system/initialize').get('nbeads'))
    assert int(settings.findtext('total_steps')) == 100
    assert float(settings.findtext('.//dynamics/timestep')) == 0.5
    assert settings.find('.//thermostat').get('mode') == 'pile_g'
    assert float(settings.findtext('.//thermostat/tau')) == 1000000
    assert float(settings.findtext('.//thermostat/pile_lambda')) == 0.5
    restart = ET.parse(locate('RESTART')).getroot()
    beads = restart.find('system/beads')
    assert int(restart.findtext('step')) == 100
    assert beads.attrib == {'natoms': '32', 'nbeads': str(count)}
    for tag in ('q', 'p'):
        text = beads.findtext(tag).replace('[', '').replace(']', '').replace(',', ' ')
        array = np.fromstring(text, sep=' ')
        assert array.size == count * 32 * 3 and np.isfinite(array).all(), tag
        if tag == 'q' and count > 1:
            q = array.reshape(count, 32, 3)
            assert np.max(np.abs(q - q.mean(axis=0))) > 0
    values = np.loadtxt(locate('simulation.out'))
    assert values.shape == (101, 5) and np.isfinite(values).all()
    assert np.array_equal(values[:, 0], np.arange(101))
    assert np.allclose(values[:, 1], np.arange(101) * 0.5)
    connections = locate('ipi.log').read_text().count('Handshaking was successful')
    assert connections == count
    trajectories = sorted(locate('simulation.out').parent.glob('simulation.pos_*.xyz'))
    assert len(trajectories) == count
    for trajectory in trajectories:
        lines = trajectory.read_text().splitlines()
        assert len(lines) == 11 * 34
        frames = []
        for start in range(0, len(lines), 34):
            assert int(lines[start]) == 32
            atoms = [line.split() for line in lines[start + 2:start + 34]]
            assert all(atom[0] == 'Si' for atom in atoms)
            frames.append([[float(x) for x in atom[1:4]] for atom in atoms])
        assert np.isfinite(frames).all() and not np.array_equal(frames[0], frames[-1])
    logs = list(locate('ipi.log').parent.glob('lammps_client_*.log'))
    assert len(logs) == count
    for log in logs:
        errors = [line for line in log.read_text().splitlines() if 'ERROR' in line]
        assert all('Got EXIT message from i-PI' in line for line in errors), (log, errors)
    names = ['run_source.sh', 'input.xml', 'lmp_Si_ipi.in', 'monitor_run.sh',
             'Si_110_base.lmp', 'Si_110_2x2x2.lmp', 'SNAPotential.snapcoeff',
             'SNAPotential.snapparam']
    return dict(directory=str(path), atoms=32, beads=count, clients_connected=connections,
                final_step=100, time_fs=50, finite_property_rows=101, frames_per_bead=11,
                finite_checkpoint_q_p=True, source_sha256={
                    name: hashlib.sha256(locate(name).read_bytes()).hexdigest()
                    for name in names})


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python tests/check_smoke.py results/RUN [results/RUN ...]')
    print(json.dumps({'runs': [check(Path(arg)) for arg in sys.argv[1:]]}, indent=2))
