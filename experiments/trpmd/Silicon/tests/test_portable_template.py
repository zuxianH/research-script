"""Check template identities and new-destination preparation; no engine runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

MATERIAL = Path(__file__).resolve().parents[1]


class PortableTemplate(unittest.TestCase):
    def test_recorded_files_match_manifest(self):
        manifest = json.loads((MATERIAL / 'template.json').read_text())
        for name, expected in manifest['files_sha256'].items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((MATERIAL / name).read_bytes()).hexdigest(), expected)

    def test_prepare_after_copying_to_new_location(self):
        with tempfile.TemporaryDirectory(prefix='trpmd-template-') as temporary:
            destination = Path(temporary) / 'another-machine/TRPMD'
            shutil.copytree(MATERIAL.parent, destination)
            # Reuse this test interpreter's environment; do not reinstall packages.
            (destination / '.venv').symlink_to(Path(sys.prefix), target_is_directory=True)
            env = dict(os.environ, DRY_RUN='1')
            for name in ('PYTHON_BIN', 'LMP_BIN', 'IPI_BIN', 'SCIENCE_ACTIVATE',
                         'RUN_DIR', 'REPLICATE', 'RUNSTEPS', 'INIT_XYZ_SOURCE'):
                env.pop(name, None)
            for beads in (1, 5):
                env.update(NUM_BEADS=str(beads), NUM_CLIENT=str(beads), RUN_NAME=f'prepare_b{beads}')
                process = subprocess.run(['bash', 'scripts/run.sh'], cwd=destination / 'Silicon',
                                         env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                run = destination / f'Silicon/results/prepare_b{beads}'
                xml = ET.parse(run / 'inputs/input.xml').getroot()
                self.assertEqual(xml.find('system/initialize').get('nbeads'), str(beads))
                self.assertEqual(int(xml.findtext('total_steps')), 100)
                self.assertEqual((run / 'inputs/init_2x2x2.xyz').read_text().splitlines()[0], '32')
                entries = {entry.name for entry in run.iterdir()}
                self.assertEqual(entries, {'README.md', 'inputs', 'models', 'scripts', 'provenance'})
                metadata = dict(line.split('=', 1) for line in
                                (run / 'provenance/run_metadata.env').read_text().splitlines())
                python = Path(metadata['PYTHON_BIN'].strip('"'))
                self.assertEqual(python.resolve(), (destination / '.venv/bin/python').resolve())
                self.assertTrue(str(python).startswith(str(destination)))
                layout = json.loads((run / 'provenance/layout.json').read_text())['original_files']
                for record in layout.values():
                    self.assertEqual(hashlib.sha256((run / record['path']).read_bytes()).hexdigest(),
                                     record['sha256'])
                subprocess.run(['bash', '-n', str(run / 'scripts/continue.sh')], check=True)


if __name__ == '__main__':
    unittest.main()
