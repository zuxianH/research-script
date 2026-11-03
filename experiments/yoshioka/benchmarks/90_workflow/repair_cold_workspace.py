"""Prepare fresh repair outputs while reusing immutable reference inputs.

This is a same-machine reproduction workspace, not a portable archive. Source
is copied; the environment and historical/data inputs are linked read-only by
workflow convention. New runners write only below the fresh repair directory.
"""
import argparse,shutil
from pathlib import Path
from .workflow_io import ROOT


def prepare(destination):
    destination=Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):raise ValueError('destination must be absent or empty')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('yoshioka','benchmarks','tests'):
        shutil.copytree(ROOT/name,destination/name,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    for path in ROOT.iterdir():
        if path.is_file():(destination/path.name).symlink_to(path.resolve())
    (destination/'.venv').symlink_to(ROOT/'.venv',target_is_directory=True)
    old=ROOT/'benchmark_results';new=destination/'benchmark_results';new.mkdir()
    for path in old.iterdir():
        if path.name!='repair_20261102_v2':(new/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
    fresh=new/'repair_20261102_v2';fresh.mkdir()
    evidence=fresh/'reused_evidence';evidence.mkdir()
    # Clean implementation controls and the documented failed prototype are
    # read-only reference evidence, not cold calculation output.
    for name in ('B11_audit','B11','MD_checkpoint_force_audit.json','MD_checked_pilot.json'):
        source=old/'repair_20261102_v2'/name
        if source.exists():(evidence/name).symlink_to(source.resolve(),target_is_directory=source.is_dir())
    for name in ('resume_profiles.json','protected_before.json','initial_diagnosis.json','initial_diagnosis.md','methods_and_limits.md','reproduce_and_resume.sh','preserve.py'):
        source=old/'repair_20261102_v2'/name
        if source.exists():shutil.copy2(source,fresh/name)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',required=True)
    args=parser.parse_args();print(prepare(args.destination))


if __name__=='__main__':main()
