"""Diagnostic bulk-q spectra from independently refitted, corrected MD data."""

import argparse
import json
from pathlib import Path
import time
from hiphive import ForceConstantPotential
from yoshioka.lattice_dynamics import real_space_from_supercell
from .approximate_eels import run_covariance, run_born
from .workflow_io import ROOT, ResourceLimits, json_write, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='benchmarks/configs/approximate_eels_ehm_20261102_v1.json')
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    output = ROOT/config['output']; output.mkdir(parents=True, exist_ok=True)
    limits = ResourceLimits(config['runtime_limit_seconds'], config['memory_limit_gib'], 10000)
    records = []
    for temperature in config['temperatures_k']:
        source = ROOT/f'benchmark_results/repair_20261102_v2/B11_refit/T{temperature}_n128.fcp'
        target = ROOT/config['primitive_force_constants_by_temperature'][str(temperature)]
        if not target.exists():
            start = time.monotonic(); fcp = ForceConstantPotential.read(str(source))
            unit = fcp.primitive_structure; repeats = (5, 5, 5)
            fc = fcp.get_force_constants(unit.repeat(repeats)).get_fc_array(2)
            model, metadata = real_space_from_supercell(unit, repeats, fc)
            model.save(target)
            json_write(target.with_suffix('.json'), {'source_fcp': str(source), 'source_sha256': sha256(source),
                'representation_change': 'Exact primitive-cell export of existing corrected-MD fit; no refitting to EELS.',
                'model_metadata': metadata, 'seconds': time.monotonic()-start,
                'status': 'UNACCEPTED_EFFECTIVE_HARMONIC_DIAGNOSTIC',
                'training_reference': f'benchmark_results/repair_20261102_v2/B11_refit/T{temperature}_n128.json'})
        records.append(str(target))
    run_covariance(config, output, limits)
    run_born(config, output, limits)
    json_write(output/'scope.json', {'status': 'DIAGNOSTIC_CANDIDATE', 'force_constants': records,
        'original_B11A_status': 'FAIL preserved', 'volume': 'same fixed zero-temperature equilibrium geometry',
        'thermal_occupation': 'Bose occupations at labelled300/1000K; actual corrected-MD fit temperatures are recorded separately in B11_refit',
        'not_included': ['intrinsic phonon linewidths', 'FC3/FC4 self energies', 'thermal expansion'],
        'resources': limits.costs()})


if __name__ == '__main__': main()
