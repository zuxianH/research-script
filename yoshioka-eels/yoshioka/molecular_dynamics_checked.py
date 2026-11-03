"""Checkpoint-safe classical MOD trajectories; original implementation preserved.

write_restart can remap atoms and invalidate neighbor communication state.
Before continuing with run pre no, explicitly reinitialize the force/neighbor
state with a zero-step run. This does not advance time or rescale velocities.

Only the LAMMPS boundary uses metal units. Stored displacements are m, forces N,
velocities m/s, energies J, times s and temperatures K. Production is NVE after
NVT equilibration, so no thermostat linewidth acts in the production segment.
"""

import ctypes
import json
from pathlib import Path
import time

import numpy as np
from scipy.constants import elementary_charge

from .mod_forces import ModifiedTersoffForces


def checkpointed_md(unit, potential, settings, output, limits):
    from .provenance import fingerprint, json_write, sha256
    atoms = unit.repeat(tuple(settings['repetitions']))
    frames = settings['production_steps']//settings['sample_stride']
    if settings['production_steps'] % settings['sample_stride']:
        raise ValueError('production length must be an integer number of samples')
    contract = {'settings': settings, 'cell_angstrom': np.asarray(atoms.cell).tolist(),
                'reference_positions_angstrom': atoms.positions.tolist(), 'masses_amu': atoms.get_masses().tolist(),
                'potential_sha256': sha256(potential), 'source_sha256': sha256(__file__)}
    digest = fingerprint(contract)
    folder = Path(output)/digest[:20]
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'digest': digest, 'contract': contract, 'completed_frames': 0, 'cumulative_seconds': 0., 'status': 'INCOMPLETE'}
    if state['digest'] != digest or not 0 <= state['completed_frames'] <= frames:
        raise ValueError('incompatible molecular-dynamics checkpoint')
    shapes = {'displacements_m': (frames, len(atoms), 3), 'forces_n': (frames, len(atoms), 3),
              'velocities_m_s': (frames, len(atoms), 3), 'temperature_k': (frames,), 'total_energy_j': (frames,)}
    limits.check(estimated_bytes=frames*len(atoms)*3*8*3+250*1024**2)
    arrays = {}
    for key, shape in shapes.items():
        path = folder/f'{key}.npy'
        if path.exists():
            arrays[key] = np.load(path, mmap_mode='r+', allow_pickle=False)
            if arrays[key].shape != shape or arrays[key].dtype != np.dtype('float64'):
                raise ValueError('trajectory checkpoint shape/dtype mismatch')
        else:
            if state['completed_frames']:
                raise ValueError('missing trajectory array for a committed prefix')
            arrays[key] = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=shape)
    if state['status'] == 'COMPLETE':
        return folder, state
    started = time.monotonic()
    last_commit = started
    resumed = state['completed_frames']
    with ModifiedTersoffForces(atoms, potential) as calculator:
        lmp = calculator.lmp
        if resumed:
            restart = folder/state['restart_file']
            if sha256(restart) != state['restart_sha256']:
                raise ValueError('MD restart checksum mismatch')
            lmp.commands_list(['clear', f'read_restart "{restart.resolve()}"', 'pair_style tersoff/mod',
                              f'pair_coeff * * "{Path(potential).resolve()}" Si', 'neighbor 0.5 bin',
                              'neigh_modify every 1 delay 0 check yes'])
        else:
            lmp.commands_list([f"timestep {settings['timestep_fs']/1000:.17g}",
                f"velocity all create {settings['temperature_k']} {settings['seed']} mom yes rot no dist gaussian",
                'thermo 0', 'thermo_style custom step temp pe ke etotal press',
                f"fix equil all nvt temp {settings['temperature_k']} {settings['temperature_k']} {settings['thermostat_time_fs']/1000:.17g}",
                f"run {settings['equilibration_steps']} pre yes post no", 'unfix equil'])
            state['equilibration_final_temperature_k'] = float(lmp.get_thermo('temp'))
        lmp.commands_list(['thermo 0', 'thermo_style custom step temp pe ke etotal press', 'fix production all nve', 'run 0 pre yes post no'])
        cell = np.diag(np.asarray(atoms.cell))
        try:
            for frame in range(resumed, frames):
                if frame % 100 == 0:
                    limits.check()
                lmp.command(f"run {settings['sample_stride']} pre no post no")
                def gather(name):
                    pointer = lmp.gather_atoms(name, 1, 3)
                    return np.ctypeslib.as_array(pointer, shape=(3*len(atoms),)).reshape(-1, 3).copy()
                positions = gather('x')
                displacement = positions-atoms.positions
                displacement -= np.rint(displacement/cell)*cell
                # ASR makes uniform COM shifts irrelevant. Record removal to
                # avoid a free rigid-translation coordinate in a crystal fit.
                displacement -= np.average(displacement, axis=0, weights=atoms.get_masses())
                arrays['displacements_m'][frame] = displacement*1e-10
                arrays['forces_n'][frame] = gather('f')*elementary_charge/1e-10
                arrays['velocities_m_s'][frame] = gather('v')*100  # angstrom/ps
                arrays['temperature_k'][frame] = lmp.get_thermo('temp')
                arrays['total_energy_j'][frame] = lmp.get_thermo('etotal')*elementary_charge
                state['completed_frames'] = frame+1
                if (frame+1) % 200 == 0 or frame+1 == frames:
                    for array in arrays.values():
                        array.flush()
                    restart = folder/f'restart_{frame+1}.bin'
                    lmp.command(f'write_restart "{restart.resolve()}"')
                    # write_restart remaps atoms; run pre no alone may reuse stale neighbors.
                    lmp.command('run 0 pre yes post no')
                    previous = state.get('restart_file')
                    state.update(restart_file=restart.name, restart_sha256=sha256(restart))
                    state['cumulative_seconds'] += time.monotonic()-last_commit
                    json_write(state_path, state)
                    last_commit = time.monotonic()
                    if previous and previous != restart.name:
                        (folder/previous).unlink(missing_ok=True)
        except (TimeoutError, MemoryError) as exc:
            # Roll back to the last restart/prefix pair; uncommitted array rows
            # are overwritten on resume. No partial trajectory is normalized.
            committed = json.loads(state_path.read_text()) if state_path.exists() else None
            if committed is not None:
                state = committed
            else:
                state['completed_frames'] = 0
            state['stop_reason'] = str(exc)
    state['status'] = 'COMPLETE' if state['completed_frames'] == frames else 'INCOMPLETE'
    state['latest_invocation_seconds'] = time.monotonic()-started
    state['resumed_from_frames'] = resumed
    state['sample_interval_s'] = settings['sample_stride']*settings['timestep_fs']*1e-15
    state['production_duration_s'] = settings['production_steps']*settings['timestep_fs']*1e-15
    state['production_ensemble'] = 'classical NVE after NVT equilibration'
    state['checkpoint_neighbor_policy'] = 'write_restart followed by run 0 pre yes post no before run pre no'
    state['periodic_displacement_policy'] = 'minimum image relative to fixed equilibrium reference; mass-weighted COM displacement removed; no force symmetrization or odd-force projection'
    state['units'] = {'displacements': 'm', 'forces': 'N', 'velocities': 'm/s', 'temperature': 'K', 'total_energy': 'J'}
    if state['status'] == 'COMPLETE':
        state['mean_temperature_k'] = float(np.mean(arrays['temperature_k']))
        state['temperature_std_k'] = float(np.std(arrays['temperature_k']))
        drift = np.asarray(arrays['total_energy_j'])-arrays['total_energy_j'][0]
        kinetic_scale = 1.5*(len(atoms)-1)*1.380649e-23*state['mean_temperature_k']
        state['max_energy_drift_relative_to_thermal_kinetic_energy'] = float(np.max(abs(drift))/kinetic_scale)
        state['max_displacement_angstrom'] = float(np.max(np.linalg.norm(arrays['displacements_m'], axis=-1))*1e10)
    json_write(state_path, state)
    return folder, state
