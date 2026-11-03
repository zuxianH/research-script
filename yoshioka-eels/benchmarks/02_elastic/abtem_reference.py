"""ASE structure verification and the independent abTEM 1.0.11 adapter.

Only this benchmark adapter imports abTEM. The SI elastic solver is independent.
Reference source was inspected for integrated potentials, axes, normalization,
transmission ordering, FFTs, and filtering; see source_audit() and the report.
"""

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np

from yoshioka import TransverseGrid


REFERENCE_SETTINGS = {
    "device": "cpu",
    "precision": "float64",
    "fft": "fftw",
    "fftw.threads": 1,
    "fftw.planning_effort": "FFTW_ESTIMATE",
    "dask.lazy": False,
    "antialias.cutoff": 0.6666666,
    "antialias.taper": 0.01,
}


def reference_configuration():
    import abtem
    return abtem.config.set(REFERENCE_SETTINGS)


def load_structure(path: Path):
    """Read Å coordinates unchanged, explicitly mapping LAMMPS type 1 to Si."""
    from ase.io import read
    from ase.neighborlist import neighbor_list

    path = Path(path).resolve()
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    atoms = read(path, format="lammps-data", atom_style="atomic", units="metal", Z_of_type={1: 14})
    expected_cell = np.array([5.43053, 3.83996, 3.83996])
    cell = np.asarray(atoms.cell)
    if len(atoms) != 4 or not np.all(atoms.numbers == 14):
        raise ValueError("the supplied cell must contain four Si atoms (LAMMPS type 1 -> Z=14)")
    if not np.allclose(cell, np.diag(expected_cell), rtol=0, atol=1e-8):
        raise ValueError(f"unexpected Si [110] orthorhombic cell: {cell.tolist()}")
    # ASE neighbor_list retains distinct periodic images, so tetrahedral
    # coordination is counted correctly even in this four-atom cell.
    ideal_distance = expected_cell[0] * np.sqrt(3) / 4
    i, j, distances, images = neighbor_list("ijdS", atoms, ideal_distance * 1.03)
    coordination = np.bincount(i, minlength=4)
    if not np.all(coordination == 4) or not np.allclose(distances, ideal_distance, rtol=2e-5, atol=0):
        raise ValueError("periodic nearest neighbors are inconsistent with diamond Si")
    scaled = atoms.get_scaled_positions(wrap=False)
    expected_scaled = np.array([[1/8, 1/4, 1/4], [7/8, 1/4, 3/4], [3/8, 3/4, 1/4], [5/8, 3/4, 3/4]])
    order = np.lexsort((scaled[:, 2], scaled[:, 1], scaled[:, 0]))
    expected_order = np.lexsort((expected_scaled[:, 2], expected_scaled[:, 1], expected_scaled[:, 0]))
    projected_ok = np.allclose(scaled[order], expected_scaled[expected_order], rtol=0, atol=1e-7)
    if not projected_ok:
        raise ValueError("projected columns do not match the supplied [110] arrangement")
    verification = {
        "input_path": str(path), "input_sha256": checksum,
        "coordinate_assumption": "LAMMPS coordinates and cell lengths are angstrom; ASE units='metal'",
        "atom_type_mapping": {"1": 14}, "atom_count": len(atoms),
        "atomic_numbers": atoms.numbers.tolist(), "cell_angstrom": cell.tolist(),
        "positions_angstrom": atoms.positions.tolist(), "fractional_positions": scaled.tolist(),
        "periodic_boundary_flags": atoms.pbc.tolist(),
        "ideal_diamond_nn_angstrom": float(ideal_distance),
        "nearest_neighbor_min_angstrom": float(np.min(distances)),
        "nearest_neighbor_max_angstrom": float(np.max(distances)),
        "coordination_numbers": coordination.tolist(),
        "neighbor_atom_indices": np.column_stack((i, j)).tolist(),
        "neighbor_image_shifts": images.tolist(),
        "projected_arrangement_verified": bool(projected_ok),
        "projected_dumbbell_separation_angstrom": float(expected_cell[0] / 4),
        "orientation": "use supplied x,y,z unchanged; propagation along supplied z ([110])",
        "atomic_transform_applied": False, "thermal_displacements_added": False,
        "debye_waller_smearing_added": False,
    }
    return atoms, verification


@dataclass
class SharedSlices:
    native: object  # abTEM PotentialArray, axes (slice,x,y), units V Å.
    projected_potential_v_m: np.ndarray  # axes (slice,y,x).
    thicknesses_m: np.ndarray
    grid: TransverseGrid
    incident_native: object
    incident_si: np.ndarray
    native_to_si_wave_factor: float
    metadata: dict


def build_shared_slices(atoms, n: int, slices_per_repeat: int, energy_ev: float) -> SharedSlices:
    """Build each distinct finite projected slice once for a perfect periodic cell.

    abTEM's finite integrator pads periodic atomic images in x,y,z. Therefore
    the bulk potential of an integer z-repeat is exactly a repetition of these
    integrated slices. We verify this against a direct 52-atom build separately.
    Exporting unit arrays plus an explicit slice-index map is lossless and avoids
    storing hundreds of identical copies. No frozen-phonon ensemble is created.
    """
    import abtem

    lengths = atoms.cell.lengths()
    thickness = float(lengths[2]) / slices_per_repeat
    builder = abtem.Potential(
        atoms, gpts=(n, n), slice_thickness=(thickness,) * slices_per_repeat,
        parametrization="lobato", projection="finite", plane="xy",
        origin=(0.0, 0.0, 0.0), periodic=True, device="cpu",
    )
    native = builder.build(lazy=False)
    transformed = builder.get_transformed_atoms()
    if not np.array_equal(transformed.positions, atoms.positions) or not np.array_equal(transformed.cell, atoms.cell):
        raise RuntimeError("reference unexpectedly transformed the input atoms")
    if native.array.dtype != np.float64:
        raise RuntimeError(f"reference potential is not double precision: {native.array.dtype}")
    labels = [axis.label for axis in native.base_axes_metadata[-2:]]
    if labels != ["x", "y"]:
        raise RuntimeError(f"unsupported reference spatial axis order {labels}")
    grid = TransverseGrid(float(native.extent[0]) * 1e-10, float(native.extent[1]) * 1e-10, n, n)
    # No physical shift: arrays keep the supplied [0,L) coordinate origin.
    # Benchmark 1's FFT kernel is translation invariant and depends only on dx,dy.
    potential_si = np.ascontiguousarray(native.array.swapaxes(-2, -1) * 1e-10, dtype=np.float64)
    incident = abtem.PlaneWave(energy=energy_ev, gpts=n, extent=native.extent, normalize=True, device="cpu").build(lazy=False)
    factor = (n * n) / np.sqrt(grid.lx_m * grid.ly_m)
    incident_si = np.ascontiguousarray(incident.array.T * factor, dtype=np.complex128)
    if incident.array.dtype != np.complex128:
        raise RuntimeError("reference entrance wave is not complex128")
    metadata = {
        "native_potential_shape": list(native.array.shape),
        "native_potential_dtype": str(native.array.dtype), "si_potential_dtype": str(potential_si.dtype),
        "native_potential_units": "volt angstrom (integrated); misleading metadata label eV/e is not used for conversion",
        "si_potential_units": "volt metre (integrated)",
        "potential_conversion": "transpose slice,x,y -> slice,y,x; multiply by 1e-10; no additional slice thickness factor",
        "native_axis_order": ["slice", "x", "y"], "si_axis_order": ["slice", "y", "x"],
        "native_extent_angstrom": list(native.extent), "native_sampling_angstrom": list(native.sampling),
        "native_slice_thickness_angstrom": list(native.slice_thickness),
        "cell_mean_electrostatic_potential_v": float(np.sum(np.mean(native.array, axis=(-2, -1))) / np.sum(native.slice_thickness)),
        "native_wave_dtype": str(incident.array.dtype), "si_wave_dtype": str(incident_si.dtype),
        "native_incident_amplitude": float(incident.array[0, 0].real),
        "si_incident_amplitude_m_inv": float(incident_si[0, 0].real),
        "native_to_si_wave_factor": float(factor),
        "normalization": "abTEM amplitude=1/(Nx*Ny), sum|DFT psi|^2=1; SI phi=psi*(Nx*Ny)/sqrt(Lx*Ly)",
        "array_coordinate_origin_m": [0.0, 0.0], "atomic_transform_applied": False,
        "finite_integrator": type(builder.integrator).__name__,
        "potential_cutoff_tolerance_v": builder.integrator.cutoff_tolerance,
        "integration_step_angstrom": builder.integrator.integration_step,
        "quadrature_order": builder.integrator.quad_order,
        "atomic_cutoff_radius_angstrom": float(builder.integrator.cutoff("Si")),
        "boundary_convention": "abTEM periodic bulk potential, including periodic image tails at z=0 and z=L; no vacuum padding",
    }
    return SharedSlices(native, potential_si, np.asarray(native.slice_thickness) * 1e-10, grid, incident, incident_si, float(factor), metadata)


def verify_direct_repeated_build(shared: SharedSlices, atoms, repetitions: int) -> dict:
    """Check lossless unit-slice reuse against a full repeated-atom construction."""
    import abtem

    repeated = atoms.repeat((1, 1, repetitions))
    thicknesses = tuple(shared.native.slice_thickness) * repetitions
    builder = abtem.Potential(
        repeated, gpts=shared.native.gpts, slice_thickness=thicknesses,
        parametrization="lobato", projection="finite", plane="xy", periodic=True,
    )
    direct = builder.build(lazy=False)
    tiled = np.tile(shared.native.array, (repetitions, 1, 1))
    relative = float(np.linalg.norm(direct.array - tiled) / np.linalg.norm(direct.array))
    direct_exit = shared.incident_native.copy().multislice(direct)
    crystal = abtem.CrystalPotential(shared.native, repetitions=(1, 1, repetitions))
    crystal_exit = shared.incident_native.copy().multislice(crystal)
    wave_error = float(np.linalg.norm(direct_exit.array - crystal_exit.array) / np.linalg.norm(direct_exit.array))
    return {
        "repetitions": [1, 1, repetitions], "atom_count": len(repeated),
        "relative_potential_error": relative,
        "max_potential_difference_v_angstrom": float(np.max(np.abs(direct.array - tiled))),
        "relative_exit_wave_error": wave_error,
        "passed": bool(relative < 1e-10 and wave_error < 1e-10),
    }


def export_shared_inputs(shared: SharedSlices, path: Path, repetitions: tuple[int, ...]) -> None:
    """Save actual unit arrays plus every specimen's full ordered slice map."""
    from abtem.antialias import antialias_aperture

    n_unit = len(shared.thicknesses_m)
    arrays = {
        "potential_v_angstrom_sxy": shared.native.array,
        "unit_slice_thickness_angstrom": np.asarray(shared.native.slice_thickness),
        "extent_angstrom": np.asarray(shared.native.extent),
        "sampling_angstrom": np.asarray(shared.native.sampling),
        "incident_native_xy": shared.incident_native.array,
        "incident_si_yx": shared.incident_si,
        "native_to_si_wave_factor": np.asarray(shared.native_to_si_wave_factor),
        "native_antialias_aperture_yx": antialias_aperture(shared.native.gpts, shared.native.sampling, xp=np).T,
    }
    for repeat in repetitions:
        arrays[f"slice_unit_indices_r{repeat}"] = np.tile(np.arange(n_unit, dtype=np.int64), repeat)
        arrays[f"slice_thickness_m_r{repeat}"] = np.tile(shared.thicknesses_m, repeat)
        arrays[f"slice_edges_m_r{repeat}"] = np.r_[0.0, np.cumsum(np.tile(shared.thicknesses_m, repeat))]
    np.savez_compressed(path, **arrays)


@dataclass
class ReferenceResult:
    wave_si: np.ndarray
    diffraction: np.ndarray
    norm_history: np.ndarray
    metadata: dict


def propagate_reference(shared: SharedSlices, repetitions: int, check_driver: bool = False) -> ReferenceResult:
    """Use abTEM's own multislice step, FFTW, transmission and Fresnel routines.

    A transparent loop around the native step permits norm recording at every
    slice without retaining a large wave stack. Primary cases are additionally
    compared against the public Waves.multislice driver.
    """
    import abtem
    from abtem.antialias import AntialiasAperture
    from abtem.multislice import FresnelPropagator, conventional_multislice_step

    crystal = abtem.CrystalPotential(shared.native, repetitions=(1, 1, repetitions))
    waves = shared.incident_native.copy()
    propagator, aperture = FresnelPropagator(), AntialiasAperture()
    native_norm = lambda wave: float(np.prod(shared.grid.shape) * np.sum(np.abs(wave.array) ** 2))
    norms = [native_norm(waves)]
    for potential_slice in crystal.generate_slices():
        waves = conventional_multislice_step(
            waves, potential_slice, propagator, aperture,
            conjugate=False, transpose=False, order=1,
        )
        norms.append(native_norm(waves))
    metadata = {
        "exit_wave_dtype": str(waves.array.dtype),
        "ordering": "filtered transmission followed by filtered full-slice Fresnel propagation",
        "entrance_z_m": 0.0, "exit_z_m": float(sum(crystal.slice_thickness) * 1e-10),
        "reference_step": "abtem.multislice.conventional_multislice_step(order=1, transpose=False, conjugate=False)",
    }
    if check_driver:
        public_exit = shared.incident_native.copy().multislice(crystal)
        driver_error = float(np.linalg.norm(public_exit.array - waves.array) / np.linalg.norm(waves.array))
        metadata["public_multislice_driver_relative_wave_error"] = driver_error
        if driver_error >= 1e-12:
            raise RuntimeError("instrumented abTEM steps disagree with the public multislice driver")
    patterns = waves.diffraction_patterns(
        max_angle="full", fftshift=False, parity="same", block_direct=False, renormalize=False,
    )
    if tuple(patterns.array.shape) != tuple(shared.native.gpts):
        raise RuntimeError("reference diffraction was unexpectedly cropped")
    metadata["diffraction_dtype"] = str(patterns.array.dtype)
    metadata["diffraction_options"] = {"max_angle": "full", "fftshift": False, "parity": "same", "block_direct": False, "renormalize": False}
    return ReferenceResult(
        np.ascontiguousarray(waves.array.T * shared.native_to_si_wave_factor),
        np.ascontiguousarray(patterns.array.T), np.asarray(norms, dtype=np.float64), metadata,
    )


def source_audit(energy_ev: float) -> dict:
    import abtem
    from abtem.core.energy import energy2sigma, energy2wavelength
    from scipy.constants import elementary_charge, hbar, c, m_e
    from yoshioka import electron_wavenumber, interaction_coefficient

    root = Path(abtem.__file__).parent
    source_files = ["potentials/iam.py", "integrals.py", "multislice.py", "antialias.py", "waves.py", "core/energy.py", "core/fft.py"]
    energy_j = energy_ev * elementary_charge
    wavelength_m = 2 * np.pi / electron_wavenumber(energy_j)
    sigma_si = interaction_coefficient(energy_j)
    return {
        "reference_settings": dict(REFERENCE_SETTINGS),
        "versions": {name: importlib.metadata.version(name) for name in ("abtem", "ase", "numpy", "scipy", "matplotlib", "numba", "pyfftw", "dask")},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
        "integrated_potential_evidence": "QuadratureProjectionIntegrals integrates between slice bounds; PotentialArray._transmission_function uses exp(i*sigma*array) without slice thickness",
        "native_sigma_per_v_angstrom": energy2sigma(energy_ev),
        "ours_sigma_per_v_m": sigma_si,
        "sigma_relative_difference_native_to_ours": energy2sigma(energy_ev) / (sigma_si * 1e-10) - 1,
        "native_wavelength_angstrom": energy2wavelength(energy_ev),
        "ours_wavelength_m": wavelength_m,
        "wavelength_relative_difference_native_to_ours": energy2wavelength(energy_ev) * 1e-10 / wavelength_m - 1,
        "ours_constants_si": {"e_c": elementary_charge, "hbar_j_s": hbar, "c_m_per_s": c, "electron_mass_kg": m_e},
        "constant_difference_explanation": "abTEM energy routines use ASE's default older CODATA constants; our code uses installed scipy.constants. No coefficient fit or phase correction is applied.",
        "documentation": [
            "https://abtem.github.io/doc/user_guide/walkthrough/potentials.html",
            "https://abtem.github.io/doc/user_guide/walkthrough/multislice.html",
            "https://abtem.github.io/doc/user_guide/appendix/antialiasing.html",
        ],
    }
