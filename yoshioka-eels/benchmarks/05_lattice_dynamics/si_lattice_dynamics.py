"""Benchmark 5: harmonic Si MOD lattice dynamics and independent hiPhive fit."""

import argparse
import contextlib
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.optimize import brentq

from .abtem_reference import load_structure
from .harmonic_response import constants_record
from .workflow_io import ROOT, ResourceLimits, artifact, environment, fingerprint, json_write, previous_statuses, sha256
from yoshioka.mod_forces import ModifiedTersoffForces, central_force_constants, complete_force_constants, correct_sum_rules, force_constant_residuals, potential_metadata
from yoshioka.lattice_dynamics import RealSpaceForceConstants, covariance_mesh, real_space_from_supercell


@dataclass(frozen=True)
class LatticeConfig:
    reference_supercells: tuple = ((3, 4, 4), (4, 5, 5))
    displacement_angstrom: tuple = (0.02, 0.01, 0.005, 0.0025)
    training_supercells: tuple = ((2, 3, 3), (3, 4, 4))
    training_pair_counts: tuple = (1, 2, 4, 8, 16)
    training_sigmas_angstrom: tuple = (0.01, 0.003, 0.001)
    second_order_cutoffs_angstrom: tuple = (3.0, 3.6, 4.0, 4.5, 5.0)
    regression_ridges: tuple = (0.0, 1e-12, 1e-8)
    covariance_meshes: tuple = (8, 12, 16, 24, 32, 48, 64)
    covariance_relative_tolerance: float = 0.01
    force_relative_tolerance: float = 0.01
    force_constant_relative_tolerance: float = 1e-3
    phonon_energy_tolerance_mev: float = 0.1
    hermiticity_tolerance: float = 1e-10
    sum_rule_tolerance: float = 1e-8
    gamma_zero_tolerance_mev: float = 1e-3
    selected_cutoff_angstrom: float = 4.0
    selected_sigma_angstrom: float = 0.001
    training_seed: int = 120713
    validation_seed: int = 902813
    validation_structures: int = 8
    runtime_limit_seconds: float = 1800
    memory_limit_gib: float = 4


def _unit_at_lattice(original, a):
    unit = original.copy()
    unit.set_cell(np.diag([a, a/np.sqrt(2), a/np.sqrt(2)]), scale_atoms=True)
    return unit


def equilibrium_structure(original, potential, output):
    from ase.io import read, write
    cache = output/"equilibrium.json"
    if cache.exists():
        record = json.loads(cache.read_text())
        if record["potential_sha256"] == sha256(potential):
            return read(output/"equilibrium.extxyz"), record
    evaluations = []
    def evaluate(a):
        unit = _unit_at_lattice(original, a)
        supercell = unit.repeat((2, 2, 2))
        with ModifiedTersoffForces(supercell, potential) as calculator:
            data = calculator.evaluate()
        record = {"lattice_parameter_angstrom": a, "energy_ev_per_atom": data["energy_ev"]/len(supercell),
                  "pressure_bar_voigt": data["pressure_bar_voigt"].tolist(), "maximum_force_ev_per_angstrom": float(np.max(np.linalg.norm(data["forces_ev_per_angstrom"], axis=1)))}
        evaluations.append(record)
        return unit, record
    a = brentq(lambda value: np.mean(evaluate(value)[1]["pressure_bar_voigt"][:3]), 5.3, 5.6, xtol=1e-11)
    unit, optimum = evaluate(a)
    record = dict(optimum)
    volume_checks = [evaluate(a*(1+change))[1] for change in (-1e-4, -2e-5, 2e-5, 1e-4)]
    record.update(original_cell_angstrom=np.asarray(original.cell).tolist(), equilibrium_cell_angstrom=np.asarray(unit.cell).tolist(),
                  cartesian_transform_original_to_equilibrium=(np.linalg.inv(original.cell)@unit.cell).tolist(),
                  fractional_positions_unchanged=True, atom_index_mapping=list(range(len(unit))),
                  cubic_geometry="diag(a,a/sqrt(2),a/sqrt(2)); original rounding anisotropy repaired explicitly before isotropic scale optimization",
                  internal_relaxation="unnecessary: symmetry-equivalent fractional coordinates already have forces below 1e-6 eV/angstrom",
                  potential_sha256=sha256(potential), input_sha256=sha256(ROOT/"Si_110_temp300.lmp"),
                  volume_checks=volume_checks, energy_minimum_verified=all(row["energy_ev_per_atom"] > record["energy_ev_per_atom"] for row in volume_checks),
                  optimization_evaluations=evaluations, original_input_overwritten=False, benchmark2_geometry_changed=False)
    write(output/"equilibrium.extxyz", unit)
    write(output/"equilibrium.lmp", unit, format="lammps-data", atom_style="atomic", masses=True)
    json_write(cache, record)
    return unit, record


def _model_energies(model, q):
    return model.modes(q)[0]*constants.hbar/(1e-3*constants.elementary_charge)


def _paired_dataset(unit, repetitions, sigma, count, seed, potential, directory):
    reference = unit.repeat(repetitions)
    key = fingerprint({"reps": repetitions, "sigma": sigma, "count": count, "seed": seed, "potential": sha256(potential), "cell": np.asarray(unit.cell).tolist()})[:16]
    path = directory/f"paired_{key}.npz"
    if path.exists():
        with np.load(path) as values:
            return reference, {name: values[name] for name in values.files}
    rng = np.random.default_rng(seed)
    displacements = rng.normal(scale=sigma, size=(count, len(reference), 3))
    displacements -= np.mean(displacements, axis=1, keepdims=True)
    plus, minus = [], []
    with ModifiedTersoffForces(reference, potential) as calculator:
        for displacement in displacements:
            plus.append(calculator.evaluate(reference.positions+displacement)["forces_ev_per_angstrom"])
            minus.append(calculator.evaluate(reference.positions-displacement)["forces_ev_per_angstrom"])
    values = {"displacements_angstrom": displacements, "forces_plus_ev_per_angstrom": np.array(plus),
              "forces_minus_ev_per_angstrom": np.array(minus), "odd_forces_ev_per_angstrom": (np.array(plus)-np.array(minus))/2,
              "even_force_residual_ev_per_angstrom": (np.array(plus)+np.array(minus))/2,
              "reference_positions_angstrom": reference.positions, "cell_angstrom": np.asarray(reference.cell)}
    np.savez_compressed(path, **values)
    return reference, values


def _fit(cs, reference, dataset, count, ridge):
    from hiphive import StructureContainer, ForceConstantPotential
    container = StructureContainer(cs)
    for displacement, forces in zip(dataset["displacements_angstrom"][:count], dataset["odd_forces_ev_per_angstrom"][:count]):
        structure = reference.copy()
        structure.new_array("displacements", displacement)
        structure.new_array("forces", forces)
        container.add_structure(structure)
    matrix, target = container.get_fit_data()
    scale = np.linalg.norm(matrix, axis=0)
    normalized = matrix/scale
    if ridge:
        coefficients = np.linalg.solve(normalized.T@normalized + ridge*np.eye(len(scale)), normalized.T@target)
    else:
        coefficients = np.linalg.lstsq(normalized, target, rcond=None)[0]
    parameters = coefficients/scale
    potential = ForceConstantPotential(cs, parameters)
    return potential, {"rank": int(np.linalg.matrix_rank(normalized)), "parameters": len(parameters),
                       "condition_number_after_column_scaling": float(np.linalg.cond(normalized)),
                       "training_odd_relative_force_rms": float(np.linalg.norm(matrix@parameters-target)/np.linalg.norm(target)),
                       "regression": "column-scaled least squares" if not ridge else "column-scaled ridge", "ridge": ridge,
                       "training_independent_pairs": count, "training_MOD_force_evaluations": 2*count}


def _heldout(potential, reference, dataset, independent_fc):
    fc = potential.get_force_constants(reference).get_fc_array(2)
    displacement = dataset["displacements_angstrom"]
    predicted = -np.einsum("ijab,sjb->sia", fc, displacement)
    fd_predicted = -np.einsum("ijab,sjb->sia", independent_fc, displacement)
    exact = dataset["forces_plus_ev_per_angstrom"]
    odd = dataset["odd_forces_ev_per_angstrom"]
    rms = lambda values: float(np.sqrt(np.mean(values**2)))
    return {"heldout_force_rms_ev_per_angstrom": rms(exact), "heldout_absolute_force_rms_error_ev_per_angstrom": rms(predicted-exact),
            "heldout_relative_force_rms_error": rms(predicted-exact)/rms(exact),
            "heldout_relative_odd_force_error": rms(predicted-odd)/rms(odd),
            "independent_FD_heldout_relative_force_error": rms(fd_predicted-exact)/rms(exact),
            "fitted_vs_independent_harmonic_force_relative_error": rms(predicted-fd_predicted)/rms(fd_predicted),
            "even_anharmonic_force_relative_size": rms(dataset["even_force_residual_ev_per_angstrom"])/rms(exact)}, fc


def _phonopy_validation(unit, repetitions, fc, model, q, output):
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.file_IO import write_force_constants_to_hdf5
    from phonopy.structure.atoms import PhonopyAtoms
    from scipy.spatial import cKDTree
    atoms = PhonopyAtoms(symbols=unit.get_chemical_symbols(), masses=unit.get_masses(), cell=np.asarray(unit.cell), scaled_positions=unit.get_scaled_positions())
    phonon = Phonopy(atoms, np.diag(repetitions), primitive_matrix=np.eye(3), lang="C", hermitianize_dynamical_matrix=False)
    ase_sc = unit.repeat(repetitions)
    tree = cKDTree(ase_sc.get_scaled_positions()%1, boxsize=1)
    distances, permutation = tree.query(phonon.supercell.scaled_positions%1)
    if np.max(distances) > 1e-8 or len(set(permutation)) != len(permutation):
        raise ValueError("ASE-to-Phonopy supercell atom mapping failed")
    reordered = fc[permutation][:, permutation]
    native_factor = phonon.unit_conversion_factor
    si_factor = np.sqrt(constants.elementary_charge/(1e-20*constants.atomic_mass))/(2*np.pi*1e12)
    phonon.unit_conversion_factor = si_factor
    phonon.force_constants = reordered
    points = phonon.run_qpoints(q, with_eigenvectors=True, with_dynamical_matrices=True)
    energies = points.frequencies*1e12*constants.h/(1e-3*constants.elementary_charge)
    ours, vectors = model.modes(q)
    ours = ours*constants.hbar/(1e-3*constants.elementary_charge)
    projector_max = 0.0
    for values, first, second in zip(ours, vectors, points.eigenvectors):
        begin = 0
        while begin < len(values):
            end = begin+1
            while end < len(values) and abs(values[end]-values[begin]) < 1e-3:
                end += 1
            projector_max = max(projector_max, float(np.linalg.norm(first[:, begin:end]@first[:, begin:end].conj().T-second[:, begin:end]@second[:, begin:end].conj().T)))
            begin = end
    write_force_constants_to_hdf5(reordered, filename=output/"force_constants_phonopy.hdf5")
    phonon.save(output/"phonopy.yaml", settings={"force_constants": False})
    primitive = Phonopy(atoms, np.diag(repetitions), primitive_matrix="auto", lang="C")
    primitive.unit_conversion_factor = si_factor
    primitive.force_constants = reordered
    primitive_vectors = np.asarray(primitive.primitive.cell)
    supermatrix = np.asarray(unit.cell)@np.linalg.inv(primitive_vectors)
    candidates = np.eye(3)@np.linalg.inv(np.asarray(unit.cell)).T@primitive_vectors.T
    folded = next(value for value in candidates if not np.allclose(value-np.rint(value), 0))
    primitive_points = primitive.run_qpoints([[0, 0, 0], folded.tolist()], with_eigenvectors=True)
    primitive_energies = primitive_points.frequencies*1e12*constants.h/(1e-3*constants.elementary_charge)
    gamma_oriented = _model_energies(model, [0, 0, 0])
    primitive_positions = primitive.primitive.scaled_positions @ primitive_vectors
    oriented_basis, oriented_translations = [], []
    for position in unit.positions:
        fractional = (position[None]-primitive_positions) @ np.linalg.inv(primitive_vectors)
        residual = np.linalg.norm(fractional-np.rint(fractional), axis=1)
        basis = int(np.argmin(residual))
        if residual[basis] > 1e-7:
            raise ValueError('oriented-to-primitive atom mapping failed')
        oriented_basis.append(basis)
        oriented_translations.append(np.rint(fractional[basis]).astype(int).tolist())
    return {"maximum_energy_difference_mev": float(np.max(np.abs(ours-energies))), "maximum_degenerate_projector_error": projector_max,
            "phonopy_native_THZ_factor": native_factor, "project_constants_THZ_factor": si_factor,
            "frequency_convention": "Phonopy frequencies are ordinary THz; energy=h*f, angular omega=2*pi*f",
            "primitive_cell_angstrom": primitive_vectors.tolist(), "primitive_atom_count": len(primitive.primitive), "oriented_atom_count": len(unit),
            "primitive_to_oriented_integer_matrix": supermatrix.tolist(), "primitive_folded_q_fractional": folded.tolist(),
            "folded_six_plus_six_vs_twelve_maximum_energy_error_mev": float(np.max(abs(np.sort(primitive_energies.ravel())-gamma_oriented))),
            "ASE_to_Phonopy_atom_permutation": permutation.tolist(),
            "primitive_fractional_positions": primitive.primitive.scaled_positions.tolist(),
            "primitive_cartesian_positions_angstrom": primitive_positions.tolist(),
            "primitive_to_Phonopy_supercell_atom_mapping": primitive.primitive.p2s_map.tolist(),
            "oriented_atom_to_primitive_basis": oriented_basis,
            "oriented_atom_primitive_lattice_translation": oriented_translations}


def run_benchmark(config: LatticeConfig, output):
    from ase.io import write
    from hiphive import ClusterSpace
    output = Path(output).resolve();output.mkdir(parents=True, exist_ok=True)
    for name in ("cache", "training", "models"):
        (output/name).mkdir(exist_ok=True)
    limits = ResourceLimits(config.runtime_limit_seconds, config.memory_limit_gib, 100)
    original, original_verification = load_structure(ROOT/"Si_110_temp300.lmp")
    potential_path = ROOT/"Si.tersoff.mod"
    prior = previous_statuses()
    unit, equilibrium = equilibrium_structure(original, potential_path, output)
    constants_data = constants_record()
    for name in ("Boltzmann constant",):
        value, units, uncertainty = constants.physical_constants[name]
        constants_data["values"][name] = {"value": value, "units": units, "standard_uncertainty": uncertainty}
    results = {"benchmark": "Benchmark 5: Si MOD harmonic lattice dynamics with hiPhive validation", "status": "RUNNING", "parameters": asdict(config),
               "environment": environment(), "constants": constants_data, "original_verification": original_verification, "equilibrium": equilibrium,
               "potential": {"path": str(potential_path), "sha256": sha256(potential_path), **potential_metadata(potential_path), "units": "LAMMPS metal",
                             "citation": "Kumagai, Izumi, Hara and Sakai, Computational Materials Science 39, 457 (2007)", "ordinary_tersoff_substitution": False},
               "prior_benchmark_statuses_preserved": prior, "benchmark_3B_status": "NOT_AVAILABLE; no saved status changed", "reference_study": [], "fit_study": [], "covariance_study": []}
    json_write(output/"checkpoint.json", results)
    rng = np.random.default_rng(34017)
    test_q = np.vstack(([0, 0, 0], rng.uniform(-0.5, 0.5, (30, 3)), [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]))
    reference_models = {}
    selected_full = None
    for repetitions in config.reference_supercells:
        for displacement in config.displacement_angstrom:
            limits.check(estimated_bytes=(np.prod(repetitions)*4*3)**2*8*6)
            name = "reference_"+"x".join(map(str, repetitions))+f"_d{displacement:g}"
            cache = output/"cache"/f"{name}.npz"
            if cache.exists():
                with np.load(cache) as data:
                    raw = data["raw_ev_per_angstrom2"];corrected = data["corrected_ev_per_angstrom2"]
            else:
                columns, _, _ = central_force_constants(unit, repetitions, displacement, potential_path)
                raw = complete_force_constants(columns, repetitions);corrected = correct_sum_rules(raw)
                np.savez_compressed(cache, raw_ev_per_angstrom2=raw, corrected_ev_per_angstrom2=corrected)
            raw_model, raw_range = real_space_from_supercell(unit, repetitions, raw)
            model, range_record = real_space_from_supercell(unit, repetitions, corrected)
            energies = _model_energies(model, test_q)
            raw_dynamical = raw_model.dynamical(test_q)
            raw_values = np.linalg.eigvalsh((raw_dynamical+raw_dynamical.swapaxes(-1,-2).conj())/2)
            raw_energies = np.sign(raw_values)*np.sqrt(abs(raw_values))*constants.hbar/(1e-3*constants.elementary_charge)
            row = {"supercell": repetitions, "atoms": len(raw), "displacement_angstrom": displacement, "raw_residuals": force_constant_residuals(raw),
                   "corrected_residuals": force_constant_residuals(corrected), "relative_sum_rule_and_permutation_correction": float(np.linalg.norm(corrected-raw)/np.linalg.norm(raw)),
                   "maximum_raw_to_corrected_energy_change_mev": float(np.max(abs(raw_energies-energies))), "range": range_record,
                   "raw_frequency_diagnostic": "spectrum of explicitly Hermitianized raw dynamical matrix; raw non-Hermitian matrix is not treated as an exact Hamiltonian",
                   "raw_dynamical_hermiticity_residual": float(np.max(np.linalg.norm(raw_dynamical-raw_dynamical.swapaxes(-1,-2).conj(),axis=(-1,-2))/np.linalg.norm(raw_dynamical,axis=(-1,-2)))),
                   "cache": str(cache.relative_to(output)), "independent_MOD_force_calls": 24}
            reference_models[(tuple(repetitions), displacement)] = model
            results["reference_study"].append(row)
            if repetitions == config.reference_supercells[-1] and displacement == config.displacement_angstrom[-1]:
                selected_full = corrected
                reference = model
                reference.save(output/"reference_model.npz")
                np.savez_compressed(output/"force_constants_reference.npz", raw_ev_per_angstrom2=raw, corrected_ev_per_angstrom2=corrected, supercell_repetitions=repetitions)
            json_write(output/"checkpoint.json", results)
    reference_energies = _model_energies(reference, test_q)
    for row in results["reference_study"]:
        model = reference_models[(tuple(row["supercell"]), row["displacement_angstrom"])]
        row["maximum_energy_difference_from_finest_mev"] = float(np.max(abs(_model_energies(model, test_q)-reference_energies)))
    selected_repetitions = tuple(config.training_supercells[-1])
    max_pairs = max(config.training_pair_counts)
    setups = [(selected_repetitions, config.selected_sigma_angstrom, config.selected_cutoff_angstrom, count, 0.0, "training_size") for count in config.training_pair_counts]
    setups += [(selected_repetitions, sigma, config.selected_cutoff_angstrom, max_pairs, 0.0, "displacement") for sigma in config.training_sigmas_angstrom]
    setups += [(selected_repetitions, config.selected_sigma_angstrom, cutoff, max_pairs, 0.0, "second_order_cutoff") for cutoff in config.second_order_cutoffs_angstrom]
    setups += [(tuple(reps), config.selected_sigma_angstrom, config.selected_cutoff_angstrom, max_pairs, 0.0, "training_supercell") for reps in config.training_supercells]
    setups += [(selected_repetitions, config.selected_sigma_angstrom, config.selected_cutoff_angstrom, max_pairs, ridge, "regression") for ridge in config.regression_ridges]
    cluster_spaces, fitted = {}, {}
    independent_training_fcs = {tuple(config.reference_supercells[-1]): selected_full}
    log = io.StringIO()
    selected_potential = None
    for repetitions, sigma, cutoff, count, ridge, category in setups:
        limits.check()
        cache_key = (repetitions, sigma, cutoff, count, ridge)
        if cache_key in fitted:
            row, fcp, fitted_model = fitted[cache_key]
            row = {**row, "category": category}
        else:
            if cutoff not in cluster_spaces:
                with contextlib.redirect_stdout(log):
                    cluster_spaces[cutoff] = ClusterSpace(unit, [cutoff])
            training_reference, training = _paired_dataset(unit, repetitions, sigma, max_pairs, config.training_seed, potential_path, output/"training")
            validation_reference, validation = _paired_dataset(unit, repetitions, config.selected_sigma_angstrom, config.validation_structures, config.validation_seed, potential_path, output/"training")
            with contextlib.redirect_stdout(log):
                fcp, fitting = _fit(cluster_spaces[cutoff], training_reference, training, count, ridge)
            if repetitions not in independent_training_fcs:
                columns, _, _ = central_force_constants(unit, repetitions, config.displacement_angstrom[-1], potential_path)
                independent_training_fcs[repetitions] = correct_sum_rules(complete_force_constants(columns, repetitions))
            independent_fc = independent_training_fcs[repetitions]
            validation_record, fit_fc_validation = _heldout(fcp, validation_reference, validation, independent_fc)
            fit_fc = fcp.get_force_constants(unit.repeat(config.reference_supercells[-1])).get_fc_array(2)
            fitted_model, fitted_range = real_space_from_supercell(unit, config.reference_supercells[-1], fit_fc)
            energy_error = float(np.max(abs(_model_energies(fitted_model, test_q)-reference_energies)))
            row = {"category": category, "training_supercell": repetitions, "sigma_angstrom": sigma, "second_order_cutoff_angstrom": cutoff,
                   "hiPhive_expansion_order_cutoff_list": [cutoff], **fitting, **validation_record,
                   "relative_force_constant_error": float(np.linalg.norm(fit_fc-selected_full)/np.linalg.norm(selected_full)),
                   "maximum_phonon_energy_error_mev": energy_error, "fit_force_constant_residuals": force_constant_residuals(fit_fc), "range": fitted_range,
                   "status": "PASS" if validation_record["heldout_relative_force_rms_error"] < config.force_relative_tolerance and energy_error < config.phonon_energy_tolerance_mev and np.linalg.norm(fit_fc-selected_full)/np.linalg.norm(selected_full) < config.force_constant_relative_tolerance else "DIAGNOSTIC_UNDERCONVERGED"}
            fitted[cache_key] = row, fcp, fitted_model
        results["fit_study"].append(row)
        if repetitions == selected_repetitions and sigma == config.selected_sigma_angstrom and cutoff == config.selected_cutoff_angstrom and count == max_pairs and ridge == 0:
            selected_potential, selected_model, selected_fit = fcp, fitted_model, row
        json_write(output/"checkpoint.json", results)
    (output/"hiphive_setup.log").write_text(log.getvalue())
    selected_potential.write(str(output/"harmonic_model.fcp"))
    selected_model.save(output/"hiphive_model.npz")
    final_fc = selected_potential.get_force_constants(unit.repeat(config.reference_supercells[-1])).get_fc_array(2)
    results["selected_fit"] = selected_fit
    comparisons = {
        "training_size": ((selected_repetitions, config.selected_sigma_angstrom, config.selected_cutoff_angstrom, config.training_pair_counts[-2], 0.0), (selected_repetitions, config.selected_sigma_angstrom, config.selected_cutoff_angstrom, max_pairs, 0.0)),
        "training_displacement": ((selected_repetitions, config.training_sigmas_angstrom[-2], config.selected_cutoff_angstrom, max_pairs, 0.0), (selected_repetitions, config.training_sigmas_angstrom[-1], config.selected_cutoff_angstrom, max_pairs, 0.0)),
        "interaction_radius": ((selected_repetitions, config.selected_sigma_angstrom, config.second_order_cutoffs_angstrom[-2], max_pairs, 0.0), (selected_repetitions, config.selected_sigma_angstrom, config.second_order_cutoffs_angstrom[-1], max_pairs, 0.0)),
        "training_supercell": ((tuple(config.training_supercells[-2]), config.selected_sigma_angstrom, config.selected_cutoff_angstrom, max_pairs, 0.0), (selected_repetitions, config.selected_sigma_angstrom, config.selected_cutoff_angstrom, max_pairs, 0.0)),
        "regression": ((selected_repetitions, config.selected_sigma_angstrom, config.selected_cutoff_angstrom, max_pairs, config.regression_ridges[0]), (selected_repetitions, config.selected_sigma_angstrom, config.selected_cutoff_angstrom, max_pairs, config.regression_ridges[-1])),
    }
    results["fit_parameter_refinement_energy_changes_mev"] = {name: float(np.max(abs(_model_energies(fitted[first][2], test_q)-_model_energies(fitted[second][2], test_q)))) for name, (first, second) in comparisons.items()}
    dense_q = np.array(list(np.ndindex(12, 12, 12)), dtype=float)/12
    results["dense_12cubed_maximum_phonon_energy_difference_mev"] = float(np.max(abs(_model_energies(selected_model, dense_q)-_model_energies(reference, dense_q))))
    results["phonopy_validation"] = _phonopy_validation(unit, config.reference_supercells[-1], final_fc, selected_model, test_q, output)
    dynamical = selected_model.dynamical(test_q)
    results["maximum_dynamical_hermiticity_residual"] = float(np.max(np.linalg.norm(dynamical-dynamical.swapaxes(-1,-2).conj(), axis=(-1,-2))/np.linalg.norm(dynamical, axis=(-1,-2))))
    gamma = _model_energies(selected_model, [0, 0, 0])
    results["gamma_energies_mev"] = gamma.tolist()
    results["minimum_nonGamma_energy_mev"] = float(np.min(_model_energies(selected_model, test_q[1:])))
    covariance_previous = None
    final_covariance = None
    for mesh in config.covariance_meshes:
        limits.check()
        path = output/"cache"/f"covariance_mesh{mesh}.npz"
        if path.exists():
            with np.load(path) as data:
                covariance = data["covariance_m2"];diagnostic = json.loads(str(data["diagnostic_json"]))
        else:
            covariance, diagnostic = covariance_mesh(selected_model, mesh)
            np.savez_compressed(path, covariance_m2=covariance, diagnostic_json=json.dumps(diagnostic))
        row = {**diagnostic, "covariance_m2": covariance.tolist(), "relative_change_from_previous": None if covariance_previous is None else float(np.max(np.linalg.norm(covariance-covariance_previous, axis=(-1,-2))/np.linalg.norm(covariance, axis=(-1,-2))))}
        results["covariance_study"].append(row)
        covariance_previous, final_covariance = covariance, covariance
        json_write(output/"checkpoint.json", results)
    np.savez_compressed(output/"covariances.npz", temperatures_k=[0, 300], covariance_m2=final_covariance, q_mesh=[config.covariance_meshes[-1]]*3)
    results["debye_waller_examples"] = []
    for q in ([1e10, 0, 0], [0, 1e10, 0], [0, 0, 1e10]):
        results["debye_waller_examples"].append({"angular_Q_m_inv": q, "W_thermal_by_atom": (np.einsum("a,tiab,b->ti", q, final_covariance, q)/2).tolist()})
    final_disp_changes = []
    for a, b in zip(config.displacement_angstrom[:-1], config.displacement_angstrom[1:]):
        first, second = reference_models[(tuple(config.reference_supercells[-1]), a)], reference_models[(tuple(config.reference_supercells[-1]), b)]
        final_disp_changes.append(float(np.max(abs(_model_energies(first, test_q)-_model_energies(second, test_q)))))
    supercell_change = float(np.max(abs(_model_energies(reference_models[(tuple(config.reference_supercells[-2]), config.displacement_angstrom[-1])], test_q)-reference_energies)))
    results["displacement_refinement_energy_changes_mev"] = final_disp_changes
    results["last_supercell_refinement_energy_change_mev"] = supercell_change
    results["checks"] = {
        "equilibrium_forces": equilibrium["maximum_force_ev_per_angstrom"] < 1e-6,
        "equilibrium_stress_and_volume": max(abs(value) for value in equilibrium["pressure_bar_voigt"][:3]) < 1e-2 and equilibrium["energy_minimum_verified"],
        "heldout_forces": selected_fit["heldout_relative_force_rms_error"] < config.force_relative_tolerance,
        "force_constants": selected_fit["relative_force_constant_error"] < config.force_constant_relative_tolerance,
        "phonon_model_agreement": max(selected_fit["maximum_phonon_energy_error_mev"], results["dense_12cubed_maximum_phonon_energy_difference_mev"]) < config.phonon_energy_tolerance_mev,
        "training_and_regression_refinement": all(change < config.phonon_energy_tolerance_mev for change in results["fit_parameter_refinement_energy_changes_mev"].values()),
        "displacement_convergence": final_disp_changes[-1] < config.phonon_energy_tolerance_mev,
        "force_supercell_convergence": supercell_change < config.phonon_energy_tolerance_mev,
        "hermiticity": results["maximum_dynamical_hermiticity_residual"] < config.hermiticity_tolerance,
        "translational_sum_rule": selected_fit["fit_force_constant_residuals"]["translational"] < config.sum_rule_tolerance,
        "permutation_symmetry": selected_fit["fit_force_constant_residuals"]["permutation"] < config.hermiticity_tolerance,
        "three_Gamma_translations": bool(np.sum(abs(gamma) < config.gamma_zero_tolerance_mev) == 3),
        "physical_stability": results["minimum_nonGamma_energy_mev"] > 0,
        "phonopy_convention_and_folding": max(results["phonopy_validation"]["maximum_energy_difference_mev"], results["phonopy_validation"]["folded_six_plus_six_vs_twelve_maximum_energy_error_mev"]) < config.phonon_energy_tolerance_mev and results["phonopy_validation"]["maximum_degenerate_projector_error"] < 1e-6,
        "covariance_sampling": len(results["covariance_study"]) >= 3 and all(row["relative_change_from_previous"] < config.covariance_relative_tolerance for row in results["covariance_study"][-2:]),
    }
    results["status"] = "PASS" if all(results["checks"].values()) else "FAIL"
    results["resource_costs"] = limits.costs()
    results["conventions"] = {"cell": "row vectors along supplied x,y,z; z=[110]; exact cubic a,a/sqrt(2),a/sqrt(2)",
                              "force_constants": "Phi_iα,jβ=-dF_iα/du_jβ; LAMMPS and exported Phonopy arrays eV/angstrom^2; reusable model N/m",
                              "reciprocal": "B=2*pi*A^-T, angular inverse metres; q_fractional@B",
                              "Fourier_phase": "exp(+i q.(R+tau_j-tau_i)), Phonopy atomic gauge",
                              "eigenvectors": "sum_iα e_iα* e_iα=1 per q and branch; mass-weighted dynamical matrix",
                              "covariance": "uniform q weights 1/Nq applied once; remove only the three rigid Gamma translations",
                              "phonon_model": "bulk harmonic periodic force constants; no MD or electron propagation",
                              "training": "paired +/- random displacements; fit odd force component; held-out actual MOD forces remain independent"}
    json_write(output/"results.json", results)
    from .lattice_outputs import write_outputs
    results["figures"] = write_outputs(results, unit, reference, selected_model, output)
    json_write(output/"results.json", results)
    pv=results["phonopy_validation"]
    mapping={name:pv[name] for name in ("primitive_cell_angstrom","primitive_cartesian_positions_angstrom","oriented_atom_to_primitive_basis","oriented_atom_primitive_lattice_translation","primitive_to_Phonopy_supercell_atom_mapping")}
    reconstructed=np.array(mapping["primitive_cartesian_positions_angstrom"])[mapping["oriented_atom_to_primitive_basis"]]+np.array(mapping["oriented_atom_primitive_lattice_translation"])@np.array(mapping["primitive_cell_angstrom"])
    mapping.update(oriented_atom_positions_angstrom=unit.positions.tolist(),maximum_reconstruction_error_angstrom=float(np.max(abs(reconstructed-unit.positions))),Phonopy_supercell_repetitions=config.reference_supercells[-1])
    json_write(output/"primitive_mapping.json",mapping)
    manifest = {"schema_version": 1, "benchmark": 5, "status": results["status"], "checks": results["checks"],
                "equilibrium_input_differs_from_benchmark2": True, "force_constant_supercell_repetitions": config.reference_supercells[-1],
                "covariance_q_mesh": [config.covariance_meshes[-1]]*3, "conventions": results["conventions"], "constants": constants_data,
                "artifacts": {"equilibrium": artifact(output/"equilibrium.extxyz", output, "angstrom; masses u", "equilibrium four-atom oriented Si cell"),
                              "force_constants": artifact(output/"hiphive_model.npz", output, "m, kg, N/m", "validated bulk real-space fitted model"),
                              "independent_reference": artifact(output/"reference_model.npz", output, "m, kg, N/m", "independent central-displacement reference"),
                              "hiphive_potential": artifact(output/"harmonic_model.fcp", output, "eV/angstrom^2", "second-order hiPhive model"),
                              "phonopy_force_constants": artifact(output/"force_constants_phonopy.hdf5", output, "eV/angstrom^2", "full FC in Phonopy supercell order"),
                              "covariances": artifact(output/"covariances.npz", output, "m^2", "q-converged atom covariance tensors at 0 and 300 K"),
                              "primitive_mapping": artifact(output/"primitive_mapping.json", output, "angstrom; integer lattice translations", "explicit oriented four-atom to primitive two-atom mapping"),
                              "results": artifact(output/"results.json", output, "documented per field", "validation, training, and convergence results")}}
    json_write(output/"manifest.json", manifest)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"benchmark_results/si_lattice_dynamics")
    args = parser.parse_args(argv)
    values = {} if args.config is None else json.loads(args.config.read_text())
    for name in ("reference_supercells", "training_supercells"):
        if name in values: values[name] = tuple(tuple(row) for row in values[name])
    for name in ("displacement_angstrom", "training_pair_counts", "training_sigmas_angstrom", "second_order_cutoffs_angstrom", "regression_ridges", "covariance_meshes"):
        if name in values: values[name] = tuple(values[name])
    config = LatticeConfig(**values)
    results = run_benchmark(config, args.output)
    print(results["benchmark"], results["status"])
    print(json.dumps(results["checks"], indent=2))
    print("Results:", args.output.resolve())
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
