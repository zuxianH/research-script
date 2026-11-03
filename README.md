# EELS scripts

The [research workflow guide](docs/research-workflow.md) explains the shared
[tools](https://github.com/zuxianH/research-workflow) and
[knowledge](https://github.com/zuxianH/research-knowledge) repositories.
Clone them as siblings of this repository, then run `./rw status`.
Scientific BTO preparation and restart tools are in
[experiments/bto-single-domain](experiments/bto-single-domain/README.md).
The accepted portable **TRPMD / Si(110) SNAP** template, including native inputs
and model files, is in [experiments/trpmd](experiments/trpmd/README.md).
The installable **Yoshioka EELS** package is in [yoshioka-eels](yoshioka-eels/README.md).
Its original research results remain under `/home/zuxian/Documents/PhD/Yoshioka`.

Tools for preparing molecular-dynamics trajectories, running a GPU TACAW
calculation, and correcting its output.

```text
XYZ trajectories -> ASE .traj -> TACAW .npy -> Kubo-corrected .npy
```

## Layout

| Location | Purpose |
| --- | --- |
| `trajectories/convert_xyz_traj.py` | Stream regular, extended, or i-PI XYZ into an ASE trajectory. |
| `trajectories/convert_all_xyz_traj.sh` | Convert a directory of XYZ files, optionally in parallel. |
| `trajectories/visualize_traj.py` | View an XYZ trajectory with a Megane widget in Jupyter. |
| `simulation/tacaw_gpu.py` | Run the existing plane-wave GPU TACAW calculation. |
| `analysis/apply_kubo_correction.py` | Apply the Kubo detailed-balance factor to TACAW arrays. |
| `analysis/scattering_angles.py` | Provide the reusable `tacaw_theta_grid` function. |
| `notebooks/inspect_ipi_output.ipynb` | Monitor i-PI logs and analyze equilibrium cells. |
| [EELS Studio](EELS_studio/README.md) | Browser interface for EELS analysis. |
| [Torched TACAW Studio](Torched_TACAW_studio/README.md) | Browser interface for trajectory preparation and TACAW runs. |

The batch converter and Python converter must stay in the same folder.
The Studio applications retain their own dependencies and launch instructions.

## Dependencies

Use Python 3.11 or later. Choose dependencies for the tool you need:

| Tools | Dependencies |
| --- | --- |
| Trajectory conversion | NumPy, ASE; Bash for the batch wrapper. |
| GPU TACAW calculation | NumPy, ASE, SciPy, tqdm, and a GPU-compatible pyms/PyTorch installation. |
| Kubo correction and scattering angles | NumPy. |
| Trajectory viewer | Megane, IPython, and a Jupyter environment with widget support. |
| i-PI notebook | NumPy, pandas, Matplotlib, IPython, and Jupyter. |

Use your existing calculation environment for pyms and the platform-specific
PyTorch build. See the [Studio setup guide](Torched_TACAW_studio/docs/setup.md)
for the existing laptop and cluster setup notes.

## Commands

These examples assume the current directory is this repository root.
Replace the example data paths and calculation settings with your own.
All Python tools also accept absolute script paths from another directory.

### Convert trajectories

```bash
python3 trajectories/convert_xyz_traj.py /path/to/run/simulation.pos_0.xyz
bash trajectories/convert_all_xyz_traj.sh --dry-run /path/to/run
bash trajectories/convert_all_xyz_traj.sh -j 4 /path/to/run
```

Outputs are written beside the inputs. Existing outputs are skipped by the
batch converter. Use `--overwrite` to replace them deliberately.
The single-file converter also supports `--start`, `--stop`, and `--stride`.

### Run TACAW

```bash
python3 simulation/tacaw_gpu.py 0 1 4 10 128 64 \
    --input-dir /path/to/run \
    --output /path/to/run/TACAW.npy
```

The positional arguments are GPU index, total GPU count, CPU workers, bead
count, chunk length, and chunk offset, in that order.
The total GPU count is logged; it does not distribute work across devices.
Inputs must be named `simulation.pos_0.traj`, `simulation.pos_1.traj`, and so on.
The input directory defaults to the current directory. Without `--output`, the
original generated filename is used inside the input directory.

Grid dimensions, voltage, slice count, sampling stride, and thermalization
cutoff remain in the parameter block inside `main()`. The original numerical
calculation is retained. The generated default filename contains fixed material
and temperature labels; use `--output` to name results for another system.

### Apply the Kubo correction

```bash
python3 analysis/apply_kubo_correction.py /path/to/run/TACAW.npy \
    --temperature 1000 --timestep-fs 5 --stride 3
```

Pass one or more input files explicitly. The example settings must match the
source MD trajectory and TACAW sampling. The correction defaults remain 10 K,
2.5 fs, and stride 3; they are not inferred from the simulation script.
If conversion also subsampled the trajectory, include that factor in the
sampling interval supplied to the correction.

Inputs must have shape `(energy, y, x)` with an FFT-shifted energy axis.
Each output is saved beside its input with the suffix `_convKubo.npy`.
Existing corrected outputs are never overwritten.

### Inspect trajectories and MD logs

Run this in a Jupyter notebook opened in the repository root:

```python
%run trajectories/visualize_traj.py /path/to/run/simulation.pos_0.xyz --frame 50
```

The viewer is a notebook widget. Running the file in a plain terminal does not
provide an interactive display. Choose a frame that exists in your trajectory.

Open [the i-PI notebook](notebooks/inspect_ipi_output.ipynb) to inspect simulation
logs and equilibrium structures. Edit the input paths and system parameters in
the selected cells. The monitoring cells run continuously until interrupted.
The notebook retains its existing cells and outputs.

For angle coordinates in a Python session started in the repository root:

```python
from analysis.scattering_angles import tacaw_theta_grid

theta_x, theta_y, Theta_x, Theta_y, Theta = tacaw_theta_grid(
    V=60000, Lx=39.09, Ly=39.10, theta_max=100
)
```

## Data and reproducibility

Keep large trajectories, result arrays, logs, and exported figures in your
simulation run directories, outside this source tree.
The root `.gitignore` excludes common trajectory and array formats, Python
caches, notebook checkpoints, and virtual environments.

Keep a record of temperature, source timestep, conversion stride, TACAW stride,
voltage, grid dimensions, bead count, and the exact launch command with each
run. Copy the simulation script into the run folder before editing its
parameters when you need a frozen record. For example:

```bash
cp simulation/tacaw_gpu.py /path/to/run/tacaw_gpu.py
python3 /path/to/run/tacaw_gpu.py 0 1 4 10 128 64 \
    --input-dir /path/to/run --output /path/to/run/TACAW.npy
```

## Renamed paths

| Previous path | Current path |
| --- | --- |
| `convert_xyz_traj.py` | `trajectories/convert_xyz_traj.py` |
| `convert_all_xyz_traj.sh` | `trajectories/convert_all_xyz_traj.sh` |
| `visualize_traj.py` | `trajectories/visualize_traj.py` |
| `tacaw_gpu.py` | `simulation/tacaw_gpu.py` |
| `apply_convKubo.py` | `analysis/apply_kubo_correction.py` |
| `find_theta.py` | `analysis/scattering_angles.py` |
| `plot_out.ipynb` | `notebooks/inspect_ipi_output.ipynb` |

Update external job scripts and notebook imports to use these paths.
