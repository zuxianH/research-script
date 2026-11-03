# Yoshioka EELS

`yoshioka-eels` is the reusable Python package for the Yoshioka electron energy
loss project. Install it as `yoshioka-eels`; Python code continues to use
`import yoshioka`. Version 0.2.0 separates development code from the original
research record without changing the physical operators or acceptance criteria.

| Location | Purpose |
| --- | --- |
| `yoshioka/` | Installable propagation, coupled-channel, phonon and EELS APIs |
| `benchmarks/` | Research runners and configurations, used from this checkout |
| `tests/` | Existing scientific regressions plus package portability checks |
| `examples/free_electron.py` | Small, bounded example using the installed API |
| `docs/` | Development guidance and preserved historical documentation |
| `/home/zuxian/Documents/PhD/Yoshioka` | Original source, environment, data and results |

Browse saved results through the [visual start page](benchmark_results/START_HERE.html)
or [text index](benchmark_results/README.md). The result store has curated
`Current_results`, `Plots`, `Validation` and `Archive` folders. Original run
paths and evidence remain intact; older material is outside the main browsing
view. These local links become available after connecting the research data.

## Install and develop

```bash
cd /home/zuxian/Documents/research-script/yoshioka-eels
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/yoshioka-eels
```

The core uses NumPy, SciPy, Matplotlib and psutil. Material-reference workflows
also need the original optional engines:

```bash
.venv/bin/python -m pip install -e '.[elastic-reference,lattice-reference]'
```

The retained `requirements-*.txt` describe the original scientific environment.
They are historical environment snapshots, not evidence that every supported
Python version or minimum dependency version was tested. Benchmark modules are
checkout tools and are deliberately excluded from the installed wheel.

Run the small example with an explicit output directory under `PhD`:

```bash
.venv/bin/python examples/free_electron.py \
  --output /home/zuxian/Documents/PhD/Yoshioka/package_examples/free_electron
```

It writes a wave array and numerical checks. It does not calculate a material
EELS spectrum. APIs use SI units unless their documentation states otherwise.

## Existing research data

On this machine, connect the checkout to the original data without moving or
copying results:

```bash
python3 scripts/link_research_data.py /home/zuxian/Documents/PhD/Yoshioka
```

This creates ignored local links for `benchmark_results` and the two TACAW
arrays. The links and `.local-data.json` are machine-local; they are never part
of the wheel or Git source. The small original Tersoff model and silicon input
are included. The new SNAP files in the original directory do not change this
project's potential or benchmark definition.

For research integration tests, install the optional dependencies, connect the
data, and run from this directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

Some tests require saved fitted force constants and frozen finite-model drivers.
Historical cross-project comparison runners can also require external inputs
described in their original documentation.

## Scientific status

Yoshioka is the coupled electron–target framework. The complete leading-pair
calculation expands the inelastic interaction while propagating elastic
distorted waves in the static crystal potential. The `V0=0` Born calculations
are separately labelled controls. `N65/M135` and `N97/M195` are numerical
resolutions of the same model: retained transverse beam rectangles of 65×65 or
97×97, with product grids of 135×135 or 195×195 for the projected action.

The four-repeat bulk benchmark remains **NOT_ACCEPTED**. Corrected complete-net
production count remains **zero**. Packaging and unit tests do not resolve the
retained-support or weak-window depth errors. See
[current research scope](docs/research-status.md) and the original checkpoints
under `PhD/Yoshioka/benchmark_results/14_two_phonon_eels`.

Continue development here. Resume historical hashed cases using their original
source and commands in `PhD/Yoshioka`; the copied package has new source hashes.
New calculations need a fresh output directory and a new dependency manifest.
Do not run historical production defaults over saved outputs. Preserve failed
and unresolved results, resource guards and acceptance thresholds.
