#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Force i-PI's (and this script's Python helper's) stdout to be unbuffered,
# so ipi.log has real content if the process is killed early instead of
# sitting in a block-buffered pipe that's lost on SIGTERM.
export PYTHONUNBUFFERED=1

# === Configurable parameters ===
DEFAULT_REPLICATE="2 2 2"
DEFAULT_TEMPERATURE="300"
TIMESTEP="${TIMESTEP:-0.0005}"          # ps in LAMMPS metal units = 0.5 fs
IPI_TIMESTEP_FS="${IPI_TIMESTEP_FS:-0.5}"
RUNSTEPS="${RUNSTEPS:-100}"
NUM_BEADS="${NUM_BEADS:-1}"
NUM_CLIENT="${NUM_CLIENT:-$NUM_BEADS}"
TEMPERATURE="${TEMPERATURE:-$DEFAULT_TEMPERATURE}"
REPLICATE="${REPLICATE:-$DEFAULT_REPLICATE}"
THERMO_TAU_FS="${THERMO_TAU_FS:-1000000}" # weak centroid coupling: 1 ns

# === Executables ===
SCIENCE_ACTIVATE="${SCIENCE_ACTIVATE:-/dev/null}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/../.venv/bin/python}"
LMP_BIN="${LMP_BIN:-$ROOT/../.venv/bin/lmp}"
IPI_BIN="${IPI_BIN:-$ROOT/../.venv/bin/i-pi}"

# === Input files ===
BASE_LMP_FILE="$ROOT/inputs/lmp_Si_ipi.in"
BASE_XML_FILE="$ROOT/inputs/input.xml"
BASE_DATA_FILE="$ROOT/inputs/Si_110.lmp"
INIT_XYZ_SOURCE="${INIT_XYZ_SOURCE:-}"
SNAP_COEFF_FILE="$ROOT/models/SNAPotential.snapcoeff"
SNAP_PARAM_FILE="$ROOT/models/SNAPotential.snapparam"

if [[ -f "$SCIENCE_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$SCIENCE_ACTIVATE"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$LMP_BIN" ]]; then
  echo "LAMMPS executable not found: $LMP_BIN" >&2
  exit 1
fi

if [[ ! -x "$IPI_BIN" ]]; then
  echo "i-PI executable not found: $IPI_BIN" >&2
  exit 1
fi

for required_file in \
  "$BASE_LMP_FILE" \
  "$BASE_XML_FILE" \
  "$BASE_DATA_FILE" \
  "$SNAP_COEFF_FILE" \
  "$SNAP_PARAM_FILE" \
  "$SCRIPT_DIR/organize_run.py" \
  "$SCRIPT_DIR/monitor_run.sh"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required input file not found: $required_file" >&2
    exit 1
  fi
done

VENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
for lib_dir in "$VENV_PREFIX/lib/mpich" "$VENV_PREFIX/lib"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done

read -r REPL_X REPL_Y REPL_Z <<< "$REPLICATE"
REPL_TAG="${REPL_X}x${REPL_Y}x${REPL_Z}"
RUN_TAG="Si_110_rep${REPL_TAG}_T${TEMPERATURE}K_beads${NUM_BEADS}_steps${RUNSTEPS}"
RUN_NAME="${RUN_NAME:-${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/results/$RUN_NAME}"
SOCKET_ADDRESS="${SOCKET_ADDRESS:-si110_${REPL_TAG}_b${NUM_BEADS}_s${RUNSTEPS}_$$}"
SOCKET_FILE="/tmp/ipi_${SOCKET_ADDRESS}"

BASE_DATA_COPY="Si_110_base.lmp"
DATA_FILE="Si_110_${REPL_TAG}.lmp"
if [[ -n "$INIT_XYZ_SOURCE" ]]; then
  if [[ ! -f "$INIT_XYZ_SOURCE" ]]; then
    echo "INIT_XYZ_SOURCE does not point to a readable file: $INIT_XYZ_SOURCE" >&2
    exit 1
  fi
  INIT_XYZ_FILE="$(basename "$INIT_XYZ_SOURCE")"
  GENERATED_INIT_XYZ_FILE=".generated_init_${REPL_TAG}.xyz"
else
  INIT_XYZ_FILE="init_${REPL_TAG}.xyz"
  GENERATED_INIT_XYZ_FILE="$INIT_XYZ_FILE"
fi
WORK_LMP_FILE="$RUN_DIR/.run_lmp_Si_ipi.in"
WORK_XML_FILE="$RUN_DIR/.run_input.xml"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Set RUN_NAME or RUN_DIR to a new value, or move/remove the old folder." >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cp "$BASE_LMP_FILE" "$RUN_DIR/lmp_Si_ipi.in"
cp "$BASE_XML_FILE" "$RUN_DIR/input.xml"
cp "$BASE_DATA_FILE" "$RUN_DIR/$BASE_DATA_COPY"
cp "$SNAP_COEFF_FILE" "$RUN_DIR/SNAPotential.snapcoeff"
cp "$SNAP_PARAM_FILE" "$RUN_DIR/SNAPotential.snapparam"
cp "$SCRIPT_DIR/monitor_run.sh" "$RUN_DIR/monitor_run.sh"
cp "$SCRIPT_DIR/run.sh" "$RUN_DIR/run_source.sh"

"$PYTHON_BIN" - "$RUN_DIR/$BASE_DATA_COPY" "$RUN_DIR/$DATA_FILE" "$RUN_DIR/$GENERATED_INIT_XYZ_FILE" "$REPL_X" "$REPL_Y" "$REPL_Z" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
lammps_data_out = Path(sys.argv[2])
xyz_path = Path(sys.argv[3])
nx, ny, nz = (int(value) for value in sys.argv[4:7])

lines = data_path.read_text(encoding="utf-8").splitlines()

box = {}
for line in lines:
    fields = line.split()
    if len(fields) >= 4 and fields[-2:] == ["xlo", "xhi"]:
        box["x"] = (float(fields[0]), float(fields[1]))
    elif len(fields) >= 4 and fields[-2:] == ["ylo", "yhi"]:
        box["y"] = (float(fields[0]), float(fields[1]))
    elif len(fields) >= 4 and fields[-2:] == ["zlo", "zhi"]:
        box["z"] = (float(fields[0]), float(fields[1]))

for axis in ("x", "y", "z"):
    if axis not in box:
        raise ValueError(f"Missing {axis} box bounds in {data_path}")

masses = {}
type_symbols = {}
in_masses = False
for line in lines:
    stripped = line.strip()
    if stripped == "Masses":
        in_masses = True
        continue
    if in_masses and stripped.startswith("Atoms"):
        break
    if in_masses and stripped:
        match = re.match(r"(\d+)\s+(\S+)(?:\s+#\s*(\S+))?", stripped)
        if match:
            atom_type = int(match.group(1))
            mass = float(match.group(2))
            symbol = match.group(3) or f"X{atom_type}"
            masses[atom_type] = (mass, symbol)
            type_symbols[atom_type] = symbol

atoms = []
in_atoms = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("Atoms"):
        in_atoms = True
        continue
    if in_atoms:
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 5:
            continue
        atom_type = int(fields[1])
        x, y, z = (float(value) for value in fields[2:5])
        atoms.append((atom_type, type_symbols.get(atom_type, f"X{atom_type}"), x, y, z))

if not atoms:
    raise ValueError(f"No Atoms section found in {data_path}")

lx = box["x"][1] - box["x"][0]
ly = box["y"][1] - box["y"][0]
lz = box["z"][1] - box["z"][0]
xlo, ylo, zlo = box["x"][0], box["y"][0], box["z"][0]

replicated = []
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            dx = ix * lx
            dy = iy * ly
            dz = iz * lz
            for atom_type, symbol, x, y, z in atoms:
                replicated.append((atom_type, symbol, x + dx, y + dy, z + dz))

with lammps_data_out.open("w", encoding="utf-8") as handle:
    handle.write("# Si data generated for i-PI/LAMMPS TRPMD\n\n")
    handle.write(f"{len(replicated)} atoms\n")
    handle.write(f"{len(masses)} atom types\n\n")
    handle.write(f"{xlo:16.12f} {xlo + nx * lx:16.12f} xlo xhi\n")
    handle.write(f"{ylo:16.12f} {ylo + ny * ly:16.12f} ylo yhi\n")
    handle.write(f"{zlo:16.12f} {zlo + nz * lz:16.12f} zlo zhi\n\n")
    handle.write("Masses\n\n")
    for atom_type in sorted(masses):
        mass, symbol = masses[atom_type]
        handle.write(f"{atom_type:6d} {mass:14.8f} # {symbol}\n")
    handle.write("\nAtoms # atomic\n\n")
    for atom_id, (atom_type, _symbol, x, y, z) in enumerate(replicated, start=1):
        handle.write(f"{atom_id:8d} {atom_type:4d} {x:16.8f} {y:16.8f} {z:16.8f}\n")

with xyz_path.open("w", encoding="utf-8") as handle:
    handle.write(f"{len(replicated)}\n")
    handle.write(
        "# CELL(abcABC): "
        f"{nx * lx:.12g} {ny * ly:.12g} {nz * lz:.12g} "
        "90.0 90.0 90.0 positions{angstrom} cell{angstrom}\n"
    )
    for _atom_type, symbol, x, y, z in replicated:
        handle.write(f"{symbol} {x:.12g} {y:.12g} {z:.12g}\n")

print(f"Wrote {lammps_data_out} with {len(replicated)} atoms")
print(f"Wrote {xyz_path} with {len(replicated)} atoms")
PY

if [[ -n "$INIT_XYZ_SOURCE" ]]; then
  cp "$INIT_XYZ_SOURCE" "$RUN_DIR/$INIT_XYZ_FILE"
fi

cp "$RUN_DIR/lmp_Si_ipi.in" "$WORK_LMP_FILE"
cp "$RUN_DIR/input.xml" "$WORK_XML_FILE"

sed -i "s|^read_data[[:space:]].*|read_data       ${DATA_FILE}|" "$WORK_LMP_FILE"
sed -i "s|^replicate[[:space:]].*|# replicate       ${REPL_X} ${REPL_Y} ${REPL_Z} already applied in ${DATA_FILE}|" "$WORK_LMP_FILE"
sed -i "s|^timestep[[:space:]].*|timestep        ${TIMESTEP}|" "$WORK_LMP_FILE"
sed -i "s|^fix[[:space:]].*ipi.*|fix             ipi_client all ipi ${SOCKET_ADDRESS} 32345 unix|" "$WORK_LMP_FILE"

# Clients pull work dynamically, so one can serve more than the average.
# Budget enough force calls even if one client serves every bead; i-PI owns
# the actual stopping condition and the supervisor stops remaining clients.
LMP_RUN_STEPS=$(( (RUNSTEPS + 1) * NUM_BEADS ))
sed -i "s|^run[[:space:]].*|run             ${LMP_RUN_STEPS}|" "$WORK_LMP_FILE"

sed -i "s|<initialize nbeads='[0-9]\\+'>|<initialize nbeads='${NUM_BEADS}'>|" "$WORK_XML_FILE"
sed -i "s|<file mode='xyz'[^>]*>.*</file>|<file mode='xyz' units='angstrom' cell_units='angstrom'> ${INIT_XYZ_FILE} </file>|" "$WORK_XML_FILE"
sed -i "s|<velocities mode='thermal' units='kelvin'>.*</velocities>|<velocities mode='thermal' units='kelvin'> ${TEMPERATURE} </velocities>|" "$WORK_XML_FILE"
sed -i "s|<temperature units='kelvin'>.*</temperature>|<temperature units='kelvin'> ${TEMPERATURE} </temperature>|" "$WORK_XML_FILE"
sed -i "s|<total_steps>[0-9]\\+</total_steps>|<total_steps>${RUNSTEPS}</total_steps>|" "$WORK_XML_FILE"
sed -i "s|<timestep units='femtosecond'>.*</timestep>|<timestep units='femtosecond'> ${IPI_TIMESTEP_FS} </timestep>|" "$WORK_XML_FILE"
sed -i "s|<tau units='femtosecond'>.*</tau>|<tau units='femtosecond'> ${THERMO_TAU_FS} </tau>|" "$WORK_XML_FILE"
sed -i "s|<address>.*</address>|<address> ${SOCKET_ADDRESS} </address>|" "$WORK_XML_FILE"

# Keep the visible run-folder inputs runnable too. The hidden .run_* files are
# retained as exact provenance copies, but lmp_Si_ipi.in and input.xml should
# not remain stale templates.
cp "$WORK_LMP_FILE" "$RUN_DIR/lmp_Si_ipi.in"
cp "$WORK_XML_FILE" "$RUN_DIR/input.xml"

cat > "$RUN_DIR/run_metadata.env" <<EOF
RUN_NAME="${RUN_NAME}"
REPLICATE="${REPLICATE:-$DEFAULT_REPLICATE}"
REPL_TAG="${REPL_TAG}"
RUNSTEPS="${RUNSTEPS}"
NUM_BEADS="${NUM_BEADS}"
NUM_CLIENT="${NUM_CLIENT}"
TEMPERATURE="${TEMPERATURE:-$DEFAULT_TEMPERATURE}"
TIMESTEP="${TIMESTEP}"
IPI_TIMESTEP_FS="${IPI_TIMESTEP_FS}"
THERMO_TAU_FS="${THERMO_TAU_FS}"
SOCKET_ADDRESS="${SOCKET_ADDRESS}"
PYTHON_BIN="${PYTHON_BIN}"
LMP_BIN="${LMP_BIN}"
IPI_BIN="${IPI_BIN}"
EOF

cat > "$RUN_DIR/continue.sh" <<'EOF'
#!/bin/bash

set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_DIR"

if [[ -f "$RUN_DIR/run_metadata.env" ]]; then
  # shellcheck source=/dev/null
  source "$RUN_DIR/run_metadata.env"
fi

ADD_STEPS="${ADD_STEPS:-100}"
NUM_BEADS="${NUM_BEADS:-4}"
NUM_CLIENT="${NUM_CLIENT:-${NUM_BEADS}}"
TIMESTEP="${TIMESTEP:-0.0005}"
SOCKET_ADDRESS="${SOCKET_ADDRESS:-si110_continue_$$}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LMP_BIN="${LMP_BIN:-lmp}"
IPI_BIN="${IPI_BIN:-i-pi}"
RESTART_FILE="${RESTART_FILE:-RESTART}"

if [[ ! -f "$RESTART_FILE" ]]; then
  if [[ -f simulation.restart ]]; then
    RESTART_FILE="simulation.restart"
  else
    echo "No RESTART or simulation.restart file found in $RUN_DIR" >&2
    exit 1
  fi
fi

CURRENT_STEP="$("$PYTHON_BIN" - "$RESTART_FILE" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"<step>\s*([0-9]+)\s*</step>", text)
print(int(match.group(1)) if match else 0)
PY
)"

TOTAL_STEPS="${TOTAL_STEPS:-$((CURRENT_STEP + ADD_STEPS))}"
CONTINUE_XML=".continue_input.xml"
CONTINUE_LMP=".continue_lmp_Si_ipi.in"

"$PYTHON_BIN" - "$RESTART_FILE" "$CONTINUE_XML" "$TOTAL_STEPS" "$SOCKET_ADDRESS" <<'PY'
import re
import sys

src, dst, total_steps, socket_address = sys.argv[1:5]
text = open(src, encoding="utf-8").read()
text = re.sub(
    r"<address>\s*.*?\s*</address>",
    f"<address>{socket_address}</address>",
    text,
    count=1,
    flags=re.S,
)

if re.search(r"<total_steps>\s*[0-9]+\s*</total_steps>", text):
    text = re.sub(
        r"<total_steps>\s*[0-9]+\s*</total_steps>",
        f"<total_steps>{total_steps}</total_steps>",
        text,
        count=1,
    )
else:
    text = text.replace(
        "<simulation verbosity='low'>",
        f"<simulation verbosity='low'>\n   <total_steps>{total_steps}</total_steps>",
        1,
    )

open(dst, "w", encoding="utf-8").write(text)
PY

cp .run_lmp_Si_ipi.in "$CONTINUE_LMP"
sed -i "s|^fix[[:space:]].*ipi.*|fix             ipi_client all ipi ${SOCKET_ADDRESS} 32345 unix|" "$CONTINUE_LMP"
sed -i "s|^timestep[[:space:]].*|timestep        ${TIMESTEP}|" "$CONTINUE_LMP"

REMAINING_STEPS=$((TOTAL_STEPS - CURRENT_STEP))
LMP_RUN_STEPS=$(( (REMAINING_STEPS + 1) * NUM_BEADS ))
echo "Remaining i-PI steps: ${REMAINING_STEPS}; LAMMPS force-call budget: ${LMP_RUN_STEPS}"
sed -i "s|^run[[:space:]].*|run             ${LMP_RUN_STEPS}|" "$CONTINUE_LMP"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

VENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
for lib_dir in "$VENV_PREFIX/lib/mpich" "$VENV_PREFIX/lib"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done

SOCKET_FILE="/tmp/ipi_${SOCKET_ADDRESS}"
IPI_PID=""
client_pids=()
client_logs=()

cleanup() {
  local cleanup_pid watchdog_pid
  # Escalate only our own still-running children after a short grace period.
  (
    sleep 10
    for cleanup_pid in "${client_pids[@]}" "${IPI_PID}"; do
      [[ -n "$cleanup_pid" ]] && kill -KILL "$cleanup_pid" 2>/dev/null || true
    done
  ) &
  watchdog_pid=$!
  for pid in "${client_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done

  if [[ -n "${IPI_PID}" ]] && kill -0 "${IPI_PID}" 2>/dev/null; then
    kill "${IPI_PID}" 2>/dev/null || true
    wait "${IPI_PID}" 2>/dev/null || true
  fi

  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
  rm -f "$SOCKET_FILE"
}

trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
rm -f "$SOCKET_FILE"

echo "Continuing from step ${CURRENT_STEP} to ${TOTAL_STEPS}"
"$IPI_BIN" "$CONTINUE_XML" > "ipi_continue_${CURRENT_STEP}_to_${TOTAL_STEPS}.log" 2>&1 &
IPI_PID=$!

echo "Waiting for i-PI socket: $SOCKET_FILE"
for _ in $(seq 1 120); do
  if [[ -S "$SOCKET_FILE" ]]; then
    break
  fi
  if ! kill -0 "$IPI_PID" 2>/dev/null; then
    echo "i-PI (PID $IPI_PID) exited before opening its socket; see ipi_continue_${CURRENT_STEP}_to_${TOTAL_STEPS}.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ ! -S "$SOCKET_FILE" ]]; then
  echo "Timed out waiting for i-PI socket $SOCKET_FILE" >&2
  exit 1
fi

for ((i = 1; i <= NUM_CLIENT; i++)); do
  echo "Launching continuation LAMMPS client $i/${NUM_CLIENT}"
  GPU_CLIENT_INDEX="$i" "$LMP_BIN" -in "$CONTINUE_LMP" > "lammps_continue_${CURRENT_STEP}_to_${TOTAL_STEPS}_client_${i}.log" 2>&1 &
  client_pids+=("$!")
  client_logs+=("lammps_continue_${CURRENT_STEP}_to_${TOTAL_STEPS}_client_${i}.log")
  sleep 1
done

source "$RUN_DIR/monitor_run.sh"
monitor_run "$TOTAL_STEPS" "ipi_continue_${CURRENT_STEP}_to_${TOTAL_STEPS}.log"

echo "Continuation finished at target step ${TOTAL_STEPS}."
EOF
chmod +x "$RUN_DIR/continue.sh"

cat > "$RUN_DIR/README_run.md" <<EOF
# ${RUN_NAME}

This folder preserves the inputs for this Si(110) i-PI/LAMMPS SNAP job.
The recorded external executables are still required.

Fresh run command used:

\`\`\`bash
SCIENCE_ACTIVATE=/dev/null PYTHON_BIN="${PYTHON_BIN}" LMP_BIN="${LMP_BIN}" IPI_BIN="${IPI_BIN}" RUNSTEPS=${RUNSTEPS} NUM_BEADS=${NUM_BEADS} NUM_CLIENT=${NUM_CLIENT} REPLICATE="${REPLICATE}" TEMPERATURE=${TEMPERATURE} THERMO_TAU_FS=${THERMO_TAU_FS} IPI_TIMESTEP_FS=${IPI_TIMESTEP_FS} TIMESTEP=${TIMESTEP} bash "${SCRIPT_DIR}/run.sh"
\`\`\`

Important files:

- \`.run_input.xml\`: i-PI input used for the fresh run.
- \`.run_lmp_Si_ipi.in\`: LAMMPS client input used for the fresh run.
- \`${DATA_FILE}\`: replicated Si(110) LAMMPS data file.
- \`${INIT_XYZ_FILE}\`: replicated i-PI initial XYZ file.
- \`SNAPotential.snapcoeff\` and \`SNAPotential.snapparam\`: SNAP potential files.
- \`continue.sh\`: continue from \`RESTART\` or \`simulation.restart\`.

Continue by adding more steps:

\`\`\`bash
ADD_STEPS=100 bash continue.sh
\`\`\`

Or continue to a specific total i-PI step:

\`\`\`bash
TOTAL_STEPS=40000 bash continue.sh
\`\`\`
EOF

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/organize_run.py" "$RUN_DIR"
  echo "DRY_RUN=1: prepared run directory without launching i-PI/LAMMPS."
  echo "Run directory: $RUN_DIR"
  exit 0
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

IPI_PID=""
client_pids=()
client_logs=()

cleanup() {
  local cleanup_pid watchdog_pid
  # Escalate only our own still-running children after a short grace period.
  (
    sleep 10
    for cleanup_pid in "${client_pids[@]}" "${IPI_PID}"; do
      [[ -n "$cleanup_pid" ]] && kill -KILL "$cleanup_pid" 2>/dev/null || true
    done
  ) &
  watchdog_pid=$!
  for pid in "${client_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done

  if [[ -n "${IPI_PID}" ]] && kill -0 "${IPI_PID}" 2>/dev/null; then
    kill "${IPI_PID}" 2>/dev/null || true
    wait "${IPI_PID}" 2>/dev/null || true
  fi

  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
  rm -f "$SOCKET_FILE"
}

trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

rm -f "$SOCKET_FILE"

cd "$RUN_DIR"
echo "Run directory: $RUN_DIR"

"$IPI_BIN" "$WORK_XML_FILE" > "$RUN_DIR/ipi.log" 2>&1 &
IPI_PID=$!
echo "i-PI started with PID $IPI_PID"

# A fixed sleep here is a race: i-PI's socket bind can take longer than a
# few seconds under load (e.g. several Kokkos/HIP clients initializing their
# GPU contexts at once), and a client that connects too early aborts the
# whole run. Poll for the actual socket file instead, and fail fast (with
# ipi.log intact, see PYTHONUNBUFFERED above) if i-PI itself died first.
echo "Waiting for i-PI socket: $SOCKET_FILE"
for _ in $(seq 1 120); do
  if [[ -S "$SOCKET_FILE" ]]; then
    break
  fi
  if ! kill -0 "$IPI_PID" 2>/dev/null; then
    echo "i-PI (PID $IPI_PID) exited before opening its socket; see $RUN_DIR/ipi.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ ! -S "$SOCKET_FILE" ]]; then
  echo "Timed out waiting for i-PI socket $SOCKET_FILE" >&2
  exit 1
fi

for ((i = 1; i <= NUM_CLIENT; i++)); do
  echo "Launching LAMMPS client $i/${NUM_CLIENT} with ${OMP_NUM_THREADS} OpenMP thread(s)"
  GPU_CLIENT_INDEX="$i" "$LMP_BIN" -in "$WORK_LMP_FILE" > "$RUN_DIR/lammps_client_${i}.log" 2>&1 &
  client_pids+=("$!")
  client_logs+=("$RUN_DIR/lammps_client_${i}.log")
  sleep 1
done

source "$RUN_DIR/monitor_run.sh"
monitor_run "$RUNSTEPS" "$RUN_DIR/ipi.log"

echo "Si(110) TRPMD run finished."
cleanup
trap - EXIT
"$PYTHON_BIN" "$SCRIPT_DIR/organize_run.py" "$RUN_DIR"
