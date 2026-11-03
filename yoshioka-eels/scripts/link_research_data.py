"""Connect a development checkout to an existing Yoshioka data directory."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    data = args.data_root.expanduser().resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    names = ("benchmark_results", "Si_EELS_300K_Tersoff.npy", "Si_EELS_1000K_Tersoff.npy")
    # Check every destination before making links. Never replace existing files.
    for name in names:
        target, link = data / name, root / name
        if not target.exists():
            parser.error(f"Missing data: {target}")
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target.resolve():
                parser.error(f"Refusing to replace: {link}")
    for name in names:
        link = root / name
        if not link.is_symlink():
            link.symlink_to(data / name, target_is_directory=(data / name).is_dir())
    (root / ".local-data.json").write_text(json.dumps(
        {"data_root": str(data), "links": list(names)}, indent=2) + "\n")
    print(f"Research data linked from {data}; no results moved.")


if __name__ == "__main__":
    main()
