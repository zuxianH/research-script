"""Show installed package information."""

import argparse
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=version("yoshioka-eels"))
    parser.parse_args()
    print(f"yoshioka-eels {version('yoshioka-eels')}")
    print(f"Code: {Path(__file__).resolve().parent}")
    print("Units: SI unless a function explicitly states otherwise.")
    print("Includes paraxial propagation, coupled channels, and leading-pair EELS.")


if __name__ == "__main__":
    main()
