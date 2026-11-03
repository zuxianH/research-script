"""Show the installed package and its scientific scope."""

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
    print("The four-repeat complete-pair benchmark remains NOT_ACCEPTED.")


if __name__ == "__main__":
    main()
