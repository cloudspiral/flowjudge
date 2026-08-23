#!/usr/bin/env python3
from __future__ import annotations

from flowjudge.experiment_history import write_experiment_history


def main() -> None:
    for label, path in write_experiment_history().items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
