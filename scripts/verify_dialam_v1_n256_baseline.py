#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "baselines" / "dialam_v1_n256" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen DialAM v1/n=256 baseline")
    parser.add_argument(
        "--allow-missing-private",
        action="store_true",
        help="verify metadata but do not fail when ignored private artifacts are absent",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("baseline_id") != "dialam-v1-n256" or not manifest.get("immutable"):
        raise ValueError("unexpected or mutable baseline manifest")
    if manifest["frozen_evaluation"]["scenarios"] != 30:
        raise ValueError("v1/n=256 baseline must retain exactly 30 frozen scenarios")

    missing: list[str] = []
    mismatched: list[str] = []
    verified = 0
    for artifact in manifest["private_artifacts"]:
        path = PROJECT_ROOT / artifact["path"]
        if not path.is_file():
            missing.append(artifact["path"])
            continue
        verified += 1
        if sha256(path) != artifact["sha256"]:
            mismatched.append(artifact["path"])
    if mismatched:
        raise RuntimeError(f"baseline artifact hash mismatch: {mismatched}")
    if missing and not args.allow_missing_private:
        raise FileNotFoundError(f"missing private baseline artifacts: {missing}")

    print(
        json.dumps(
            {
                "baseline_id": manifest["baseline_id"],
                "immutable": True,
                "private_artifacts_verified": verified,
                "private_artifacts_missing": missing,
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
