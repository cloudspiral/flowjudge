from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowjudge.dialam import DEFAULT_ARCHIVE_PATH, DEFAULT_SOURCE_DIR, ensure_qt30_source
from flowjudge.patch_data import DEFAULT_OUTPUT_DIR, build_smoke_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic DialAM/QT30 feasibility-gate data")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = ensure_qt30_source(args.archive, args.source_dir)
    manifest = build_smoke_dataset(source_dir=args.source_dir, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "source": source,
                "statistics": manifest["statistics"],
                "eval_summary": manifest["eval_summary"],
                "validation": manifest["validation"],
                "diagnostic_validation": manifest["diagnostic_validation"],
                "relation_retention_funnel": manifest["corpus_audit"][
                    "relation_retention_funnel"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
