from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .dialam_training import DEFAULT_TRAINING_DIR, TrainingMessage, file_sha256
from .dialam_training_v3 import _seeded_hash
from .dialam_training_v5 import (
    CandidateHardness,
    DialAMTrainingRowV5,
    PairwiseLabel,
    V5_OUTPUT_FILENAME,
)
from .patch_data import PROJECT_ROOT
from .schemas import StrictModel


V7_TRAINING_SIZE = 8192
V7_SMOKE_SIZE = 256
V7_TRAINING_SEED = "dialam-qt30-qlora-v7-reciprocal-preference"
V7_SOURCE_V5_SHA256 = "23f4b8d552a6bca16cdb79507d89ec9675bb16a8d2f5b7b573c6a0981a42d8db"
V7_OUTPUT_FILENAME = "dialam_v7_n8192.jsonl"
V7_SMOKE_OUTPUT_FILENAME = "dialam_v7_resume_smoke_n256.jsonl"
V7_MANIFEST_FILENAME = "training_v7_manifest.json"
V7_SCHEMA_FILENAME = "training_example_v7.schema.json"
V7_PREREGISTRATION_PATH = PROJECT_ROOT / "docs" / "dialam_v7_preregistration.md"


class DialAMPreferenceRowV7(StrictModel):
    schema_version: Literal["dialam_qlora_reciprocal_preference_v7"]
    preference_row_id: str
    source_v5_training_row_id: str
    pair_group_id: str
    pair_role: Literal["POSITIVE", "NONE"]
    source_example_id: str
    update_id: str
    dialogue_id: str
    block_index: int = Field(ge=0)
    candidate_target_id: str
    paired_candidate_target_id: str
    chosen_label: PairwiseLabel
    rejected_label: PairwiseLabel
    positive_label: Literal["SUPPORT", "ATTACK", "REPHRASE"]
    repetition_index: int = Field(ge=0)
    is_repeated_positive: bool
    negative_hardness: CandidateHardness
    messages: list[TrainingMessage] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_reciprocal_preference(self) -> "DialAMPreferenceRowV7":
        if self.chosen_label == self.rejected_label:
            raise ValueError("chosen and rejected labels must differ")
        expected = (
            (self.positive_label, "NONE")
            if self.pair_role == "POSITIVE"
            else ("NONE", self.positive_label)
        )
        if (self.chosen_label, self.rejected_label) != expected:
            raise ValueError("preference direction differs from the reciprocal pair policy")
        if self.messages[1].role != "assistant":
            raise ValueError("second message must be the assistant target")
        if self.messages[1].content != self.chosen_label:
            raise ValueError("assistant target must equal the chosen label")
        return self


def _load_v5_rows(path: Path) -> list[DialAMTrainingRowV5]:
    return [
        DialAMTrainingRowV5.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def convert_v5_rows_to_preferences(
    rows: list[DialAMTrainingRowV5],
) -> list[DialAMPreferenceRowV7]:
    if len(rows) % 2:
        raise ValueError("v5 rows must contain complete adjacent contrast pairs")
    converted: list[DialAMPreferenceRowV7] = []
    for index in range(0, len(rows), 2):
        positive, negative = rows[index : index + 2]
        if [positive.pair_role, negative.pair_role] != ["POSITIVE", "NONE"]:
            raise ValueError(f"v5 rows {index}:{index + 2} are not POSITIVE/NONE")
        if positive.pair_group_id != negative.pair_group_id:
            raise ValueError(f"v5 rows {index}:{index + 2} cross pair groups")
        if positive.source_example_id != negative.source_example_id:
            raise ValueError("reciprocal preferences must preserve the exact source block")
        if positive.positive_label != negative.positive_label:
            raise ValueError("v5 pair positive labels disagree")
        for row in (positive, negative):
            chosen: PairwiseLabel = row.label
            rejected: PairwiseLabel = (
                "NONE" if row.pair_role == "POSITIVE" else row.positive_label
            )
            converted.append(
                DialAMPreferenceRowV7(
                    schema_version="dialam_qlora_reciprocal_preference_v7",
                    preference_row_id="v7pref-"
                    + _seeded_hash(
                        V7_TRAINING_SEED,
                        f"{row.pair_group_id}|{row.pair_role}|{chosen}|{rejected}",
                    )[:24],
                    source_v5_training_row_id=row.training_row_id,
                    pair_group_id=row.pair_group_id,
                    pair_role=row.pair_role,
                    source_example_id=row.source_example_id,
                    update_id=row.update_id,
                    dialogue_id=row.dialogue_id,
                    block_index=row.block_index,
                    candidate_target_id=row.candidate_target_id,
                    paired_candidate_target_id=row.paired_candidate_target_id,
                    chosen_label=chosen,
                    rejected_label=rejected,
                    positive_label=row.positive_label,
                    repetition_index=row.repetition_index,
                    is_repeated_positive=row.is_repeated_positive,
                    negative_hardness=row.negative_hardness,
                    messages=[row.messages[0], TrainingMessage(role="assistant", content=chosen)],
                )
            )
    return converted


def _validate_pairs(rows: list[DialAMPreferenceRowV7]) -> None:
    groups: dict[str, list[DialAMPreferenceRowV7]] = defaultdict(list)
    for row in rows:
        groups[row.pair_group_id].append(row)
    for pair_group_id, members in groups.items():
        if len(members) != 2:
            raise ValueError(f"{pair_group_id} does not contain two preferences")
        by_role = {item.pair_role: item for item in members}
        if set(by_role) != {"POSITIVE", "NONE"}:
            raise ValueError(f"{pair_group_id} lacks reciprocal roles")
        positive, negative = by_role["POSITIVE"], by_role["NONE"]
        if positive.messages[0].content.split("CANDIDATE TARGET ID", 1)[0] != (
            negative.messages[0].content.split("CANDIDATE TARGET ID", 1)[0]
        ):
            raise ValueError(f"{pair_group_id} pairwise prompt policy differs")
        if positive.chosen_label != negative.rejected_label:
            raise ValueError(f"{pair_group_id} does not reverse the positive relation")
        if positive.rejected_label != negative.chosen_label:
            raise ValueError(f"{pair_group_id} does not reverse NONE")


def _write_jsonl(path: Path, rows: list[DialAMPreferenceRowV7]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def build_dialam_v7_preference_corpus(
    *,
    source_v5_path: Path | None = None,
    output_dir: Path = DEFAULT_TRAINING_DIR,
) -> dict[str, object]:
    source_path = source_v5_path or output_dir / V5_OUTPUT_FILENAME
    source_hash = file_sha256(source_path)
    if source_hash != V7_SOURCE_V5_SHA256:
        raise ValueError("v7 source bytes differ from the frozen v5 training corpus")
    if not V7_PREREGISTRATION_PATH.is_file():
        raise FileNotFoundError("v7 preregistration must exist before corpus construction")

    v5_rows = _load_v5_rows(source_path)
    rows = convert_v5_rows_to_preferences(v5_rows)
    if len(rows) != V7_TRAINING_SIZE:
        raise ValueError(f"v7 must contain exactly {V7_TRAINING_SIZE} preferences")
    if len({row.preference_row_id for row in rows}) != len(rows):
        raise ValueError("v7 preference-row IDs are not unique")
    _validate_pairs(rows)

    smoke_rows = rows[:V7_SMOKE_SIZE]
    if len({row.pair_group_id for row in smoke_rows}) != V7_SMOKE_SIZE // 2:
        raise ValueError("v7 smoke subset must contain complete adjacent pairs")
    _validate_pairs(smoke_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / V7_OUTPUT_FILENAME
    smoke_path = output_dir / V7_SMOKE_OUTPUT_FILENAME
    schema_path = output_dir / V7_SCHEMA_FILENAME
    _write_jsonl(output_path, rows)
    _write_jsonl(smoke_path, smoke_rows)
    schema_path.write_text(
        json.dumps(DialAMPreferenceRowV7.model_json_schema(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v7",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": "v7-qt30-reciprocal-preference",
        "size": len(rows),
        "smoke_size": len(smoke_rows),
        "selection_seed": V7_TRAINING_SEED,
        "source_corpus": "exact frozen DialAM v5 QT30 pair corpus",
        "source_v5_sha256": source_hash,
        "source_v5_rows": len(v5_rows),
        "source_v5_pair_groups": len({row.pair_group_id for row in v5_rows}),
        "split_unit": "original_parent_episode",
        "dialogue_ids_unchanged_from_v5": True,
        "new_external_data": False,
        "chosen_label_counts": dict(
            sorted(Counter(row.chosen_label for row in rows).items())
        ),
        "rejected_label_counts": dict(
            sorted(Counter(row.rejected_label for row in rows).items())
        ),
        "pair_role_counts": dict(sorted(Counter(row.pair_role for row in rows).items())),
        "objective": {
            "name": "reciprocal_candidate_preference",
            "positive_candidate_preference": "gold direct relation > NONE",
            "hard_negative_candidate_preference": "NONE > paired gold relation",
            "same_complete_block": True,
            "inference_alignment": "mean allowed-label token log probability",
            "preference_loss": "softplus(-(chosen_score-rejected_score))",
            "chosen_label_nll_weight": 0.1,
        },
        "continuation": {
            "source_adapter": "v5_n8192",
            "source_adapter_tree_sha256": (
                "cbd9a2a6cae8ab988d78b7694ac9f9829bf9262bbb7a6fab838894a5080f122a"
            ),
            "optimizer_state_reset_between_v5_and_v7": True,
        },
        "resumability": {
            "smoke_subset_is_nested_prefix": True,
            "periodic_full_trainer_state": True,
            "persistent_modal_volume_commit_on_every_save": True,
            "identity_mismatch_refuses_resume": True,
            "completed_run_is_idempotent_in_auto_mode": True,
        },
        "development_gate": {
            "exact_patch_accuracy_min": 0.5333333333333333,
            "edge_f1_min": 0.56,
            "relation_macro_f1_min": 0.5427350427350427,
            "false_edges_per_update_max": 0.3,
            "none_scenarios_with_false_edges_max": 2,
            "json_validity_rate_min": 1.0,
            "schema_validity_rate_min": 1.0,
        },
        "frozen_promotion_gate": {
            "exact_patch_accuracy_min": 0.5333333333333333,
            "edge_f1_min": 0.5061904761904762,
            "relation_macro_f1_min": 0.4743589743589744,
            "attack_f1_min": 0.3076923076923077,
            "false_edges_per_update_max": 0.26666666666666666,
            "none_scenarios_with_false_edges_max": 0,
            "json_validity_rate_min": 1.0,
            "schema_validity_rate_min": 1.0,
        },
        "preregistration": {
            "path": str(V7_PREREGISTRATION_PATH.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(V7_PREREGISTRATION_PATH),
        },
        "output": {
            "path": str(output_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(output_path),
        },
        "smoke_output": {
            "path": str(smoke_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(smoke_path),
        },
        "schema": {
            "path": str(schema_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(schema_path),
        },
        "publication": (
            "Rows contain transformed QT30 text and remain local/ignored unless the "
            "project owner's redistribution permission is documented for publication."
        ),
    }
    manifest_path = output_dir / V7_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
