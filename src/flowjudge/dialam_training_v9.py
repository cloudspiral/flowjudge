from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import Field, model_validator

from .dialam import (
    DEFAULT_DIALOGUES_PATH,
    DEFAULT_SOURCE_DIR,
    SOURCE_ARCHIVE_SHA256,
    load_canonical_maps,
)
from .dialam_training import DEFAULT_TRAINING_DIR, TrainingMessage, file_sha256
from .dialam_training_v3 import V3_DEV_EXAMPLES_FILENAME, _dialogue_split, _seeded_hash
from .dialam_training_v5 import (
    CandidateHardness,
    PairwiseDecision,
    PairwiseLabel,
    POSITIVE_LABELS,
    V5_OUTPUT_FILENAME,
    V5_POSITIVE_TARGET_COUNTS,
    _candidate_hardness,
    build_pairwise_prompt,
    candidate_label_distribution,
    pairwise_decisions,
)
from .patch_data import (
    DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    DEFAULT_EVAL_SUMMARY_PATH,
    PROJECT_ROOT,
    PatchExample,
    PatchSplit,
    _materialize_candidate,
    build_update_candidates,
    load_jsonl,
)
from .schemas import StrictModel


V9_MINING_PREFILTER_SIZE = 12_288
V9_SELECTED_NONE_SIZE = 4_096
V9_POSITIVE_SIZE = 4_096
V9_TRAINING_SIZE = 8_192
V9_SMOKE_SIZE = 256
V9_SELECTION_SEED = "dialam-qt30-qlora-v9-selected-model-error-mining"
V9_SOURCE_V5_SHA256 = "23f4b8d552a6bca16cdb79507d89ec9675bb16a8d2f5b7b573c6a0981a42d8db"
V9_SOURCE_ADAPTER_TREE_SHA256 = (
    "cbd9a2a6cae8ab988d78b7694ac9f9829bf9262bbb7a6fab838894a5080f122a"
)
V9_MINING_INPUT_FILENAME = "v9_mining_candidates_n12288.jsonl"
V9_MINING_SCORE_FILENAME = "v9_mining_scores_n12288.jsonl"
V9_MINING_MANIFEST_FILENAME = "v9_mining_manifest.json"
V9_MINING_SCHEMA_FILENAME = "mining_candidate_v9.schema.json"
V9_SCORE_SCHEMA_FILENAME = "mining_score_v9.schema.json"
V9_OUTPUT_FILENAME = "dialam_v9_n8192.jsonl"
V9_SMOKE_OUTPUT_FILENAME = "dialam_v9_resume_smoke_n256.jsonl"
V9_MANIFEST_FILENAME = "training_v9_manifest.json"
V9_TRAINING_SCHEMA_FILENAME = "training_example_v9.schema.json"
V9_PREREGISTRATION_PATH = PROJECT_ROOT / "docs" / "dialam_v9_preregistration.md"


class DialAMMiningCandidateV9(StrictModel):
    schema_version: Literal["dialam_selected_model_mining_candidate_v9"]
    mining_candidate_id: str
    decision_key: str
    source_example_id: str
    update_id: str
    dialogue_id: str
    block_index: int = Field(ge=0)
    candidate_target_id: str
    gold_label: Literal["NONE"]
    candidate_hardness: CandidateHardness
    prompt: str = Field(min_length=1)


class DialAMMiningScoreV9(StrictModel):
    schema_version: Literal["dialam_selected_model_mining_score_v9"]
    mining_candidate_id: str
    decision_key: str
    selected_label: PairwiseLabel
    winning_positive_label: Literal["SUPPORT", "ATTACK", "REPHRASE"]
    label_scores: dict[str, float]
    model_hardness: float
    support_evidence: float
    adapter_tree_sha256: str

    @model_validator(mode="after")
    def validate_scores(self) -> "DialAMMiningScoreV9":
        if set(self.label_scores) != {"NONE", "SUPPORT", "ATTACK", "REPHRASE"}:
            raise ValueError("v9 mining scores must cover the exact four labels")
        if not all(math.isfinite(value) for value in self.label_scores.values()):
            raise ValueError("v9 mining scores must be finite")
        expected_positive = max(
            ("SUPPORT", "ATTACK", "REPHRASE"),
            key=lambda label: (
                self.label_scores[label],
                -("SUPPORT", "ATTACK", "REPHRASE").index(label),
            ),
        )
        if self.winning_positive_label != expected_positive:
            raise ValueError("v9 winning positive label disagrees with scores")
        expected_hardness = (
            self.label_scores[expected_positive] - self.label_scores["NONE"]
        )
        if not math.isclose(self.model_hardness, expected_hardness, abs_tol=1e-7):
            raise ValueError("v9 model hardness disagrees with scores")
        if not math.isclose(
            self.support_evidence,
            self.label_scores["SUPPORT"] - self.label_scores["NONE"],
            abs_tol=1e-7,
        ):
            raise ValueError("v9 SUPPORT evidence disagrees with scores")
        if self.adapter_tree_sha256 != V9_SOURCE_ADAPTER_TREE_SHA256:
            raise ValueError("v9 mining used an unexpected source adapter")
        return self


class DialAMTrainingRowV9(StrictModel):
    schema_version: Literal["dialam_qlora_model_error_corrective_v9"]
    training_row_id: str
    row_role: Literal["POSITIVE_REHEARSAL", "MINED_NONE"]
    source_row_id: str
    source_example_id: str
    update_id: str
    dialogue_id: str
    block_index: int = Field(ge=0)
    candidate_target_id: str
    label: PairwiseLabel
    candidate_hardness: CandidateHardness | None = None
    mining_score: DialAMMiningScoreV9 | None = None
    messages: list[TrainingMessage] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_training_row(self) -> "DialAMTrainingRowV9":
        if self.row_role == "MINED_NONE":
            if self.label != "NONE" or self.mining_score is None:
                raise ValueError("v9 mined rows require a NONE label and mining score")
            if self.candidate_hardness is None:
                raise ValueError("v9 mined rows require candidate hardness")
        elif self.label == "NONE" or self.mining_score is not None:
            raise ValueError("v9 positive rehearsal rows cannot carry mining state")
        if self.messages[1].role != "assistant" or self.messages[1].content != self.label:
            raise ValueError("v9 assistant target must be the exact row label")
        return self


def _distribution(values: list[int | float]) -> dict[str, int | float]:
    return {"minimum": min(values), "mean": mean(values), "maximum": max(values)}


def _write_jsonl(path: Path, rows: list[StrictModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _load_source_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _lexical_rank(decision: PairwiseDecision) -> tuple[object, ...]:
    hardness = _candidate_hardness(decision)
    return (
        -int(hardness.shared_content_tokens >= 2),
        -hardness.overlap_coefficient,
        -hardness.shared_content_tokens,
        -hardness.jaccard_similarity,
        hardness.turn_gap,
        _seeded_hash(V9_SELECTION_SEED, decision.key),
        decision.key,
    )


def select_v9_mining_candidates(
    examples: list[PatchExample],
    *,
    excluded_none_keys: set[str],
    size: int = V9_MINING_PREFILTER_SIZE,
) -> list[DialAMMiningCandidateV9]:
    decisions: dict[str, PairwiseDecision] = {}
    for example in examples:
        block_decisions, _ = pairwise_decisions(example)
        for decision in block_decisions:
            if decision.label != "NONE" or decision.key in excluded_none_keys:
                continue
            if decision.key in decisions:
                raise ValueError(f"duplicate v9 mining decision: {decision.key}")
            decisions[decision.key] = decision
    ranked = sorted(decisions.values(), key=_lexical_rank)
    if len(ranked) < size:
        raise ValueError(f"v9 needs {size} novel NONE candidates, found {len(ranked)}")
    selected = ranked[:size]
    return [
        DialAMMiningCandidateV9(
            schema_version="dialam_selected_model_mining_candidate_v9",
            mining_candidate_id="v9mine-"
            + _seeded_hash(V9_SELECTION_SEED, decision.key)[:24],
            decision_key=decision.key,
            source_example_id=decision.example.example_id,
            update_id=decision.example.update_id,
            dialogue_id=decision.example.dialogue_id,
            block_index=decision.example.block_index,
            candidate_target_id=decision.candidate_target_id,
            gold_label="NONE",
            candidate_hardness=_candidate_hardness(decision),
            prompt=build_pairwise_prompt(
                decision.example,
                decision.candidate_target_id,
            ),
        )
        for decision in selected
    ]


def _load_training_examples_and_splits(
    *,
    source_dir: Path,
    dialogues_path: Path,
    eval_summary_path: Path,
) -> tuple[list[PatchExample], set[str], set[str], set[str]]:
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    frozen_dialogues = set(eval_summary["heldout_dialogue_ids"])
    candidates = build_update_candidates(load_canonical_maps(source_dir, dialogues_path))
    nonfrozen = [
        item
        for item in candidates
        if item.canonical_map.dialogue_id not in frozen_dialogues
    ]
    training_dialogues, development_dialogues = _dialogue_split(
        {item.canonical_map.dialogue_id for item in nonfrozen}
    )
    if (
        training_dialogues & development_dialogues
        or training_dialogues & frozen_dialogues
        or development_dialogues & frozen_dialogues
    ):
        raise ValueError("v9 parent-episode splits overlap")
    examples = [
        example
        for candidate in nonfrozen
        if candidate.canonical_map.dialogue_id in training_dialogues
        for example in _materialize_candidate(candidate, PatchSplit.TRAIN)
    ]
    return examples, training_dialogues, development_dialogues, frozen_dialogues


def build_v9_mining_inputs(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    dialogues_path: Path = DEFAULT_DIALOGUES_PATH,
    eval_summary_path: Path = DEFAULT_EVAL_SUMMARY_PATH,
    frozen_examples_path: Path = DEFAULT_DIAGNOSTIC_EVAL_EXAMPLES_PATH,
    output_dir: Path = DEFAULT_TRAINING_DIR,
) -> dict[str, object]:
    if not V9_PREREGISTRATION_PATH.is_file():
        raise FileNotFoundError("v9 preregistration must exist before mining inputs")
    source_v5_path = output_dir / V5_OUTPUT_FILENAME
    if file_sha256(source_v5_path) != V9_SOURCE_V5_SHA256:
        raise ValueError("v9 provenance v5 corpus differs from the frozen bytes")
    source_rows = _load_source_rows(source_v5_path)
    excluded_none_keys = {
        f"{row['source_example_id']}|{row['candidate_target_id']}|NONE"
        for row in source_rows
        if row["label"] == "NONE"
    }
    examples, training, development, frozen = _load_training_examples_and_splits(
        source_dir=source_dir,
        dialogues_path=dialogues_path,
        eval_summary_path=eval_summary_path,
    )
    available_counts, ambiguous_count = candidate_label_distribution(examples)
    rows = select_v9_mining_candidates(
        examples,
        excluded_none_keys=excluded_none_keys,
    )
    if len({row.mining_candidate_id for row in rows}) != len(rows):
        raise ValueError("v9 mining candidate IDs are not unique")
    if len({row.decision_key for row in rows}) != len(rows):
        raise ValueError("v9 mining decision keys are not unique")
    if {row.dialogue_id for row in rows} - training:
        raise ValueError("v9 mining contains a development or frozen parent episode")
    if {row.decision_key for row in rows} & excluded_none_keys:
        raise ValueError("v9 mining reused a v5 NONE decision")

    dev_path = output_dir / V3_DEV_EXAMPLES_FILENAME
    dev_examples = load_jsonl(dev_path, PatchExample)
    frozen_examples = load_jsonl(frozen_examples_path, PatchExample)
    if {item.dialogue_id for item in dev_examples} != development:
        raise ValueError("v9 development episodes differ from the frozen split")
    if {item.dialogue_id for item in frozen_examples} != frozen:
        raise ValueError("v9 frozen episodes differ from the immutable evaluation")

    output_path = output_dir / V9_MINING_INPUT_FILENAME
    schema_path = output_dir / V9_MINING_SCHEMA_FILENAME
    score_schema_path = output_dir / V9_SCORE_SCHEMA_FILENAME
    _write_jsonl(output_path, rows)
    schema_path.write_text(
        json.dumps(DialAMMiningCandidateV9.model_json_schema(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    score_schema_path.write_text(
        json.dumps(DialAMMiningScoreV9.model_json_schema(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    hardness = [row.candidate_hardness for row in rows]
    manifest: dict[str, object] = {
        "schema_version": "dialam_v9_mining_manifest_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_seed": V9_SELECTION_SEED,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_v5_sha256": file_sha256(source_v5_path),
        "source_adapter_tree_sha256": V9_SOURCE_ADAPTER_TREE_SHA256,
        "split_unit": "original_parent_episode",
        "training_parent_episode_count": len(training),
        "development_parent_episode_count": len(development),
        "frozen_parent_episode_count": len(frozen),
        "dialogue_leakage": False,
        "available_candidate_label_counts": dict(sorted(available_counts.items())),
        "ambiguous_same_pair_decisions_excluded": ambiguous_count,
        "v5_none_decision_count_excluded": len(excluded_none_keys),
        "prefilter_size": len(rows),
        "prefilter_policy": (
            "novel unambiguous training NONE decisions in frozen v5 lexical-hardness "
            "order with seeded stable ties"
        ),
        "candidate_shared_content_tokens": _distribution(
            [item.shared_content_tokens for item in hardness]
        ),
        "candidate_overlap_coefficient": _distribution(
            [item.overlap_coefficient for item in hardness]
        ),
        "mining": {
            "labels": ["NONE", "SUPPORT", "ATTACK", "REPHRASE"],
            "sequence_score": "mean label-token log probability",
            "model_hardness": "max(positive label score) - NONE score",
            "chunk_size_max": 128,
            "persistent_chunk_commit": True,
            "identity_mismatch_refuses_resume": True,
            "completed_run_is_idempotent": True,
        },
        "evaluation_hashes": {
            "development_examples_sha256": file_sha256(dev_path),
            "frozen_examples_sha256": file_sha256(frozen_examples_path),
        },
        "preregistration": {
            "path": str(V9_PREREGISTRATION_PATH.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(V9_PREREGISTRATION_PATH),
        },
        "output": {
            "path": str(output_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(output_path),
        },
        "schema": {
            "path": str(schema_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(schema_path),
        },
        "score_schema": {
            "path": str(score_schema_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(score_schema_path),
        },
        "publication": (
            "Mining inputs contain transformed QT30 text and remain local/ignored; "
            "this tracked manifest and schemas contain no corpus text."
        ),
    }
    manifest_path = output_dir / V9_MINING_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def select_v9_mined_none(
    candidates: list[DialAMMiningCandidateV9],
    scores: list[DialAMMiningScoreV9],
    *,
    size: int = V9_SELECTED_NONE_SIZE,
) -> list[tuple[DialAMMiningCandidateV9, DialAMMiningScoreV9]]:
    by_id = {item.mining_candidate_id: item for item in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("v9 mining candidates contain duplicate IDs")
    score_by_id = {item.mining_candidate_id: item for item in scores}
    if len(score_by_id) != len(scores) or set(score_by_id) != set(by_id):
        raise ValueError("v9 mining score coverage differs from candidate inputs")
    for candidate_id, score in score_by_id.items():
        if score.decision_key != by_id[candidate_id].decision_key:
            raise ValueError("v9 mining score decision key mismatch")

    def rank(item: DialAMMiningScoreV9) -> tuple[object, ...]:
        candidate = by_id[item.mining_candidate_id]
        hardness = candidate.candidate_hardness
        return (
            -item.model_hardness,
            -item.support_evidence,
            -int(hardness.shared_content_tokens >= 2),
            -hardness.overlap_coefficient,
            -hardness.shared_content_tokens,
            -hardness.jaccard_similarity,
            hardness.turn_gap,
            _seeded_hash(V9_SELECTION_SEED, item.decision_key),
        )

    ranked = sorted(scores, key=rank)
    if len(ranked) < size:
        raise ValueError(f"v9 needs {size} mined NONE scores, found {len(ranked)}")
    return [(by_id[item.mining_candidate_id], item) for item in ranked[:size]]


def _ordered_with_nested_smoke(
    rows: list[DialAMTrainingRowV9],
) -> list[DialAMTrainingRowV9]:
    target = {"NONE": 128, "ATTACK": 43, "REPHRASE": 43, "SUPPORT": 42}
    by_label = {
        label: sorted(
            (row for row in rows if row.label == label),
            key=lambda row: _seeded_hash(V9_SELECTION_SEED, row.training_row_id),
        )
        for label in target
    }
    smoke = [row for label, count in target.items() for row in by_label[label][:count]]
    smoke = sorted(
        smoke,
        key=lambda row: _seeded_hash(V9_SELECTION_SEED + "-smoke", row.training_row_id),
    )
    smoke_ids = {row.training_row_id for row in smoke}
    remainder = sorted(
        (row for row in rows if row.training_row_id not in smoke_ids),
        key=lambda row: _seeded_hash(V9_SELECTION_SEED + "-full", row.training_row_id),
    )
    return smoke + remainder


def materialize_v9_rows(
    positive_source_rows: list[dict],
    mined_none: list[tuple[DialAMMiningCandidateV9, DialAMMiningScoreV9]],
) -> list[DialAMTrainingRowV9]:
    rows: list[DialAMTrainingRowV9] = []
    for source in positive_source_rows:
        if source["label"] == "NONE":
            raise ValueError("v9 positive rehearsal source contains NONE")
        rows.append(
            DialAMTrainingRowV9(
                schema_version="dialam_qlora_model_error_corrective_v9",
                training_row_id="v9pos-"
                + _seeded_hash(V9_SELECTION_SEED, source["training_row_id"])[:24],
                row_role="POSITIVE_REHEARSAL",
                source_row_id=source["training_row_id"],
                source_example_id=source["source_example_id"],
                update_id=source["update_id"],
                dialogue_id=source["dialogue_id"],
                block_index=source["block_index"],
                candidate_target_id=source["candidate_target_id"],
                label=source["label"],
                messages=[TrainingMessage.model_validate(item) for item in source["messages"]],
            )
        )
    for candidate, score in mined_none:
        rows.append(
            DialAMTrainingRowV9(
                schema_version="dialam_qlora_model_error_corrective_v9",
                training_row_id="v9none-"
                + _seeded_hash(V9_SELECTION_SEED, candidate.mining_candidate_id)[:24],
                row_role="MINED_NONE",
                source_row_id=candidate.mining_candidate_id,
                source_example_id=candidate.source_example_id,
                update_id=candidate.update_id,
                dialogue_id=candidate.dialogue_id,
                block_index=candidate.block_index,
                candidate_target_id=candidate.candidate_target_id,
                label="NONE",
                candidate_hardness=candidate.candidate_hardness,
                mining_score=score,
                messages=[
                    TrainingMessage(role="user", content=candidate.prompt),
                    TrainingMessage(role="assistant", content="NONE"),
                ],
            )
        )
    return _ordered_with_nested_smoke(rows)


def build_dialam_v9_training_corpus(
    *,
    scores_path: Path | None = None,
    output_dir: Path = DEFAULT_TRAINING_DIR,
) -> dict[str, object]:
    source_v5_path = output_dir / V5_OUTPUT_FILENAME
    mining_input_path = output_dir / V9_MINING_INPUT_FILENAME
    mining_manifest_path = output_dir / V9_MINING_MANIFEST_FILENAME
    scores_path = scores_path or output_dir / V9_MINING_SCORE_FILENAME
    if file_sha256(source_v5_path) != V9_SOURCE_V5_SHA256:
        raise ValueError("v9 source v5 corpus differs from the frozen bytes")
    mining_manifest = json.loads(mining_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(mining_input_path) != mining_manifest["output"]["sha256"]:
        raise ValueError("v9 mining inputs differ from their frozen manifest")
    candidates = [
        DialAMMiningCandidateV9.model_validate_json(line)
        for line in mining_input_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    scores = [
        DialAMMiningScoreV9.model_validate_json(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected_none = select_v9_mined_none(candidates, scores)
    source_rows = _load_source_rows(source_v5_path)
    positive_rows = [row for row in source_rows if row["label"] != "NONE"]
    if len(positive_rows) != V9_POSITIVE_SIZE:
        raise ValueError("v9 requires the exact 4,096 positive v5 rows")
    if Counter(row["label"] for row in positive_rows) != Counter(V5_POSITIVE_TARGET_COUNTS):
        raise ValueError("v9 positive rehearsal mix differs from v5")
    rows = materialize_v9_rows(positive_rows, selected_none)
    if len(rows) != V9_TRAINING_SIZE:
        raise ValueError("v9 training corpus has the wrong size")
    if len({row.training_row_id for row in rows}) != len(rows):
        raise ValueError("v9 training row IDs are not unique")
    if Counter(row.label for row in rows) != Counter(
        {"NONE": 4096, **V5_POSITIVE_TARGET_COUNTS}
    ):
        raise ValueError("v9 training label mix differs from the preregistration")
    smoke = rows[:V9_SMOKE_SIZE]
    if Counter(row.label for row in smoke) != Counter(
        {"NONE": 128, "ATTACK": 43, "REPHRASE": 43, "SUPPORT": 42}
    ):
        raise ValueError("v9 nested smoke prefix has the wrong label mix")

    output_path = output_dir / V9_OUTPUT_FILENAME
    smoke_path = output_dir / V9_SMOKE_OUTPUT_FILENAME
    schema_path = output_dir / V9_TRAINING_SCHEMA_FILENAME
    _write_jsonl(output_path, rows)
    _write_jsonl(smoke_path, smoke)
    schema_path.write_text(
        json.dumps(DialAMTrainingRowV9.model_json_schema(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    selected_scores = [score for _, score in selected_none]
    selected_hardness = [candidate.candidate_hardness for candidate, _ in selected_none]
    manifest: dict[str, object] = {
        "schema_version": "dialam_training_manifest_v9",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": "v9-selected-model-error-corrective",
        "size": len(rows),
        "smoke_size": len(smoke),
        "selection_seed": V9_SELECTION_SEED,
        "split_unit": "original_parent_episode",
        "dialogue_leakage": False,
        "new_external_data": False,
        "source_v5_sha256": file_sha256(source_v5_path),
        "source_adapter_tree_sha256": V9_SOURCE_ADAPTER_TREE_SHA256,
        "mining_input_sha256": file_sha256(mining_input_path),
        "mining_score_sha256": file_sha256(scores_path),
        "mining_score_count": len(scores),
        "selected_none_count": len(selected_none),
        "row_label_counts": dict(sorted(Counter(row.label for row in rows).items())),
        "row_role_counts": dict(sorted(Counter(row.row_role for row in rows).items())),
        "selected_winning_positive_labels": dict(
            sorted(Counter(item.winning_positive_label for item in selected_scores).items())
        ),
        "selected_uncalibrated_positive_predictions": sum(
            item.model_hardness > 0 for item in selected_scores
        ),
        "selected_v5_1_margin_false_predictions": sum(
            item.model_hardness > 3.0 for item in selected_scores
        ),
        "selected_model_hardness": _distribution(
            [item.model_hardness for item in selected_scores]
        ),
        "selected_support_evidence": _distribution(
            [item.support_evidence for item in selected_scores]
        ),
        "selected_shared_content_tokens": _distribution(
            [item.shared_content_tokens for item in selected_hardness]
        ),
        "objective": {
            "name": "restricted_four_label_score_cross_entropy",
            "labels": ["NONE", "SUPPORT", "ATTACK", "REPHRASE"],
            "sequence_score": "mean label-token log probability",
            "loss": "cross_entropy(stack(four_label_scores), gold_label_index)",
            "class_weights": False,
            "label_smoothing": False,
        },
        "continuation": {
            "source_adapter": "v5_n8192",
            "source_adapter_tree_sha256": V9_SOURCE_ADAPTER_TREE_SHA256,
            "optimizer_state_reset_between_v5_and_v9": True,
        },
        "fixed_training_config": {
            "epochs": 1.0,
            "learning_rate": 5e-6,
            "effective_batch_size": 8,
            "seed": 20260823,
            "save_steps": {"v9-smoke": 8, "v9": 100},
            "save_total_limit": 2,
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
            "path": str(V9_PREREGISTRATION_PATH.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(V9_PREREGISTRATION_PATH),
        },
        "mining_manifest_sha256": file_sha256(mining_manifest_path),
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
            "Training rows and mining scores remain local/ignored; the tracked "
            "manifest, schemas, and reconstruction code contain no corpus text."
        ),
    }
    manifest_path = output_dir / V9_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
