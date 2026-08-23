from __future__ import annotations

from flowjudge.dialam_training_v2 import _content_tokens, _hard_negative_metadata
from flowjudge.patch_data import PatchExample


def _example(*, relations: list[dict]) -> PatchExample:
    return PatchExample.model_validate(
        {
            "schema_version": "dialam_incremental_patch_v1",
            "example_id": "qt30:episode:map:new:b01",
            "update_id": "qt30:episode:map:new",
            "split": "train",
            "dialogue_id": "episode",
            "dialogue_title": "Episode",
            "map_id": "map",
            "block_index": 0,
            "block_count": 1,
            "block_size": 8,
            "new_proposition": {
                "id": "new",
                "text": "Renewable energy lowers carbon emissions",
                "chronological_turn": 3,
                "speaker_id": "s1",
                "speaker": "Speaker 1",
                "raw_locution_id": "l3",
                "raw_locution_text": "Renewable energy lowers carbon emissions",
            },
            "earlier_propositions": [
                {
                    "id": "old",
                    "text": "Carbon emissions fall with renewable energy",
                    "chronological_turn": 2,
                    "speaker_id": "s2",
                    "speaker": "Speaker 2",
                    "raw_locution_id": "l2",
                    "raw_locution_text": "Carbon emissions fall with renewable energy",
                }
            ],
            "gold_patch": {"relations": relations},
            "gold_relation_provenance": [],
        }
    )


def test_content_tokens_remove_stop_words_and_keep_argument_terms() -> None:
    assert _content_tokens("This is about the renewable energy policy") == {
        "energy",
        "policy",
        "renewable",
    }


def test_hard_negative_requires_positive_sibling_and_lexical_overlap() -> None:
    example = _example(relations=[])
    metadata = _hard_negative_metadata(example, positive_sibling_block=True)
    assert metadata is not None
    assert metadata.kind == "positive_sibling_lexical_no_edge"
    assert metadata.max_shared_content_tokens == 4
    assert metadata.closest_turn_gap == 1
    assert _hard_negative_metadata(example, positive_sibling_block=False) is None
