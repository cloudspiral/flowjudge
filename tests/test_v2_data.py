import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "build_v2_attachment_data.py"
    spec = importlib.util.spec_from_file_location("build_v2_attachment_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_id_remap_preserves_text_graph_and_direction() -> None:
    module = _load_module()
    row = {
        "example_id": "x",
        "messages": [
            {"role": "user", "content": "U2 [AFF]: Claim.\nU9 [NEG]: Attack."},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"relations": [{"source": "U9", "target": "U2", "type": "responds_to"}]}
                ),
            },
        ],
    }

    result = module._remap_example(row, start_id=1001)
    graph = json.loads(result["messages"][1]["content"])

    assert "U1001 [AFF]: Claim." in result["messages"][0]["content"]
    assert "U1002 [NEG]: Attack." in result["messages"][0]["content"]
    assert graph["relations"][0] == {
        "source": "U1002",
        "target": "U1001",
        "type": "responds_to",
    }
    assert row["example_id"] == "x"


def test_curriculum_case_has_three_cross_side_edges_and_two_hard_negatives() -> None:
    path = Path(__file__).parents[1] / "scripts" / "build_v2_curriculum_data.py"
    spec = importlib.util.spec_from_file_location("build_v2_curriculum_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    row = module._curriculum_example(0, "test service")
    graph = json.loads(row["messages"][1]["content"])

    assert len(graph["relations"]) == 3
    assert len(row["v2_augmentation"]["hard_negative_pairs"]) == 2
    assert "U2002 [AFF]" in row["messages"][0]["content"]
