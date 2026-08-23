from __future__ import annotations

import json
import hashlib
import urllib.request
import zipfile
from collections import Counter, defaultdict
from enum import StrEnum
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field, model_validator

from .schemas import StrictModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "source" / "dialam_qt30" / "extracted" / "dataset"
DEFAULT_DIALOGUES_PATH = PROJECT_ROOT / "data" / "dialam" / "dialogues.json"
DEFAULT_ARCHIVE_PATH = PROJECT_ROOT / "data" / "source" / "dialam_qt30" / "dataset.zip"
SOURCE_ARCHIVE_URL = "http://dialam.arg.tech/res/files/dataset.zip"
SOURCE_ARCHIVE_SHA256 = "9d3d70c5ad86c815f821928dc515ee80a8e2a24d3f5d224557045f50f7a73791"
SOURCE_MAP_COUNT = 1478


class NormalizedRelationLabel(StrEnum):
    SUPPORT = "SUPPORT"
    ATTACK = "ATTACK"
    REPHRASE = "REPHRASE"


class OriginalEdgeDirection(StrEnum):
    LATER_TO_EARLIER = "later_to_earlier"
    EARLIER_TO_LATER = "earlier_to_later"
    SAME_TURN = "same_turn"
    UNKNOWN = "unknown"


RELATION_LABEL_BY_NODE_TYPE = {
    "RA": NormalizedRelationLabel.SUPPORT,
    "CA": NormalizedRelationLabel.ATTACK,
    "MA": NormalizedRelationLabel.REPHRASE,
}


class LocutionGrounding(StrictModel):
    ya_node_id: str
    ya_label: str
    raw_locution_id: str
    raw_locution_text: str
    speaker_id: str | None = None
    speaker: str | None = None
    start: str | None = None
    timestamp: str | None = None
    chronological_turn: int = Field(gt=0)


class CanonicalProposition(StrictModel):
    dialogue_id: str
    map_id: str
    proposition_id: str
    proposition_text: str
    proposition_order: int = Field(gt=0)
    source_node_index: int = Field(ge=0)
    chronological_turn: int | None = Field(default=None, gt=0)
    groundings: list[LocutionGrounding]

    @property
    def has_unambiguous_grounding(self) -> bool:
        return len(self.groundings) == 1 and self.chronological_turn is not None


class CanonicalRelation(StrictModel):
    dialogue_id: str
    map_id: str
    relation_node_id: str
    relation_node_type: str
    original_relation_label: str
    normalized_relation_label: NormalizedRelationLabel
    original_source_proposition_ids: list[str]
    original_target_proposition_ids: list[str]
    original_incoming_node_ids: list[str]
    original_outgoing_node_ids: list[str]
    original_edge_direction: OriginalEdgeDirection
    canonical_source_proposition_id: str | None = None
    canonical_target_proposition_id: str | None = None
    is_binary_direct: bool
    is_unambiguous_binary: bool

    @model_validator(mode="after")
    def validate_binary_fields(self) -> "CanonicalRelation":
        if self.is_binary_direct != (
            len(self.original_source_proposition_ids) == 1
            and len(self.original_target_proposition_ids) == 1
        ):
            raise ValueError("binary flag does not match proposition endpoint arity")
        has_canonical_pair = (
            self.canonical_source_proposition_id is not None
            and self.canonical_target_proposition_id is not None
        )
        if self.is_unambiguous_binary and not (self.is_binary_direct and has_canonical_pair):
            raise ValueError("unambiguous binary relation requires a canonical pair")
        return self


class CanonicalMap(StrictModel):
    dialogue_id: str
    dialogue_title: str
    dialogue_order: int = Field(ge=0)
    map_id: str
    map_order: int = Field(ge=0)
    source_file: str
    raw_node_type_counts: dict[str, int]
    raw_edge_count: int = Field(ge=0)
    source_locution_count: int = Field(ge=0)
    chronology_issues: list[str]
    propositions: list[CanonicalProposition]
    relations: list[CanonicalRelation]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "CanonicalMap":
        proposition_ids = [item.proposition_id for item in self.propositions]
        relation_ids = [item.relation_node_id for item in self.relations]
        if len(proposition_ids) != len(set(proposition_ids)):
            raise ValueError(f"duplicate proposition IDs in {self.map_id}")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError(f"duplicate relation IDs in {self.map_id}")
        return self


class DialogueMapEntry(StrictModel):
    dialogue_id: str
    dialogue_title: str
    dialogue_order: int = Field(ge=0)
    map_order: int = Field(ge=0)


def _as_id(value: Any) -> str:
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_qt30_source(
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> dict[str, Any]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        partial_path = archive_path.with_suffix(".zip.part")
        request = urllib.request.Request(
            SOURCE_ARCHIVE_URL,
            headers={"User-Agent": "FlowJudge-QT30-feasibility-gate/1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, partial_path.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if _sha256(partial_path) != SOURCE_ARCHIVE_SHA256:
            raise ValueError("downloaded QT30 archive checksum mismatch")
        partial_path.replace(archive_path)
    archive_digest = _sha256(archive_path)
    if archive_digest != SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            f"QT30 archive checksum mismatch: expected {SOURCE_ARCHIVE_SHA256}, got {archive_digest}"
        )

    existing_maps = sorted(source_dir.glob("*.json")) if source_dir.exists() else []
    if len(existing_maps) != SOURCE_MAP_COUNT:
        source_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                item
                for item in archive.infolist()
                if item.filename.startswith("dataset/") and item.filename.endswith(".json")
            )
            if len(members) != SOURCE_MAP_COUNT:
                raise ValueError(f"expected {SOURCE_MAP_COUNT} JSON maps in QT30 archive, got {len(members)}")
            for member in members:
                target = source_dir / Path(member.filename).name
                with archive.open(member) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
        existing_maps = sorted(source_dir.glob("*.json"))
    if len(existing_maps) != SOURCE_MAP_COUNT:
        raise ValueError(f"expected {SOURCE_MAP_COUNT} extracted QT30 maps, got {len(existing_maps)}")
    return {
        "archive_path": str(archive_path),
        "archive_sha256": archive_digest,
        "source_dir": str(source_dir),
        "map_count": len(existing_maps),
    }


def _speaker_from_text(text: str) -> str | None:
    if " : " in text:
        speaker = text.split(" : ", 1)[0].strip()
    elif ":" in text:
        speaker = text.split(":", 1)[0].strip()
    else:
        return None
    return speaker or None


def load_dialogue_map(path: Path = DEFAULT_DIALOGUES_PATH) -> dict[str, DialogueMapEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, DialogueMapEntry] = {}
    replacement_ids = {
        details["replaces_metadata_map_id"]
        for details in payload.get("archive_overrides", {}).values()
    }
    for dialogue_order, dialogue in enumerate(payload["dialogues"]):
        map_order = 0
        for map_id in dialogue["map_ids"]:
            if map_id in replacement_ids:
                map_order += 1
                continue
            if map_id in entries:
                raise ValueError(f"map {map_id} appears in more than one QT30 dialogue")
            entries[map_id] = DialogueMapEntry(
                dialogue_id=dialogue["dialogue_id"],
                dialogue_title=dialogue["title"],
                dialogue_order=dialogue_order,
                map_order=map_order,
            )
            map_order += 1

    for map_id, details in payload.get("archive_overrides", {}).items():
        matching = [
            (index, dialogue)
            for index, dialogue in enumerate(payload["dialogues"])
            if dialogue["dialogue_id"] == details["dialogue_id"]
        ]
        if len(matching) != 1:
            raise ValueError(f"invalid dialogue override for {map_id}")
        dialogue_order, dialogue = matching[0]
        replaced_id = details["replaces_metadata_map_id"]
        entries[map_id] = DialogueMapEntry(
            dialogue_id=dialogue["dialogue_id"],
            dialogue_title=dialogue["title"],
            dialogue_order=dialogue_order,
            map_order=dialogue["map_ids"].index(replaced_id),
        )
    return entries


def _locution_order(
    node_by_id: dict[str, dict[str, Any]],
    node_index: dict[str, int],
    locution_ids: set[str],
    incoming: dict[str, list[str]],
    outgoing: dict[str, list[str]],
) -> tuple[dict[str, int], list[str]]:
    precedence: dict[str, set[str]] = {node_id: set() for node_id in locution_ids}
    successors: dict[str, set[str]] = {node_id: set() for node_id in locution_ids}
    for node_id, node in node_by_id.items():
        if node.get("type") != "TA":
            continue
        sources = [item for item in incoming[node_id] if item in locution_ids]
        targets = [item for item in outgoing[node_id] if item in locution_ids]
        for source in sources:
            for target in targets:
                if source == target:
                    continue
                successors[source].add(target)
                precedence[target].add(source)

    ready: list[tuple[int, int, str]] = []
    for node_id in locution_ids:
        if not precedence[node_id]:
            heappush(ready, (node_index[node_id], int(node_id), node_id))
    ordered: list[str] = []
    while ready:
        _, _, node_id = heappop(ready)
        ordered.append(node_id)
        for target in sorted(successors[node_id], key=lambda item: (node_index[item], int(item))):
            precedence[target].discard(node_id)
            if not precedence[target]:
                heappush(ready, (node_index[target], int(target), target))

    issues: list[str] = []
    if len(ordered) != len(locution_ids):
        remaining = sorted(locution_ids - set(ordered), key=lambda item: (node_index[item], int(item)))
        issues.append(f"TA cycle or unresolved precedence among locutions: {','.join(remaining)}")
        ordered.extend(remaining)
    return {node_id: index + 1 for index, node_id in enumerate(ordered)}, issues


def parse_nodeset(
    path: Path,
    *,
    dialogue: DialogueMapEntry,
) -> CanonicalMap:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"nodes", "edges", "locutions"}:
        raise ValueError(f"unexpected AIF keys in {path.name}: {sorted(payload)}")

    nodes = payload["nodes"]
    node_by_id = {_as_id(node["nodeID"]): node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError(f"duplicate node ID in {path.name}")
    node_index = {_as_id(node["nodeID"]): index for index, node in enumerate(nodes)}
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in payload["edges"]:
        from_id = _as_id(edge["fromID"])
        to_id = _as_id(edge["toID"])
        if from_id not in node_by_id or to_id not in node_by_id:
            raise ValueError(f"edge {edge.get('edgeID')} references an unknown node in {path.name}")
        outgoing[from_id].append(to_id)
        incoming[to_id].append(from_id)

    locution_metadata: dict[str, dict[str, Any]] = {}
    duplicate_locution_ids: set[str] = set()
    invalid_locution_ids: set[str] = set()
    missing_locution_ids: set[str] = set()
    for item in payload["locutions"]:
        locution_id = _as_id(item["nodeID"])
        if locution_id not in node_by_id:
            missing_locution_ids.add(locution_id)
            continue
        if node_by_id[locution_id]["type"] != "L":
            invalid_locution_ids.add(locution_id)
            continue
        if locution_id in locution_metadata:
            duplicate_locution_ids.add(locution_id)
            continue
        locution_metadata[locution_id] = item
    locution_ids = set(locution_metadata)

    chronological_turn, chronology_issues = _locution_order(
        node_by_id,
        node_index,
        locution_ids,
        incoming,
        outgoing,
    )
    if duplicate_locution_ids:
        chronology_issues.append(
            "duplicate locution metadata IDs deduplicated: "
            + ",".join(sorted(duplicate_locution_ids, key=int))
        )
    if invalid_locution_ids:
        chronology_issues.append(
            "non-L nodes ignored in locution metadata: "
            + ",".join(sorted(invalid_locution_ids, key=int))
        )
    if missing_locution_ids:
        chronology_issues.append(
            "unknown nodes ignored in locution metadata: "
            + ",".join(sorted(missing_locution_ids, key=int))
        )
    groundings_by_proposition: dict[str, list[LocutionGrounding]] = defaultdict(list)
    for ya_node_id, ya_node in node_by_id.items():
        if ya_node["type"] != "YA":
            continue
        source_locutions = [node_id for node_id in incoming[ya_node_id] if node_id in locution_ids]
        target_propositions = [
            node_id for node_id in outgoing[ya_node_id] if node_by_id[node_id]["type"] == "I"
        ]
        for locution_id in source_locutions:
            metadata = locution_metadata[locution_id]
            locution_node = node_by_id[locution_id]
            for proposition_id in target_propositions:
                groundings_by_proposition[proposition_id].append(
                    LocutionGrounding(
                        ya_node_id=ya_node_id,
                        ya_label=ya_node.get("text", ""),
                        raw_locution_id=locution_id,
                        raw_locution_text=locution_node.get("text", ""),
                        speaker_id=_as_id(metadata["personID"]) if metadata.get("personID") is not None else None,
                        speaker=_speaker_from_text(locution_node.get("text", "")),
                        start=metadata.get("start"),
                        timestamp=metadata.get("timestamp") or locution_node.get("timestamp"),
                        chronological_turn=chronological_turn[locution_id],
                    )
                )

    proposition_nodes = [node for node in nodes if node["type"] == "I"]
    proposition_nodes.sort(
        key=lambda node: (
            min(
                (
                    grounding.chronological_turn
                    for grounding in groundings_by_proposition[_as_id(node["nodeID"])]
                ),
                default=10**9,
            ),
            node_index[_as_id(node["nodeID"])],
            int(_as_id(node["nodeID"])),
        )
    )
    propositions: list[CanonicalProposition] = []
    for proposition_order, node in enumerate(proposition_nodes, start=1):
        proposition_id = _as_id(node["nodeID"])
        groundings = sorted(
            groundings_by_proposition[proposition_id],
            key=lambda item: (item.chronological_turn, int(item.raw_locution_id), int(item.ya_node_id)),
        )
        propositions.append(
            CanonicalProposition(
                dialogue_id=dialogue.dialogue_id,
                map_id=path.stem,
                proposition_id=proposition_id,
                proposition_text=node.get("text", ""),
                proposition_order=proposition_order,
                source_node_index=node_index[proposition_id],
                chronological_turn=(
                    min(item.chronological_turn for item in groundings) if groundings else None
                ),
                groundings=groundings,
            )
        )
    proposition_by_id = {item.proposition_id: item for item in propositions}

    relations: list[CanonicalRelation] = []
    for node in nodes:
        if node["type"] not in RELATION_LABEL_BY_NODE_TYPE:
            continue
        relation_id = _as_id(node["nodeID"])
        source_propositions = [
            node_id for node_id in incoming[relation_id] if node_by_id[node_id]["type"] == "I"
        ]
        target_propositions = [
            node_id for node_id in outgoing[relation_id] if node_by_id[node_id]["type"] == "I"
        ]
        is_binary = len(source_propositions) == 1 and len(target_propositions) == 1
        direction = OriginalEdgeDirection.UNKNOWN
        canonical_source: str | None = None
        canonical_target: str | None = None
        is_unambiguous = False
        if is_binary:
            original_source = proposition_by_id[source_propositions[0]]
            original_target = proposition_by_id[target_propositions[0]]
            if original_source.chronological_turn is not None and original_target.chronological_turn is not None:
                if original_source.chronological_turn > original_target.chronological_turn:
                    direction = OriginalEdgeDirection.LATER_TO_EARLIER
                    canonical_source = original_source.proposition_id
                    canonical_target = original_target.proposition_id
                elif original_source.chronological_turn < original_target.chronological_turn:
                    direction = OriginalEdgeDirection.EARLIER_TO_LATER
                    canonical_source = original_target.proposition_id
                    canonical_target = original_source.proposition_id
                else:
                    direction = OriginalEdgeDirection.SAME_TURN
            is_unambiguous = (
                canonical_source is not None
                and original_source.has_unambiguous_grounding
                and original_target.has_unambiguous_grounding
            )
        relations.append(
            CanonicalRelation(
                dialogue_id=dialogue.dialogue_id,
                map_id=path.stem,
                relation_node_id=relation_id,
                relation_node_type=node["type"],
                original_relation_label=node.get("text", ""),
                normalized_relation_label=RELATION_LABEL_BY_NODE_TYPE[node["type"]],
                original_source_proposition_ids=source_propositions,
                original_target_proposition_ids=target_propositions,
                original_incoming_node_ids=incoming[relation_id],
                original_outgoing_node_ids=outgoing[relation_id],
                original_edge_direction=direction,
                canonical_source_proposition_id=canonical_source,
                canonical_target_proposition_id=canonical_target,
                is_binary_direct=is_binary,
                is_unambiguous_binary=is_unambiguous,
            )
        )

    return CanonicalMap(
        dialogue_id=dialogue.dialogue_id,
        dialogue_title=dialogue.dialogue_title,
        dialogue_order=dialogue.dialogue_order,
        map_id=path.stem,
        map_order=dialogue.map_order,
        source_file=path.name,
        raw_node_type_counts=dict(Counter(node["type"] for node in nodes)),
        raw_edge_count=len(payload["edges"]),
        source_locution_count=len(payload["locutions"]),
        chronology_issues=chronology_issues,
        propositions=propositions,
        relations=relations,
    )


def load_canonical_maps(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    dialogues_path: Path = DEFAULT_DIALOGUES_PATH,
) -> list[CanonicalMap]:
    dialogue_by_map = load_dialogue_map(dialogues_path)
    source_paths = {path.stem: path for path in source_dir.glob("*.json")}
    if source_paths.keys() != dialogue_by_map.keys():
        missing_source = sorted(dialogue_by_map.keys() - source_paths.keys())
        missing_metadata = sorted(source_paths.keys() - dialogue_by_map.keys())
        raise ValueError(
            "QT30 source and dialogue metadata do not align: "
            f"missing_source={missing_source}, missing_metadata={missing_metadata}"
        )
    ordered = sorted(
        source_paths,
        key=lambda map_id: (
            dialogue_by_map[map_id].dialogue_order,
            dialogue_by_map[map_id].map_order,
            map_id,
        ),
    )
    return [
        parse_nodeset(source_paths[map_id], dialogue=dialogue_by_map[map_id])
        for map_id in ordered
    ]


def iter_binary_relations(maps: Iterable[CanonicalMap]) -> Iterable[CanonicalRelation]:
    for canonical_map in maps:
        for relation in canonical_map.relations:
            if relation.is_unambiguous_binary:
                yield relation
