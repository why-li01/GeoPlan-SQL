"""Group candidate results by agreement and select the largest group."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .similarity import tables_match


@dataclass
class CandidateExecution:
    """One SQL candidate together with its already-fetched result table."""

    sql: str
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    ok: bool = True
    error_message: str = ""


def is_votable(candidate: CandidateExecution) -> bool:
    """A candidate participates if it succeeded and returned at least one row."""
    return bool(candidate.ok and candidate.rows)


def filter_votable(
    candidates: Sequence[CandidateExecution],
) -> List[Tuple[int, CandidateExecution]]:
    return [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if is_votable(candidate)
    ]


def cluster_by_result(
    candidates: Sequence[CandidateExecution],
    *,
    threshold: float = 1.0,
) -> List[Tuple[str, List[int], List[str]]]:
    """Group votable candidates whose result tables match.

    Grouping uses connected components, so a group need not be transitive.
    Returns ``(label, indices, sqls)`` sorted by group size (descending) and
    then by the smallest candidate index.
    """
    valid = [
        index for index, candidate in enumerate(candidates) if is_votable(candidate)
    ]
    if not valid:
        return []

    parent = {index: index for index in valid}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for position, left in enumerate(valid):
        for right in valid[position + 1 :]:
            if tables_match(
                candidates[left].rows,
                candidates[right].rows,
                threshold=threshold,
            ):
                union(left, right)

    groups: Dict[int, List[int]] = defaultdict(list)
    for index in valid:
        groups[find(index)].append(index)

    clustered: List[Tuple[str, List[int], List[str]]] = []
    for root, indices in groups.items():
        indices = sorted(indices)
        clustered.append(
            (f"group:{root}", indices, [candidates[index].sql for index in indices])
        )
    clustered.sort(key=lambda item: (-len(item[1]), item[1][0]))
    return clustered


def pick_by_majority(
    candidates: Sequence[CandidateExecution],
    *,
    threshold: float = 1.0,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Select the SQL of the largest agreeing group.

    Ties are broken deterministically in favour of the group that contains the
    earliest candidate. Returns ``(selected_sql_or_None, info)``.
    """
    groups = cluster_by_result(candidates, threshold=threshold)
    info: Dict[str, Any] = {
        "threshold": threshold,
        "groups": [
            {"count": len(indices), "indices": [index + 1 for index in indices]}
            for _, indices, _ in groups
        ],
        "selected_group": None,
    }
    if not groups:
        return None, info

    _, indices, sqls = groups[0]
    info["selected_group"] = {
        "count": len(indices),
        "indices": [index + 1 for index in indices],
    }
    return sqls[0], info
