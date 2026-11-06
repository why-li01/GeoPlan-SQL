"""Generic utilities for comparing already-fetched result tables.

Callers supply sequences of rows; no database driver is required. The
comparison policy is left to the caller: numeric tolerance and the match
threshold are parameters, not module-level constants.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Sequence, Tuple

_NULL = "__NULL__"


def normalize_cell(value: Any) -> str:
    """Return a stable textual form for one cell."""
    if value is None:
        return _NULL
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        text = format(value.normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value).strip()


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (float, Decimal)) and not isinstance(value, bool)


def _close(left: float, right: float, tolerance: float) -> bool:
    if left == right:
        return True
    if tolerance <= 0.0:
        return False
    denominator = max(abs(left), abs(right))
    return denominator > 0.0 and abs(left - right) / denominator <= tolerance


def _match_numeric(left: List[float], right: List[float], tolerance: float) -> int:
    remaining = list(right)
    matched = 0
    for value in left:
        for index, other in enumerate(remaining):
            if _close(value, other, tolerance):
                matched += 1
                remaining.pop(index)
                break
    return matched


def _split_cells(rows: Sequence[Sequence[Any]]) -> Tuple[List[str], List[float]]:
    exact: List[str] = []
    numeric: List[float] = []
    for row in rows:
        for cell in row:
            if _is_numeric(cell):
                numeric.append(float(cell))
            else:
                exact.append(normalize_cell(cell))
    return exact, numeric


def table_similarity(
    rows_a: Sequence[Sequence[Any]],
    rows_b: Sequence[Sequence[Any]],
    *,
    numeric_tolerance: float = 0.0,
) -> Tuple[float, int, int, int]:
    """Return ``(similarity, matched, cells_a, cells_b)``.

    Similarity is ``matched / min(cells_a, cells_b)``; two empty tables count
    as identical. Tables with different row counts are reported with
    similarity 0.0.
    """
    if len(rows_a) != len(rows_b):
        cells_a = sum(len(row) for row in rows_a)
        cells_b = sum(len(row) for row in rows_b)
        return 0.0, 0, cells_a, cells_b

    exact_a, numeric_a = _split_cells(rows_a)
    exact_b, numeric_b = _split_cells(rows_b)

    matched = sum((Counter(exact_a) & Counter(exact_b)).values())
    matched += _match_numeric(numeric_a, numeric_b, numeric_tolerance)

    cells_a = len(exact_a) + len(numeric_a)
    cells_b = len(exact_b) + len(numeric_b)
    if cells_a == 0 and cells_b == 0:
        return 1.0, 0, 0, 0

    denominator = min(cells_a, cells_b)
    similarity = matched / denominator if denominator else 1.0
    return similarity, matched, cells_a, cells_b


def tables_match(
    rows_a: Sequence[Sequence[Any]],
    rows_b: Sequence[Sequence[Any]],
    *,
    threshold: float = 1.0,
    numeric_tolerance: float = 0.0,
) -> bool:
    """Return whether two result tables reach the requested similarity."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    similarity, _, _, _ = table_similarity(
        rows_a,
        rows_b,
        numeric_tolerance=numeric_tolerance,
    )
    return similarity >= threshold
