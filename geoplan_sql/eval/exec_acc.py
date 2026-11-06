"""Execution-accuracy evaluation over precomputed result tables.

No database connection is required: every record carries the gold and
predicted row tables. The match threshold and numeric tolerance are supplied
by the caller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from geoplan_sql.eval.stats import wilson_interval
from geoplan_sql.voting.similarity import tables_match


def execution_accurate(
    gold_rows: Sequence[Sequence[Any]],
    pred_rows: Sequence[Sequence[Any]],
    *,
    threshold: float = 1.0,
    numeric_tolerance: float = 0.0,
) -> bool:
    """Return whether the predicted table matches the gold table."""
    return tables_match(
        gold_rows,
        pred_rows,
        threshold=threshold,
        numeric_tolerance=numeric_tolerance,
    )


def evaluate_predictions(
    records: Iterable[Dict[str, Any]],
    *,
    threshold: float = 1.0,
    numeric_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate records that provide ``gold_rows`` and ``pred_rows``."""
    details: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    for record in records:
        qid = record.get("id", total)
        gold_rows = record.get("gold_rows")
        pred_rows = record.get("pred_rows")
        if gold_rows is None or pred_rows is None:
            raise ValueError(f"Record {qid}: need gold_rows and pred_rows")

        ok = execution_accurate(
            gold_rows,
            pred_rows,
            threshold=threshold,
            numeric_tolerance=numeric_tolerance,
        )
        total += 1
        correct += int(ok)
        details.append({"id": qid, "correct": ok})

    rate, low, high = wilson_interval(correct, total)
    return {
        "n": total,
        "correct": correct,
        "exec_acc": rate,
        "wilson95": [low, high],
        "threshold": threshold,
        "details": details,
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Offline execution accuracy over JSONL")
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL records with gold_rows and pred_rows",
    )
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--numeric_tolerance", type=float, default=0.0)
    args = parser.parse_args(argv)

    records = _load_jsonl(Path(args.predictions))
    summary = evaluate_predictions(
        records,
        threshold=args.threshold,
        numeric_tolerance=args.numeric_tolerance,
    )
    printable = {key: value for key, value in summary.items() if key != "details"}
    print(json.dumps(printable, indent=2))
    print(f"Per-question: {summary['correct']}/{summary['n']}")


if __name__ == "__main__":
    main()
