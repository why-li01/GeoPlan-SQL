"""Offline demo of result-table comparison and agreement-based selection."""

from __future__ import annotations

from .vote import CandidateExecution, pick_by_majority


def main() -> None:
    candidates = [
        CandidateExecution(
            sql="SELECT name FROM places ORDER BY area DESC LIMIT 1",
            rows=[("Riverside",)],
            columns=["name"],
        ),
        CandidateExecution(
            sql="SELECT p.name FROM places AS p ORDER BY p.area DESC LIMIT 1",
            rows=[("Riverside",)],
            columns=["name"],
        ),
        CandidateExecution(
            sql="SELECT name FROM places ORDER BY area DESC LIMIT 3",
            rows=[("Riverside",), ("Hillcrest",), ("Lakeside",)],
            columns=["name"],
        ),
        CandidateExecution(
            sql="SELECT 1/0",
            ok=False,
            error_message="division by zero",
        ),
        CandidateExecution(
            sql="SELECT name FROM places WHERE false",
            rows=[],
            columns=["name"],
        ),
    ]

    selected, info = pick_by_majority(candidates, threshold=1.0)
    print("Selected SQL:")
    print(selected)
    print("Groups:")
    for group in info["groups"]:
        print("  count={count} indices={indices}".format(**group))


if __name__ == "__main__":
    main()
