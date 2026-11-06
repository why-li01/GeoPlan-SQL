"""Generic table comparison and agreement-based selection."""

from .similarity import normalize_cell, table_similarity, tables_match
from .vote import (
    CandidateExecution,
    cluster_by_result,
    filter_votable,
    is_votable,
    pick_by_majority,
)

__all__ = [
    "CandidateExecution",
    "cluster_by_result",
    "filter_votable",
    "is_votable",
    "normalize_cell",
    "pick_by_majority",
    "table_similarity",
    "tables_match",
]
