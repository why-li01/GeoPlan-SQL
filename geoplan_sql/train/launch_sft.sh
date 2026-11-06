#!/usr/bin/env bash
# Thin wrapper around geoplan_sql.train.train_sft.
# All arguments are forwarded unchanged; no defaults are applied here.
# Run `python -m geoplan_sql.train.train_sft --help` for the argument list.

set -euo pipefail

exec python -m geoplan_sql.train.train_sft "$@"
