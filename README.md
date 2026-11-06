# GeoPlan-SQL (Preview)

Research prototype for **spatial Text-to-SQL** on PostGIS.

This repository is a **preview / evaluation-utilities** release. It records the project's
scope and ships a few *generic* utilities that are independent of our method.
The method description, prompts, training recipe, data, and checkpoints are **not** included.

**Full code and GeoNL2SQL-15K will be released upon paper acceptance.**

## What is included now


| Component | Status |
| --- | --- |
| Generic comparison of already-fetched result tables | available |
| Generic agreement-based selection among candidate results | available |
| Execution-accuracy, confidence-interval, McNemar helpers | available |
| Generic QLoRA SFT shell (bring your own data) | available |
| Method description, prompts, recipe, data, checkpoints | upon paper acceptance |

> Nothing in this repository describes or reproduces the paper's approach.
> Results reported in the paper cannot be reproduced from this repository.

## Quick start

```bash
pip install -r requirements.txt
python -m geoplan_sql.voting.demo_vote
python -m geoplan_sql.eval --predictions examples/toy/sample_preds.jsonl
```

## Repository layout

```text
GeoPlan-SQL/
  README.md
  LICENSE
  requirements.txt
  docs/overview.md
  geoplan_sql/
    voting/     # table comparison + agreement-based selection
    eval/       # execution accuracy, Wilson interval, McNemar
    train/      # generic QLoRA SFT shell (no data)
  examples/toy/ # minimal offline example for the eval helper
```

## Scope

The utilities are generic and make no claim about any particular approach.
They are not a reproduction package. The training shell requires its numerical
hyper-parameters explicitly; no tuned recipe is encoded here.

## License

Source-available for non-commercial academic evaluation only; **not** open
source. See [LICENSE](LICENSE). All rights reserved.

## Citation

```bibtex
@article{geoplan-sql,
  title={GeoPlan-SQL},
  author={Anonymous},
  year={2026},
  note={Code and data to be released upon acceptance}
}
```
