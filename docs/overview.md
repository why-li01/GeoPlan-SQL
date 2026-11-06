# Overview

GeoPlan-SQL is a research project on spatial Text-to-SQL for PostGIS.

This repository is a minimal preview. It makes the project's scope public while
the paper is under review and contains only generic utilities:

- comparison of already-fetched result tables,
- selection among candidate results by agreement,
- execution-accuracy and basic statistical helpers,
- a generic QLoRA supervised-fine-tuning shell.

What is **not** here: the method, prompts, training recipe, dataset, model
checkpoints, and prediction logs. These will be released upon paper acceptance.

Nothing in this repository should be read as a description of the paper's
approach. The utilities make no claim about any particular technique, and no
reported result can be reproduced from this repository alone.
