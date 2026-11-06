# Toy example

A tiny synthetic PostGIS schema used by the offline evaluation demo.

| File | Content |
| --- | --- |
| `schema.sql` | Two small PostGIS tables |
| `question.txt` | One natural-language question |
| `gold.sql` | A reference SQL query for that question |
| `sample_preds.jsonl` | Precomputed row tables for the eval demo (no database needed) |

Load `schema.sql` into a local PostGIS instance if you want to run `gold.sql`.
The coordinates and names are synthetic.
