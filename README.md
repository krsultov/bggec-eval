# bggec-eval

Test sets, system outputs and judge labels for the paper

> Krum Sultov and Georgi Balabanov. Better, or Just Closer to the Reference? A Propose-and-Verify System for Bulgarian Grammatical Error Correction. *AI*, 2026.

## Contents

| Path | Contents |
|---|---|
| `data/web-dev.jsonl` | 300 web sentences, with extra validated references |
| `data/web-1.jsonl` | 300 web sentences (200 need no correction) |
| `data/web-2.jsonl` | 300 web sentences (199 need no correction) |
| `data/klouchek.jsonl` | the 673-sentence test split of Klouchek and Batista-Navarro (2024), with extra validated references |
| `data/synthetic.jsonl` | 1,097 treebank sentences with one generated error each |
| `outputs/` | the deployed system's output for every sentence in each set |
| `judges/` | the two judges' labels for the system's edits and the missed reference edits on the web sets |
| `scorer/` | the scoring code |

Each line in `data/` is `{"id", "source", "references"}`. The first reference is the primary one. `synthetic.jsonl` also has `error_type`. Each line in `outputs/` is `{"id", "output"}`.

In `judges/`, `system-edits.jsonl` has one line per edit the system made, labelled CORRECT, ACCEPTABLE or WRONG. `missed-edits.jsonl` has one line per reference edit the system did not make, labelled NECESSARY, OPTIONAL or REF_WRONG. The judges are Claude Fable 5.1 and BgGPT-Gemma-3-27B. BgGPT judged every item twice with the answer options in swapped order; MIXED means the two passes disagreed.

## Scoring

Requires Python 3.10+ and numpy.

```
python scorer/score.py
```

This scores the outputs in `outputs/` and writes `results/results.md` and `results/results.json`. The web-set results match the paper.

To score your own system, write one file per set to a folder, in the same format and with the same file names as `outputs/`, and run:

```
python scorer/score.py --outputs my-outputs
```

The reference-based scores work for any system. The two-judge scores need a label for every edit, so for a new system they are computed only after its new edits have been judged.

## Licences

The code is under the MIT licence (`LICENSE`). The data is under CC BY 4.0, except for the files listed in `LICENSE-DATA`. To cite this work, see `CITATION.cff`.
