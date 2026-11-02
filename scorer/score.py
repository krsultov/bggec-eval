"""Score system outputs on the test sets.

    python scorer/score.py                      # the deployed system's outputs in outputs/
    python scorer/score.py --outputs my-outputs # your own outputs, same file names

Writes results.json and results.md to --out (default: results/).

For every set: single-reference, multi-reference and grammar-only F0.5 with 95%
bootstrap intervals. For the web sets, also the two-judge scores and the harm on
clean sentences. These use the judge labels in judges/, so they are computed
only when every edit in the outputs has a label.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from metrics import (B, RULERS, SEED, bootstrap, edit_is_typographic, extract_edits, rate_ci,
                     sentence_counts, tok)

ROOT = Path(__file__).resolve().parents[1]
SETS = ("web-dev", "web-1", "web-2", "klouchek", "synthetic")
WEB = ("web-dev", "web-1", "web-2")
POOLED = ("web-dev", "web-1")            # the paper pools these two
RANK = {"WRONG": 0, "ACCEPTABLE": 1, "CORRECT": 2}


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def edit_key(set_name, sid, start, end, repl):
    return (set_name, sid, start, end, tuple(repl))


def label_key(r):
    return edit_key(r["set"], r["id"], r["start"], r["end"], r["replacement"].split())


# ---------------------------------------------------------------- reference-based scores
def reference_scores(rows, hyps):
    return {ru: np.array([sentence_counts(r, hyps[r["id"]], ru) for r in rows]) for ru in RULERS}


# ---------------------------------------------------------------- two-judge scores
def resolve_system(e):
    """Strict: the lowest label of the two judges wins. Lenient: the highest.
    When BgGPT's two passes disagree (MIXED), both passes count."""
    labs = [e["claude"]]
    if e.get("bggpt"):
        labs += [e["bggpt_order1"], e["bggpt_order2"]] if e["bggpt"] == "MIXED" else [e["bggpt"]]
    labs = [x for x in labs if x in RANK]
    return min(labs, key=RANK.get), max(labs, key=RANK.get)


def resolve_missed(e):
    """Strict: NECESSARY if either judge says so. Lenient: only if both do.
    A MIXED BgGPT verdict is an abstention."""
    labs = [e["claude"]]
    if e.get("bggpt") in ("NECESSARY", "OPTIONAL", "REF_WRONG"):
        labs.append(e["bggpt"])
    strict = "NECESSARY" if "NECESSARY" in labs else "OPTIONAL"
    lenient = "NECESSARY" if all(x == "NECESSARY" for x in labs) else "OPTIONAL"
    return strict, lenient


def bggpt_only(e):
    if e["bggpt"] in RANK:
        return e["bggpt"]
    if e.get("bggpt_order1") in RANK and e.get("bggpt_order2") in RANK:
        return min(e["bggpt_order1"], e["bggpt_order2"], key=RANK.get)
    return None


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def agreement(sys_edits, missed):
    both = [e for e in sys_edits if e["claude"] and e.get("bggpt") in RANK]
    a, b = [e["claude"] for e in both], [e["bggpt"] for e in both]
    out = {"system_edits": {
        "n": len(both),
        "kappa_3way": kappa(a, b),
        "kappa_wrong_vs_not": kappa([x == "WRONG" for x in a], [x == "WRONG" for x in b]),
        "raw_agreement_3way": round(sum(x == y for x, y in zip(a, b)) / len(a), 4),
        "raw_agreement_wrong_vs_not": round(sum((x == "WRONG") == (y == "WRONG") for x, y in zip(a, b)) / len(a), 4),
        "bggpt_mixed_rate": round(sum(e.get("bggpt") == "MIXED" for e in sys_edits if e.get("bggpt"))
                                  / max(1, sum(1 for e in sys_edits if e.get("bggpt"))), 4)}}
    both = [e for e in missed if e["claude"] and e.get("bggpt") and e["bggpt"] not in ("MIXED", "error")]
    if both:
        a = [e["claude"] != "NECESSARY" for e in both]
        b = [e["bggpt"] != "NECESSARY" for e in both]
        out["missed_edits"] = {"n": len(both), "kappa_necessary_vs_not": kappa(a, b),
                               "raw_agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 4)}
    return out


def judge_metrics(per, ids):
    """per: id -> [correct, acceptable, wrong, missed necessary]."""
    arr = np.array([per.get(i, [0, 0, 0, 0]) for i in ids], dtype=np.int64)

    def m(a):
        c, acc, w, miss = a.sum(axis=0)
        n = c + acc + w
        p = (c + acc) / n if n else 0.0
        r = c / (c + miss) if (c + miss) else 0.0
        f = 1.25 * p * r / (0.25 * p + r) if (0.25 * p + r) else 0.0
        return p, r, f

    idx = np.random.default_rng(SEED).integers(0, len(arr), size=(B, len(arr)))
    boots = np.array([m(arr[i]) for i in idx])
    p0 = m(arr)
    out = {"edits": int(arr[:, :3].sum()), "correct": int(arr[:, 0].sum()),
           "acceptable": int(arr[:, 1].sum()), "wrong": int(arr[:, 2].sum()),
           "missed_necessary": int(arr[:, 3].sum())}
    for j, k in enumerate(("P", "R", "F0.5")):
        out[k] = round(float(p0[j]), 4)
        out[k + "_ci"] = [round(float(np.percentile(boots[:, j], 2.5)), 4),
                          round(float(np.percentile(boots[:, j], 97.5)), 4)]
    return out


def judge_counts(sys_edits, missed, lab_sys, lab_missed, no_typography):
    per = defaultdict(lambda: [0, 0, 0, 0])
    col = {"CORRECT": 0, "ACCEPTABLE": 1, "WRONG": 2}
    for e in sys_edits:
        if no_typography and e["category"] == "typography":
            continue
        l = lab_sys.get(label_key(e))
        if l in col:
            per[(e["set"], e["id"])][col[l]] += 1
    for e in missed:
        if no_typography and e["category"] == "typography":
            continue
        if lab_missed.get(label_key(e)) == "NECESSARY":
            per[(e["set"], e["id"])][3] += 1
    return per


def judge_scores(groups, sys_edits, missed):
    """groups: name -> list of (set, id). Scores per judge and resolved."""
    s = [e for e in sys_edits if e["claude"]]
    mi = [e for e in missed if e["claude"]]
    strict_s, lenient_s = {}, {}
    for e in s:
        strict_s[label_key(e)], lenient_s[label_key(e)] = resolve_system(e)
    strict_m, lenient_m = {}, {}
    for e in mi:
        strict_m[label_key(e)], lenient_m[label_key(e)] = resolve_missed(e)
    labels = {
        "claude": ({label_key(e): e["claude"] for e in s}, {label_key(e): e["claude"] for e in mi}),
        "bggpt": ({label_key(e): bggpt_only(e) for e in s if e.get("bggpt") and bggpt_only(e)},
                  {label_key(e): e["bggpt"] for e in mi if e.get("bggpt")}),
        "strict": (strict_s, strict_m),
        "lenient": (lenient_s, lenient_m),
    }
    out = {}
    for name, (ls, lm) in labels.items():
        for no_typo in (False, True):
            per = judge_counts(sys_edits, missed, ls, lm, no_typo)
            tag = name + (", no typography" if no_typo else "")
            out[tag] = {g: judge_metrics(per, ids) for g, ids in groups.items()}
    return out, strict_s


def clean_text(rows, hyps, set_name, strict_s):
    clean = [r for r in rows if r["source"] == r["references"][0]]
    if not clean:
        return None
    touched, harmed, typo_only = [], [], 0
    for r in clean:
        src, h = tok(r["source"]), hyps[r["id"]]
        ed = extract_edits(src, tok(h))
        touched.append(h != r["source"])
        harmed.append(any(strict_s.get(edit_key(set_name, r["id"], s, e, rep)) == "WRONG" for s, e, rep in ed))
        if h != r["source"] and all(edit_is_typographic(src[s:e], list(rep)) for s, e, rep in ed):
            typo_only += 1
    return {"n_clean": len(clean), "touched": rate_ci(touched), "harmed": rate_ci(harmed),
            "touched_typography_only": typo_only}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, hyps, ref, res = {}, {}, {}, {"reference": {}, "judges": {}, "clean_text": {}}
    for name in SETS:
        p = Path(a.outputs) / f"{name}.jsonl"
        if not p.exists():
            print(f"{name}: no outputs, skipped")
            continue
        data[name] = jsonl(ROOT / "data" / f"{name}.jsonl")
        hyps[name] = {r["id"]: r["output"] for r in jsonl(p)}
        ref[name] = reference_scores(data[name], hyps[name])
        res["reference"][name] = {ru: bootstrap(ref[name][ru]) for ru in RULERS}
    if all(s in ref for s in POOLED):
        res["reference"]["web-dev + web-1"] = {
            ru: bootstrap(np.concatenate([ref[s][ru] for s in POOLED])) for ru in RULERS}

    # two-judge scores: only when every edit in the outputs has a judge label
    sys_edits, missed, judged = [], [], []
    for name in WEB:
        if name not in data:
            continue
        s = [dict(e, set=name) for e in jsonl(ROOT / "judges" / name / "system-edits.jsonl")]
        m = [dict(e, set=name) for e in jsonl(ROOT / "judges" / name / "missed-edits.jsonl")]
        have = {label_key(e) for e in s}
        need = {edit_key(name, r["id"], *e) for r in data[name]
                for e in extract_edits(tok(r["source"]), tok(hyps[name][r["id"]]))}
        if need - have:
            print(f"{name}: {len(need - have)} edits have no judge label, two-judge scores skipped")
            continue
        sys_edits += s
        missed += m
        judged.append(name)
    if judged:
        groups = {n: [(n, r["id"]) for r in data[n]] for n in judged}
        if all(s in judged for s in POOLED):
            groups["web-dev + web-1"] = [(n, r["id"]) for n in POOLED for r in data[n]]
        res["judges"]["scores"], strict_s = judge_scores(groups, sys_edits, missed)
        res["judges"]["agreement"] = {}
        for g in groups:
            members = [g] if g in judged else list(POOLED)
            res["judges"]["agreement"][g] = agreement([e for e in sys_edits if e["set"] in members],
                                                      [e for e in missed if e["set"] in members])
        cat = defaultdict(Counter)
        for e in sys_edits:
            if label_key(e) in strict_s:
                cat[e["category"]][strict_s[label_key(e)]] += 1
        res["judges"]["strict_labels_by_category"] = {k: dict(v) for k, v in sorted(cat.items())}
        for n in judged:
            c = clean_text(data[n], hyps[n], n, strict_s)
            if c:
                res["clean_text"][n] = c

    (out_dir / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                          encoding="utf-8", newline="\n")
    (out_dir / "results.md").write_text(render(res), encoding="utf-8", newline="\n")
    print(f"wrote {out_dir / 'results.md'}")


def _f(d, k):
    return f"{d[k]:.3f} [{d[k + '_ci'][0]:.3f}, {d[k + '_ci'][1]:.3f}]"


def render(res):
    L = ["# Results", "", "95% bootstrap intervals over sentences.", "",
         "## Reference-based F0.5", "",
         "| set | ruler | P | R | F0.5 |", "|---|---|---|---|---|"]
    for s, d in res["reference"].items():
        for ru in RULERS:
            L.append(f"| {s} | {ru} | {_f(d[ru], 'P')} | {_f(d[ru], 'R')} | {_f(d[ru], 'F0.5')} |")
    if res["judges"]:
        L += ["", "## Two-judge F0.5", "",
              "| labels | set | edits | correct / acceptable / wrong | missed necessary | P | R | F0.5 |",
              "|---|---|---|---|---|---|---|---|"]
        for lab, d in res["judges"]["scores"].items():
            for g, m in d.items():
                L.append(f"| {lab} | {g} | {m['edits']} | {m['correct']} / {m['acceptable']} / {m['wrong']} | "
                         f"{m['missed_necessary']} | {_f(m, 'P')} | {_f(m, 'R')} | {_f(m, 'F0.5')} |")
        L += ["", "## Judge agreement on system edits", "",
              "| set | n | kappa (3 labels) | kappa (wrong vs not) | raw agreement (wrong vs not) |",
              "|---|---|---|---|---|"]
        for g, a in res["judges"]["agreement"].items():
            x = a["system_edits"]
            L.append(f"| {g} | {x['n']} | {x['kappa_3way']} | {x['kappa_wrong_vs_not']} | {x['raw_agreement_wrong_vs_not']} |")
        L += ["", "## Clean sentences", "",
              "| set | clean sentences | changed | harmed (strict WRONG) | changed, typography only |",
              "|---|---|---|---|---|"]
        for s, d in res["clean_text"].items():
            t, h = d["touched"], d["harmed"]
            L.append(f"| {s} | {d['n_clean']} | {t['k']} | {h['k']} | {d['touched_typography_only']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
