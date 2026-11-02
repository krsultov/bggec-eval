"""Edit extraction, typography normalisation and F0.5 with bootstrap intervals."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np

B = 5000             # bootstrap resamples
SEED = 20261003


def tok(text: str) -> list[str]:
    return text.split()


def extract_edits(src: list[str], tgt: list[str]) -> set:
    """Edits that turn src into tgt: (start, end, replacement tokens).
    An insertion has start == end."""
    edits = set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=src, b=tgt, autojunk=False).get_opcodes():
        if tag != "equal":
            edits.add((i1, i2, tuple(tgt[j1:j2])))
    return edits


def prf(tp: int, n_pred: int, n_gold: int, beta: float = 0.5):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0
    return p, r, f


# Typography. Quote, dash and ellipsis glyphs and spacing around punctuation are
# mapped to one form, so that edits which only change them are not counted as
# grammar edits.
_GLYPH = str.maketrans({
    "„": '"', "“": '"', "”": '"', "«": '"', "»": '"', "‟": '"', "″": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "–": "-", "—": "-", "―": "-", "‑": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ",
})
_LETTER = "A-Za-zА-Яа-яЁёѝЍ"
_DOTS = re.compile(r"\.{2,}")
_MULTI_PUNCT = re.compile(r"([!?])\1+")
_SP_BEFORE_PUNCT = re.compile(r"\s+([,;:.!?])")
_SP_AROUND_DASH = re.compile(r"\s*-\s*")
_SP_AFTER_PUNCT = re.compile(r"([,;:!?])(?=[" + _LETTER + "])")
_SP_AFTER_DOTS = re.compile(r"\.\.\.(?=[" + _LETTER + "])")
_QUOTE_IN = re.compile(r'(?:^|(?<=[\s(]))"\s+')
_QUOTE_OUT = re.compile(r'\s+"(?=[\s,;:.!?)]|$)')
_MULTI_SP = re.compile(r"[ \t]+")
_TYPO_CHARS = re.compile(r"[„“”«»\"'‘’–—\-…\.\s]")


def typonorm(text: str) -> str:
    t = text.translate(_GLYPH)
    t = _DOTS.sub("...", t)
    t = _MULTI_PUNCT.sub(r"\1", t)
    t = _SP_BEFORE_PUNCT.sub(r"\1", t)
    t = _SP_AROUND_DASH.sub("-", t)
    t = _SP_AFTER_PUNCT.sub(r"\1 ", t)
    t = _SP_AFTER_DOTS.sub("... ", t)
    t = _QUOTE_IN.sub('"', t)
    t = _QUOTE_OUT.sub('"', t)
    t = _MULTI_SP.sub(" ", t)
    return t.strip()


def edit_is_typographic(src_span: list[str], repl: list[str]) -> bool:
    """True when the edit only changes quote/dash/ellipsis glyphs or spacing.
    Adding or removing a sentence-final period is not typographic."""
    a, b = " ".join(src_span), " ".join(repl)
    if a == b or _TYPO_CHARS.sub("", a) != _TYPO_CHARS.sub("", b):
        return False
    if a.count(".") != b.count(".") and "..." not in a and "..." not in b:
        return False
    return True


# Rulers:
#   single     one reference
#   multi      best-matching reference
#   grammar    typography normalised first, one reference
#   grammar-multi   typography normalised first, best-matching reference
#   typography only typographic edits, one reference
RULERS = ("single", "multi", "grammar", "grammar-multi", "typography")


def _counts(src, ref, hyp, typo_only):
    pred, gold = extract_edits(src, hyp), extract_edits(src, ref)
    if typo_only:
        pred = {e for e in pred if edit_is_typographic(src[e[0]:e[1]], list(e[2]))}
        gold = {e for e in gold if edit_is_typographic(src[e[0]:e[1]], list(e[2]))}
    tp = len(pred & gold)
    return tp, len(pred) - tp, len(gold) - tp


def sentence_counts(row: dict, hyp: str, ruler: str):
    """(tp, fp, fn) for one sentence."""
    refs = row["references"] if ruler in ("multi", "grammar-multi") else row["references"][:1]
    norm = typonorm if ruler.startswith("grammar") else (lambda s: s)
    src, h = tok(norm(row["source"])), tok(norm(hyp))
    best = None
    for r in refs:
        c = _counts(src, tok(norm(r)), h, ruler == "typography")
        key = (prf(c[0], c[0] + c[1], c[0] + c[2])[2], c[0], -(c[0] + c[2]))
        if best is None or key > best[0]:
            best = (key, c)
    return best[1]


def _prf_arr(a):
    tp, fp, fn = a.sum(axis=0)
    return prf(int(tp), int(tp + fp), int(tp + fn))


def _ci(values):
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def bootstrap(arr: np.ndarray) -> dict:
    """arr: per-sentence (tp, fp, fn). Point estimates and 95% intervals."""
    tp, fp, fn = arr.sum(axis=0)
    p, r, f = prf(int(tp), int(tp + fp), int(tp + fn))
    out = {"P": round(p, 4), "R": round(r, 4), "F0.5": round(f, 4),
           "tp": int(tp), "pred": int(tp + fp), "gold": int(tp + fn)}
    idx = np.random.default_rng(SEED).integers(0, len(arr), size=(B, len(arr)))
    boots = np.array([_prf_arr(arr[i]) for i in idx])
    for j, k in enumerate(("P", "R", "F0.5")):
        out[k + "_ci"] = _ci(boots[:, j])
    return out


def rate_ci(flags) -> dict:
    x = np.asarray(flags, dtype=float)
    rng = np.random.default_rng(SEED)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return {"rate": round(float(x.mean()), 4), "n": int(len(x)), "k": int(x.sum()), "ci": _ci(bs)}
