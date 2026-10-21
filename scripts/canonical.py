"""Metric B: canonicalised answer score (rules in evaluation/canonicalization_rules.yaml)."""
import re
from collections import Counter
from pathlib import Path

import yaml

from common import ROOT, normalize_answer, is_closed, is_mcq, mcq_pred

RULES = yaml.safe_load(open(ROOT / "evaluation" / "canonicalization_rules.yaml"))
_SYN = dict(RULES["phrase_synonyms"])
for v in list(_SYN.values()):      # identity entries so a target phrase is never re-matched by a shorter key
    _SYN.setdefault(v, v)
_SYN_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in sorted(_SYN, key=len, reverse=True)) + r")\b")
_NUM = RULES["number_words"]


def _plural(tok):
    if len(tok) <= 3 or tok.endswith(("ss", "us", "is")) or tok.isdigit():
        return tok
    if tok.endswith("ies") and RULES["plural"]["ies_to_y"]:
        return tok[:-3] + "y"
    if tok.endswith("s") and RULES["plural"]["strip_trailing_s"]:
        return tok[:-1]
    return tok


def canonicalize(s):
    s = normalize_answer(s)
    s = _SYN_RE.sub(lambda m: _SYN[m.group(0)], s)   # single pass, longest match first
    toks = [_NUM.get(t, t) for t in s.split()]
    toks = [_plural(t) for t in toks]
    return " ".join(toks)


def closed_pred_canonical(pred, question):
    p = canonicalize(pred)
    toks = p.split()
    if toks and toks[0] in {"yes", "no"}:
        return toks[0]
    if p in {canonicalize(x) for x in RULES["closed_questions"]["yes_equivalents"]}:
        return "yes"
    if p in {canonicalize(x) for x in RULES["closed_questions"]["no_equivalents"]}:
        return "no"
    q = normalize_answer(question)
    for rule in RULES["closed_questions"]["conditional"]:
        if any(w in q for w in rule["if_question_contains"]):
            m = {canonicalize(k): v for k, v in rule["map"].items()}
            if p in m:
                return m[p]
    return None


def score_canonical(pred, ref, question):
    if is_mcq(ref):
        return float(mcq_pred(pred) == normalize_answer(ref))
    if is_closed(ref):
        return float(closed_pred_canonical(pred, question) == normalize_answer(ref))
    p, r = canonicalize(pred).split(), canonicalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)
