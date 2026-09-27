"""
Pair features shared by training and test (ONE function -> no train/test skew).

Input : a pandas DataFrame per bucket with columns
        s1_id, cand_id, source, mask, n1 (S1 name), a1 (S1 address), h1, p1 (S1 hnum/postcode),
        n2, a2, h2, p2 (vendor)
Output: float32 feature matrix + the list of feature names.
No country feature on purpose: France (test only) has no labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

BLOCKERS = ["compact", "sorted", "prefix", "suffix", "phonetic", "rare1", "rare2",
            "addr", "addrname", "postname", "postrare", "apair", "nameaddr", "numname"]

LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co",
         "company", "plc", "pvt", "private", "lp", "llp", "pllc", "pc", "pa", "and", "the",
         "of", "sarl", "sas", "sasu", "eurl", "sci", "snc", "enterprises", "partners",
         "sri", "shri", "shree"}

# SQL used by the scripts to fetch one bucket of pairs with both sides' text
PAIR_SQL = """
SELECT c.s1_id, c.cand_id, c.source, c.mask,
       coalesce(s.name_norm,'') AS n1, coalesce(s.address_norm,'') AS a1, coalesce(CAST(s.hnum AS VARCHAR),'') AS h1, coalesce(CAST(s.postcode AS VARCHAR),'') AS p1,
       coalesce(v.name_norm,'') AS n2, coalesce(v.address_norm,'') AS a2, coalesce(CAST(v.hnum AS VARCHAR),'') AS h2, coalesce(CAST(v.postcode AS VARCHAR),'') AS p2
FROM (SELECT * FROM read_parquet('{cands}') WHERE hash(s1_id) % {nb} = {k}) c
JOIN read_parquet('{w}/keys_s1.parquet') s ON c.s1_id = s.entity_id
JOIN read_parquet(['{w}/keys_s2.parquet', '{w}/keys_s3.parquet']) v ON c.cand_id = v.entity_id
"""


def _core(s: str) -> str:
    return " ".join(t for t in s.split() if t not in LEGAL)


def _pd(scorer, a, b, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    n1 = df["n1"].tolist(); n2 = df["n2"].tolist()
    a1 = df["a1"].tolist(); a2 = df["a2"].tolist()
    c1 = [_core(x) for x in n1]; c2 = [_core(x) for x in n2]
    F: dict[str, np.ndarray] = {}

    # ---- name ----
    F["n_ratio"] = _pd(fuzz.ratio, n1, n2)
    F["n_tset"] = _pd(fuzz.token_set_ratio, n1, n2)
    F["n_tsort"] = _pd(fuzz.token_sort_ratio, n1, n2)
    F["n_partial"] = _pd(fuzz.partial_ratio, n1, n2)
    F["n_jw"] = _pd(JaroWinkler.normalized_similarity, n1, n2)
    F["core_ratio"] = _pd(fuzz.ratio, c1, c2)
    F["core_tset"] = _pd(fuzz.token_set_ratio, c1, c2)
    F["core_partial"] = _pd(fuzz.partial_ratio, c1, c2)
    cc1 = [x.replace(" ", "") for x in c1]; cc2 = [x.replace(" ", "") for x in c2]
    F["compact_eq"] = np.array([a == b and a != "" for a, b in zip(cc1, cc2)], np.float32)
    t1 = [set(x.split()) for x in c1]; t2 = [set(x.split()) for x in c2]
    F["core_subset"] = np.array([(bool(a) and bool(b)) and (a <= b or b <= a) for a, b in zip(t1, t2)], np.float32)
    F["core_jacc"] = np.array([len(a & b) / len(a | b) if (a | b) else 0 for a, b in zip(t1, t2)], np.float32)
    F["first_tok_eq"] = np.array([(x.split()[:1] == y.split()[:1]) and x != "" for x, y in zip(c1, c2)], np.float32)
    F["ntok1"] = np.array([len(x) for x in t1], np.float32)
    F["ntok2"] = np.array([len(x) for x in t2], np.float32)
    F["len_diff"] = np.abs(np.array([len(x) for x in n1], np.float32) - np.array([len(x) for x in n2], np.float32))

    # ---- address ----
    a_empty = np.array([x == "" for x in a2], bool)
    F["v_addr_empty"] = a_empty.astype(np.float32)
    at = _pd(fuzz.token_set_ratio, a1, a2); at[a_empty] = np.nan
    ar = _pd(fuzz.ratio, a1, a2); ar[a_empty] = np.nan
    ap = _pd(fuzz.partial_ratio, a1, a2); ap[a_empty] = np.nan
    F["a_tset"], F["a_ratio"], F["a_partial"] = at, ar, ap
    h1 = df["h1"].astype(str).to_numpy(); h2 = df["h2"].astype(str).to_numpy()
    hb = (h1 != "") & (h2 != "")
    F["hnum_eq"] = (hb & (h1 == h2)).astype(np.float32)
    F["hnum_conflict"] = (hb & (h1 != h2)).astype(np.float32)
    F["hnum_missing"] = (~hb).astype(np.float32)
    p1 = df["p1"].astype(str).to_numpy(); p2 = df["p2"].astype(str).to_numpy()
    pb = (p1 != "") & (p2 != "")
    F["post_eq"] = (pb & (p1 == p2)).astype(np.float32)
    F["post_conflict"] = (pb & (p1 != p2)).astype(np.float32)
    # numbers shared anywhere in the address (house numbers are noisy)
    d1 = [{t for t in x.split() if t.isdigit()} for x in a1]
    d2 = [{t for t in x.split() if t.isdigit()} for x in a2]
    F["num_jacc"] = np.array([len(a & b) / len(a | b) if (a | b) else np.nan for a, b in zip(d1, d2)], np.float32)

    # ---- blocking evidence ----
    mask = df["mask"].to_numpy(np.int64)
    for i, b in enumerate(BLOCKERS):
        F[f"blk_{b}"] = ((mask >> i) & 1).astype(np.float32)
    F["blk_count"] = np.array([bin(m).count("1") for m in mask], np.float32)
    F["source"] = df["source"].to_numpy(np.float32)

    # ---- per-S1 context (all candidates of an S1 are in the same bucket) ----
    tmp = pd.DataFrame({"s1": df["s1_id"].to_numpy(), "n": F["n_tset"], "c": F["core_ratio"],
                        "a": np.nan_to_num(F["a_tset"], nan=-1.0)})
    g = tmp.groupby("s1", sort=False)
    F["s1_ncand"] = g["n"].transform("size").to_numpy(np.float32)
    for col, name in (("n", "n_tset"), ("c", "core_ratio"), ("a", "a_tset")):
        mx = g[col].transform("max").to_numpy(np.float32)
        F[f"{name}_gap"] = mx - tmp[col].to_numpy(np.float32)
        F[f"{name}_rank"] = g[col].rank(ascending=False, method="min").to_numpy(np.float32)
    # how many candidates of this S1 look very similar (duplicates across vendors)
    F["s1_n_strong"] = g["n"].transform(lambda s: (s >= 90).sum()).to_numpy(np.float32)

    names = list(F)
    X = np.column_stack([F[k].astype(np.float32) for k in names])
    return X, names
