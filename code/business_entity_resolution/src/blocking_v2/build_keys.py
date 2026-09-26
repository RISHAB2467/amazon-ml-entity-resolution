"""
Step 1 of multi-rule blocking: compute blocking keys for S1, S2, S3.

Reads the RAW challenge TSVs in chunks (normalising on the fly) (never all 12.5M rows at once) and writes
one Parquet file per source with the keys every blocker needs.

Two passes:
  pass 1  token document frequency (DF) over all three sources  -> rarity
  pass 2  per-record keys                                       -> keys_s{1,2,3}.parquet

Usage:
  python build_keys.py --s1 dataset/train/train_source1.tsv
      --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv
      --out output/blocking_train
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import time
import unicodedata
from collections import Counter

import jellyfish
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# Column names in your normalized TSVs. Change here if yours differ.
# --------------------------------------------------------------------------
# Raw challenge schema (all three sources, train and test).
COL_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDR = "business_address"
COL_COUNTRY = "country"

# --------------------------------------------------------------------------
# Name canonicalisation (keys only - features will use richer logic later)
# --------------------------------------------------------------------------
# Legal forms stripped from the END of a name (repeatedly).
LEGAL = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "pvt", "private", "lp", "llp", "pllc", "pc", "pa",
    "gmbh", "ag", "sa", "srl", "bv", "nv", "pty", "lllp", "psc", "ltda",
    # India
    "opc", "pvtltd",
    # France (test-only country): SARL, SAS, SASU, EURL, SCI, SNC, "& Cie"
    "sarl", "sas", "sasu", "eurl", "sci", "snc", "selarl", "scp", "scop", "cie",
}
# Legal forms glued onto a compact name ("ashishserviceslimited").
# Longest first; "co" is deliberately excluded (too many false strips).
LEGAL_GLUED = sorted(
    ["incorporated", "corporation", "limited", "company", "pvtltd", "inc",
     "llc", "ltd", "corp", "plc", "pllc", "llp", "sarl", "sasu", "eurl",
     "private", "pvt"],
    key=len, reverse=True)
# Honorific prefixes seen in Indian vendor names ("sri galaxy investment")
HONORIFIC = {"sri", "shri", "shree", "smt", "mr", "mrs", "ms", "messrs"}
# Junk tokens inside addresses ("thiruvananthapuram null")
ADDR_JUNK = {"null", "none", "nan", "na", "nil", "ind", "india", "usa", "us",
             "france"}
ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4",
                 "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
                 "ninth": "9", "tenth": "10"}
_ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th)$")
NAME_CANON = {
    "intl": "international", "svc": "services", "svcs": "services",
    "mgmt": "management", "assoc": "associates", "assocs": "associates",
    "assn": "association", "ctr": "center", "centre": "center",
    "hosp": "hospital", "med": "medical", "univ": "university",
    "dept": "department", "natl": "national", "bros": "brothers",
    "mfg": "manufacturing", "tech": "technology", "techs": "technologies",
    "grp": "group", "&": "and",
}
# English + French function words ("et" = and, "de la" = of the)
STOP = {"the", "and", "of", "a", "an", "et", "de", "la", "le", "les", "du",
        "des", "d", "l", "en", "au", "aux"}
# Digit->letter look-alikes, applied only to tokens mixing letters and digits
# ("re1iable" -> "reliable"). Pure numbers ("7 eleven") are left alone.
LEET = str.maketrans("0134578", "oleastb")

# --------------------------------------------------------------------------
# Address canonicalisation
# --------------------------------------------------------------------------
DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south",
              "east", "west", "northeast", "northwest", "southeast", "southwest"}
UNIT_WORDS = {"unit", "suite", "ste", "apt", "apartment", "fl", "floor", "bldg",
              "building", "room", "rm", "no", "number", "po", "box", "#",
              # India: landmark / premises words ("Shop No 12, Near SBI ATM")
              "near", "nr", "opp", "opposite", "behind", "beside", "next", "to",
              "shop", "plot", "flat", "house", "h", "sector", "block", "ward",
              "gali", "lane", "marg",
              # France: number suffixes and articles ("12 bis rue de la Paix")
              "bis", "ter", "quater", "de", "la", "le", "les", "du", "des",
              "d", "l", "of", "the"}
# Street-type words are skipped when picking the street NAME token, so
# "12 rue de la paix" -> "paix", "1520 dobson road" -> "dobson".
STREET_TYPES = {"street", "st", "road", "rd", "drive", "dr", "avenue", "ave",
                "av", "boulevard", "blvd", "bd", "court", "ct", "place", "pl",
                "highway", "hwy", "parkway", "pkwy", "circle", "cir", "square",
                "sq", "terrace", "ter", "trail", "trl", "way", "ln",
                "rue", "chemin", "allee", "impasse", "route", "quai", "cours",
                "voie", "passage", "sentier", "rte", "che", "imp", "all"}
STREET_CANON = {
    "street": "st", "road": "rd", "drive": "dr", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "lane": "ln", "court": "ct", "place": "pl",
    "highway": "hwy", "parkway": "pkwy", "circle": "cir", "square": "sq",
    "terrace": "ter", "trail": "trl", "way": "way", "mount": "mt",
    "saint": "st", "fort": "ft", "first": "1st", "second": "2nd", "third": "3rd",
}

PREFIX_LEN = 8          # length of prefix / suffix typo keys


def canon_token(t: str) -> str:
    if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
        t = t.translate(LEET)
    return NAME_CANON.get(t, t)


def singular(t: str) -> str:
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def strip_glued_legal(s: str) -> str:
    changed = True
    while changed:
        changed = False
        for sfx in LEGAL_GLUED:
            if s.endswith(sfx) and len(s) - len(sfx) >= 4:
                s = s[: -len(sfx)]
                changed = True
                break
    return s


def unglue_legal(t: str) -> str | None:
    """'academylimited' -> 'academy'; 'limitedlimited' / 'pvtltd' -> None
    (token is legal-form only)."""
    if t in LEGAL:
        return None
    s = t
    changed = True
    while changed:
        changed = False
        for sfx in LEGAL_GLUED:
            if s.endswith(sfx) and len(s) - len(sfx) >= 3:
                s = s[: -len(sfx)]
                changed = True
                break
    return None if s in LEGAL else s


def name_tokens(name: str) -> list[str]:
    """Canonical core-name tokens.

    Vendors shuffle token order ("private laxmi foods limited"), glue legal
    forms ("limitedlimited"), prepend honorifics ("sri") and duplicate tokens
    ("a1pha alpha"). So: leet-fix, expand abbreviations, drop stop words,
    honorifics, single letters and legal forms ANYWHERE (unless nothing else
    is left), dedupe keeping order.
    """
    if not name:
        return []
    raw = [canon_token(t) for t in fold(name).split()]
    raw = [t for t in raw if t and t not in STOP and t not in HONORIFIC
           and t != "dba" and len(t) > 1]
    core, seen = [], set()
    for t in raw:
        u = unglue_legal(t)
        if u and u not in seen:
            seen.add(u)
            core.append(u)
    if core:
        return core
    return list(dict.fromkeys(raw))


def dba_parts(name: str) -> list[str]:
    """'solquo dba delhi producer' -> ['solquo', 'delhi producer']."""
    if not name or " dba " not in f" {name} ":
        return []
    return [p.strip() for p in f" {name} ".split(" dba ") if p.strip()]


def fold(s: str) -> str:
    """Strip accents: "général" -> "general" (French test records)."""
    if not s or s.isascii():
        return s or ""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


_PIN_SPLIT = re.compile(r"\b(\d{3}) (\d{3})\b")   # Indian PIN "411 001"


def parse_address(addr: str, num_col: str):
    """Return (tokens, house_number, postcode).

    postcode  = LAST 5-digit (US ZIP / French code postal) or 6-digit (Indian
                PIN) token that is not the first token.
    house no. = the provided address_number unless it is the postcode, else the
                first digit token (<= 5 digits) that is not the postcode.
    Without this, PIN/ZIP codes get used as house numbers.
    """
    if not addr:
        return [], None, None
    a = _PIN_SPLIT.sub(r"\1\2", fold(addr))
    toks = []
    for t in a.split():
        if t in ADDR_JUNK:
            continue
        m = _ORDINAL.match(t)          # "217th" / corrupted "217nd" -> "217"
        if m:
            t = m.group(1)
        toks.append(ORDINAL_WORDS.get(t, t))   # "sixth" -> "6"
    post_i, post = None, None
    for i in range(len(toks) - 1, 0, -1):
        t = toks[i]
        if t.isdigit() and len(t) in (5, 6):
            post_i, post = i, t
            break
    hn = None
    if num_col and num_col.strip() and num_col.strip() != post:
        hn = num_col.strip().lstrip("0") or None
    if hn is None:
        for i, t in enumerate(toks):
            if i != post_i and t.isdigit() and len(t) <= 5:
                hn = t.lstrip("0") or None
                break
    return toks, hn, post


def street_token(toks: list[str], hn: str | None) -> str | None:
    """First street NAME token after the house number (skips types,
    directions, unit words, articles)."""
    if not toks:
        return None
    start = 0
    if hn:
        for i, t in enumerate(toks):
            if t.lstrip("0") == hn:
                start = i + 1
                break
    for t in toks[start:]:
        if (t.isdigit() or t in DIRECTIONS or t in UNIT_WORDS
                or t in STREET_TYPES or len(t) < 2):
            continue
        return STREET_CANON.get(t, t)
    return None


def addr_words(toks: list[str]) -> set[str]:
    """Alphabetic address words usable for rarity keys (numbers are too
    noisy: 12.6% of true pairs have conflicting house numbers)."""
    return {t for t in toks if len(t) >= 3 and t.isalpha()}


# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r" +")


def norm_text(s: str) -> str:
    """lowercase, accents folded, "&" -> "and", punctuation -> space."""
    if not s:
        return ""
    s = fold(s.lower()).replace("&", " and ")
    s = s.replace("'", "")            # "macy's" -> "macys", "l'etoile" -> "letoile"
    return _SPACES.sub(" ", _PUNCT.sub(" ", s)).strip()


def read_chunks(path: str, chunksize: int):
    """Raw TSV in chunks; adds normalised name/address columns.
    Raw files are never modified."""
    usecols = [COL_ID, COL_NAME, COL_ADDR, COL_COUNTRY]
    for chunk in pd.read_csv(
            path, sep="\t", dtype=str, usecols=usecols,
            keep_default_na=False, na_values=[], quoting=csv.QUOTE_NONE,
            chunksize=chunksize, on_bad_lines="warn", encoding="utf-8"):
        chunk["_name"] = [norm_text(x) for x in chunk[COL_NAME].values]
        chunk["_addr"] = [norm_text(x) for x in chunk[COL_ADDR].values]
        chunk["_country"] = [norm_text(x) for x in chunk[COL_COUNTRY].values]
        yield chunk


def peak_rss_gb():
    try:
        import resource  # Linux / macOS only
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)
    except ImportError:
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 1024 ** 3, 2)
        except Exception:
            return None


def pass1_df(paths: dict[int, str], chunksize: int) -> tuple[Counter, Counter]:
    df, adf = Counter(), Counter()
    for src, path in paths.items():
        t0 = time.time()
        n = 0
        for chunk in read_chunks(path, chunksize):
            for name in chunk["_name"].values:
                df.update({singular(t) for t in name_tokens(name)})
            for addr in chunk["_addr"].values:
                adf.update(addr_words(parse_address(addr, "")[0]))
            n += len(chunk)
        print(f"  DF pass S{src}: {n:,} rows in {time.time()-t0:.0f}s "
              f"(name vocab {len(df):,}, address vocab {len(adf):,})", flush=True)
    return df, adf


def build_keys_for_chunk(chunk: pd.DataFrame, src: int, df: Counter,
                         rare1_max_df: int, rare2_max_df: int,
                         adf: Counter | None = None) -> pa.Table:
    adf = adf or Counter()
    out = {k: [] for k in (
        "entity_id", "source", "country", "name_norm", "address_norm",
        "k_compact", "k_sorted", "k_pre", "k_suf", "k_phon",
        "k_rare1", "k_rare2", "k_addr", "k_addrname", "k_postname",
        "k_postrare", "k_apair", "k_nameaddr", "hnum", "postcode")}
    for eid, name, addr, num, country in zip(
            chunk[COL_ID].values, chunk["_name"].values,
            chunk["_addr"].values, [""] * len(chunk),
            chunk["_country"].values):
        toks = name_tokens(name)
        compact = strip_glued_legal("".join(toks)) if toks else ""
        sing = [singular(t) for t in toks]

        # list key: full compact name + each side of "x dba y"
        comps = [compact] + [strip_glued_legal("".join(name_tokens(p)))
                             for p in dba_parts(name)]
        comps = list(dict.fromkeys(c for c in comps if len(c) >= 3))
        k_compact = comps or None
        k_sorted = " ".join(sorted(set(sing))) if len(set(sing)) >= 2 else None
        long_enough = len(compact) >= PREFIX_LEN + 2
        k_pre = compact[:PREFIX_LEN] if long_enough else None
        k_suf = compact[-PREFIX_LEN:] if long_enough else None

        alpha = [t for t in sing if t.isalpha()]
        phon = "".join(jellyfish.metaphone(t) for t in alpha) if alpha else ""
        k_phon = phon.replace(" ", "") if len(phon) >= 3 else None

        # rarity-ranked tokens (ties broken alphabetically -> deterministic)
        ranked = sorted({t for t in sing if len(t) >= 3},
                        key=lambda t: (df.get(t, 1), t))
        k_rare1 = ranked[0] if ranked and df.get(ranked[0], 1) <= rare1_max_df else None
        # list key: every pair among the 3 rarest name tokens (one typo in
        # any single token still leaves a shared pair)
        cand3 = [t for t in ranked if df.get(t, 1) <= rare2_max_df][:3]
        k_rare2 = sorted({"|".join(sorted((a, b)))
                          for i, a in enumerate(cand3) for b in cand3[i + 1:]}) or None

        atoks, hn, post = parse_address(addr, num)
        st = street_token(atoks, hn)
        k_addr = f"{hn} {st}" if hn and st else None
        k_addrname = f"{hn} {st} {compact[:2]}" if hn and st and compact else None
        # postcode keys: work for landmark addresses with no house number
        k_postname = f"{post} {compact[:4]}" if post and len(compact) >= 4 else None
        k_postrare = f"{post} {ranked[0]}" if post and ranked else None

        # ---- address-rarity keys (independent of house numbers) ----------
        arank = sorted(addr_words(atoks), key=lambda t: (adf.get(t, 1), t))
        a3 = arank[:3]
        # every pair among the 3 rarest address words: same place even with
        # a typo in one word, different number, or shuffled components
        k_apair = sorted({"|".join(sorted((a, b)))
                          for i, a in enumerate(a3) for b in a3[i + 1:]}) or None
        # each of the 3 rarest name tokens x each of the 2 rarest address
        # words: truncated vendor names ("impex" vs "royal impex pvt ltd")
        # and generic names ("physical therapy center") at the same place
        n3 = ranked[:3]
        k_nameaddr = sorted({f"{n}|{a}" for n in n3 for a in arank[:2]}) or None

        out["entity_id"].append(eid)
        out["source"].append(src)
        out["country"].append(country or "")
        out["name_norm"].append(name)
        out["address_norm"].append(addr)
        out["k_compact"].append(k_compact)
        out["k_sorted"].append(k_sorted)
        out["k_pre"].append(k_pre)
        out["k_suf"].append(k_suf)
        out["k_phon"].append(k_phon)
        out["k_rare1"].append(k_rare1)
        out["k_rare2"].append(k_rare2)
        out["k_addr"].append(k_addr)
        out["k_addrname"].append(k_addrname)
        out["k_postname"].append(k_postname)
        out["k_postrare"].append(k_postrare)
        out["k_apair"].append(k_apair)
        out["k_nameaddr"].append(k_nameaddr)
        out["hnum"].append(hn)
        out["postcode"].append(post)
    out["source"] = pa.array(out["source"], pa.int8())
    return pa.table(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--rare1-max-df", type=int, default=2_000,
                    help="max DF for the single rarest-token key")
    ap.add_argument("--rare2-max-df", type=int, default=200_000,
                    help="max DF for tokens in the rarest-pair key")
    ap.add_argument("--s1-sample", type=float, default=1.0,
                    help="keep this fraction of S1 (deterministic by entity_id hash), e.g. 0.05 for B's dev sample")
    ap.add_argument("--df-cache", default=None,
                    help="reuse a pickled DF Counter (skip pass 1)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = {1: args.s1, 2: args.s2, 3: args.s3}
    t_all = time.time()

    df_path = args.df_cache or os.path.join(args.out, "token_df_v3.pkl")
    if os.path.exists(df_path):
        print(f"Loading token DF from {df_path}")
        with open(df_path, "rb") as f:
            df, adf = pickle.load(f)
    else:
        print("Pass 1: name-token and address-word document frequency")
        df, adf = pass1_df(paths, args.chunksize)
        with open(df_path, "wb") as f:
            pickle.dump((df, adf), f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Pass 2: blocking keys")
    stats = {}
    for src, path in paths.items():
        t0 = time.time()
        out_path = os.path.join(args.out, f"keys_s{src}.parquet")
        writer = None
        n = 0
        for chunk in read_chunks(path, args.chunksize):
            if src == 1 and args.s1_sample < 1.0:
                h = pd.util.hash_pandas_object(chunk[COL_ID], index=False).values
                chunk = chunk[(h % 10_000) < int(args.s1_sample * 10_000)]
            tbl = build_keys_for_chunk(chunk, src, df,
                                       args.rare1_max_df, args.rare2_max_df, adf)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n += len(chunk)
            print(f"  S{src}: {n:,} rows", end="\r", flush=True)
        writer.close()
        stats[f"s{src}"] = {"rows": n, "seconds": round(time.time() - t0, 1)}
        print(f"  S{src}: {n:,} rows -> {out_path} in {time.time()-t0:.0f}s")

    stats["total_seconds"] = round(time.time() - t_all, 1)
    stats["peak_rss_gb"] = peak_rss_gb()
    with open(os.path.join(args.out, "build_keys_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
