"""
Step 0 (Person A): profile the raw data before designing anything.

One run answers: row counts, ID uniqueness and prefixes, missing values,
country distribution (train vs test), name/address lengths, duplicates,
non-ASCII share, ground-truth distribution (per S1, S2 vs S3), one-owner rule,
cross-country true pairs, how many true pairs share an exact name, and prints
real matched examples side by side so blocking is designed from evidence.

DuckDB streams the TSVs from disk; nothing is loaded fully into pandas.

Usage (from the repo root):
  python code/business_entity_resolution/src/blocking_v2/profile_data.py --data dataset --out output/profile
"""
from __future__ import annotations

import argparse
import os
import time

import duckdb

FILES = {
    "train_s1": ("train/train_source1.tsv", "S1-"),
    "train_s2": ("train/train_source2.tsv", "S2-"),
    "train_s3": ("train/train_source3.tsv", "S3-"),
    "test_s1": ("test/test_source1.tsv", "S1-"),
    "test_s2": ("test/test_source2.tsv", "S2-"),
    "test_s3": ("test/test_source3.tsv", "S3-"),
}
GT = "train/train_ground_truth.tsv"

# simple SQL normaliser: lowercase, strip accents, punctuation -> space
NORM = "trim(regexp_replace(regexp_replace(lower(strip_accents(coalesce({c}, ''))), '[^a-z0-9 ]', ' ', 'g'), ' +', ' ', 'g'))"


def rd(path: str) -> str:
    p = path.replace("\\", "/")
    return (f"read_csv('{p}', delim='\t', header=true, quote='', escape='', "
            f"all_varchar=true, null_padding=true)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output/profile")
    ap.add_argument("--memory", default="6GB")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--examples", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_all = time.time()

    con = duckdb.connect(os.path.join(args.out, "profile.duckdb"))
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    tmp = os.path.join(args.out, "duckdb_tmp").replace("\\", "/")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")

    L: list[str] = ["# Data profile", ""]

    def out(line=""):
        print(line, flush=True)
        L.append(line)

    # ---- load each file once into DuckDB (columnar, compressed) ----------
    for key, (rel, _) in FILES.items():
        t0 = time.time()
        con.execute(f"""CREATE OR REPLACE TABLE {key} AS
            SELECT entity_id, business_name, business_address, country,
                   {NORM.format(c='business_name')} AS name_n,
                   {NORM.format(c='business_address')} AS addr_n
            FROM {rd(os.path.join(args.data, rel))}""")
        print(f"loaded {key} in {time.time()-t0:.0f}s", flush=True)

    # ---- per-file quality -------------------------------------------------
    out("## 1. Files")
    out("")
    out("| File | Rows | Distinct IDs | Bad prefix | Name missing | Addr missing | Country missing | Non-ASCII name % | Dup name+addr+country | Name len p50/p95/max | Addr len p50/p95/max |")
    out("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for key, (_, pref) in FILES.items():
        r = con.execute(f"""SELECT count(*), count(DISTINCT entity_id),
              sum(CASE WHEN NOT starts_with(entity_id, '{pref}') THEN 1 ELSE 0 END),
              sum(CASE WHEN name_n = '' THEN 1 ELSE 0 END),
              sum(CASE WHEN addr_n = '' THEN 1 ELSE 0 END),
              sum(CASE WHEN country IS NULL OR trim(country) = '' THEN 1 ELSE 0 END),
              round(100.0 * avg(CASE WHEN regexp_matches(coalesce(business_name,''), '[^\\x00-\\x7F]') THEN 1 ELSE 0 END), 2),
              quantile_disc(length(business_name), 0.5), quantile_disc(length(business_name), 0.95), max(length(business_name)),
              quantile_disc(length(business_address), 0.5), quantile_disc(length(business_address), 0.95), max(length(business_address))
            FROM {key}""").fetchone()
        dup = con.execute(f"""SELECT coalesce(sum(n - 1), 0) FROM (
              SELECT count(*) n FROM {key} GROUP BY name_n, addr_n, country HAVING count(*) > 1)""").fetchone()[0]
        out(f"| {key} | {r[0]:,} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[4]:,} | {r[5]:,} | {r[6]} | {int(dup):,} | {r[7]}/{r[8]}/{r[9]} | {r[10]}/{r[11]}/{r[12]} |")

    out("")
    out("## 2. Country values (raw strings)")
    out("")
    out("| File | Country counts |")
    out("|---|---|")
    for key in FILES:
        rows = con.execute(f"SELECT coalesce(country,'<NULL>'), count(*) FROM {key} GROUP BY 1 ORDER BY 2 DESC LIMIT 8").fetchall()
        out(f"| {key} | " + ", ".join(f"{c}: {n:,}" for c, n in rows) + " |")

    # ---- ground truth ------------------------------------------------------
    con.execute(f"""CREATE OR REPLACE TABLE gt_raw AS SELECT * FROM {rd(os.path.join(args.data, GT))}""")
    con.execute("""CREATE OR REPLACE TABLE gt AS
        SELECT DISTINCT trim(source1_entity_id) AS s1_id, trim(m) AS cand_id
        FROM gt_raw, unnest(string_split(coalesce(matched_entity_ids, ''), ',')) AS t(m)
        WHERE trim(m) <> ''""")
    con.execute("CREATE OR REPLACE VIEW vend AS SELECT * FROM train_s2 UNION ALL SELECT * FROM train_s3")

    out("")
    out("## 3. Ground truth")
    out("")
    r = con.execute("SELECT count(*), count(DISTINCT source1_entity_id) FROM gt_raw").fetchone()
    out(f"- GT rows: {r[0]:,} · distinct S1 ids: {r[1]:,}")
    r = con.execute("""SELECT
          (SELECT count(*) FROM train_s1 s WHERE s.entity_id NOT IN (SELECT source1_entity_id FROM gt_raw)),
          (SELECT count(*) FROM gt_raw g WHERE g.source1_entity_id NOT IN (SELECT entity_id FROM train_s1)),
          (SELECT count(*) FROM gt g WHERE g.cand_id NOT IN (SELECT entity_id FROM vend))""").fetchone()
    out(f"- S1 rows missing from GT: {r[0]:,} · GT S1 ids not in S1 file: {r[1]:,} · GT match ids not in S2/S3 files: {r[2]:,}")
    n_pairs = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    out(f"- True pairs: {n_pairs:,}")
    out("")
    out("Matches per S1 (total / from S2 / from S3):")
    out("")
    out("| Matches | S1 count (total) | S1 count (S2 only) | S1 count (S3 only) |")
    out("|---:|---:|---:|---:|")
    dist = con.execute("""
        WITH per AS (
          SELECT g.source1_entity_id AS s1,
                 count(p.cand_id) AS n,
                 count(CASE WHEN starts_with(p.cand_id,'S2-') THEN 1 END) AS n2,
                 count(CASE WHEN starts_with(p.cand_id,'S3-') THEN 1 END) AS n3
          FROM gt_raw g LEFT JOIN gt p ON trim(g.source1_entity_id) = p.s1_id GROUP BY 1)
        SELECT k, sum(CASE WHEN n=k THEN 1 ELSE 0 END), sum(CASE WHEN n2=k THEN 1 ELSE 0 END), sum(CASE WHEN n3=k THEN 1 ELSE 0 END)
        FROM per, range(0, 16) t(k) GROUP BY k ORDER BY k""").fetchall()
    for k, a, b, c in dist:
        if a or b or c:
            out(f"| {k} | {int(a):,} | {int(b):,} | {int(c):,} |")

    r = con.execute("""SELECT count(*), sum(CASE WHEN n > 1 THEN 1 ELSE 0 END), max(n)
                       FROM (SELECT cand_id, count(DISTINCT s1_id) n FROM gt GROUP BY 1)""").fetchone()
    out("")
    out(f"- One-owner rule: {int(r[1] or 0):,} of {r[0]:,} matched vendor records belong to 2+ S1 (max owners {r[2]}) → "
        + ("**holds**" if not r[1] else "**does NOT hold**"))
    r = con.execute("""SELECT
          (SELECT count(*) FROM train_s2) + (SELECT count(*) FROM train_s3),
          (SELECT count(DISTINCT cand_id) FROM gt)""").fetchone()
    out(f"- Vendor records that match nothing: {r[0]-r[1]:,} of {r[0]:,} ({100*(r[0]-r[1])/r[0]:.1f}%)")

    # pair-level difficulty
    r = con.execute("""
        SELECT count(*),
          sum(CASE WHEN s.country <> v.country THEN 1 ELSE 0 END),
          sum(CASE WHEN s.name_n = v.name_n THEN 1 ELSE 0 END),
          sum(CASE WHEN replace(s.name_n,' ','') = replace(v.name_n,' ','') THEN 1 ELSE 0 END),
          sum(CASE WHEN s.addr_n = v.addr_n AND s.addr_n <> '' THEN 1 ELSE 0 END),
          sum(CASE WHEN v.addr_n = '' THEN 1 ELSE 0 END),
          sum(CASE WHEN jaro_winkler_similarity(s.name_n, v.name_n) >= 0.9 THEN 1 ELSE 0 END),
          sum(CASE WHEN jaro_winkler_similarity(s.name_n, v.name_n) < 0.7 THEN 1 ELSE 0 END)
        FROM gt g JOIN train_s1 s ON g.s1_id = s.entity_id JOIN vend v ON g.cand_id = v.entity_id""").fetchone()
    tot = max(r[0], 1)
    out("")
    out("True-pair difficulty:")
    out("")
    out(f"- Different country: {r[1]:,} ({100*r[1]/tot:.2f}%)")
    out(f"- Exact same normalized name: {r[2]:,} ({100*r[2]/tot:.1f}%) · same ignoring spaces: {r[3]:,} ({100*r[3]/tot:.1f}%)")
    out(f"- Exact same normalized address: {r[4]:,} ({100*r[4]/tot:.1f}%) · vendor address empty: {r[5]:,} ({100*r[5]/tot:.1f}%)")
    out(f"- Name Jaro-Winkler ≥ 0.9: {100*r[6]/tot:.1f}% · < 0.7 (very different names): {100*r[7]/tot:.1f}%")

    # tokens that most often differ between true pairs -> real abbreviation / suffix patterns
    out("")
    out("## 4. Most common name tokens present on one side of a true pair only")
    out("(top 40 from a 300k-pair sample: these are the real suffixes / abbreviations to normalise)")
    out("")
    toks = con.execute("""
        WITH p AS (SELECT s.name_n a, v.name_n b FROM gt g
                   JOIN train_s1 s ON g.s1_id = s.entity_id JOIN vend v ON g.cand_id = v.entity_id
                   USING SAMPLE 300000 ROWS (reservoir, 42)),
             x AS (SELECT list_filter(string_split(a,' '), t -> NOT list_contains(string_split(b,' '), t)) l FROM p
                   UNION ALL
                   SELECT list_filter(string_split(b,' '), t -> NOT list_contains(string_split(a,' '), t)) FROM p)
        SELECT t, count(*) n FROM x, unnest(l) u(t) WHERE t <> '' GROUP BY 1 ORDER BY 2 DESC LIMIT 40""").fetchall()
    out(", ".join(f"{t} ({n:,})" for t, n in toks))

    # examples
    out("")
    out(f"## 5. {args.examples} random true pairs per country (S1 | vendor)")
    for (c,) in con.execute("SELECT DISTINCT country FROM train_s1 ORDER BY 1").fetchall():
        out("")
        out(f"### {c}")
        out("")
        out("| S1 name | Vendor name | S1 address | Vendor address |")
        out("|---|---|---|---|")
        ex = con.execute(f"""SELECT s.business_name, v.business_name, s.business_address, v.business_address
            FROM (SELECT * FROM gt USING SAMPLE 200000 ROWS (reservoir, 7)) g
            JOIN train_s1 s ON g.s1_id = s.entity_id JOIN vend v ON g.cand_id = v.entity_id
            WHERE s.country = ? LIMIT {args.examples}""", [c]).fetchall()
        for a, b, c1, d in ex:
            clean = lambda z: (z or "").replace("|", "/")
            out(f"| {clean(a)} | {clean(b)} | {clean(c1)} | {clean(d)} |")

    out("")
    out(f"_Profile time: {time.time()-t_all:.0f}s_")
    with open(os.path.join(args.out, "data_profile.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nSaved {args.out}/data_profile.md")


if __name__ == "__main__":
    main()
