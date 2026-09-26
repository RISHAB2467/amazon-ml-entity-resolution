"""
Step 2 of multi-rule blocking: run every blocker, then take the UNION.

Each blocker is an equi-join of S1 against the vendor records (S2 + S3) on
(country, key). Blocks where too many vendor records share a key are dropped
(the "cap"): those are generic keys ("primary care") that explode candidate
counts while adding little recall. DuckDB does the joins out-of-core and
spills to disk, so RAM stays bounded.

Outputs (in --work):
  pairs_<blocker>.parquet     s1_id, cand_id, source
  candidates.parquet          s1_id, cand_id, source, mask (bit per blocker)
  candidate_pairs.tsv         the audited file (format: see --tsv-format)
  blocking_run_stats.json     pairs, runtime, dropped blocks per blocker

Usage:
  python run_blocking.py --work output/blocking_v2 --threads 8 --memory 12GB
"""
from __future__ import annotations

import argparse
import json
import os
import time

import duckdb

# name -> (key column, max vendor records per (country, key) block)
# Order matters only for the mask bits; keep it stable across runs.
BLOCKERS: dict[str, tuple[str, int]] = {
    "compact":   ("k_compact", 200),   # suffix, spacing, inc/incorporated, leet
    "sorted":    ("k_sorted", 200),    # word-order changes
    "prefix":    ("k_pre", 100),       # typo in 2nd half of the name
    "suffix":    ("k_suf", 100),       # typo in 1st half of the name
    "phonetic":  ("k_phon", 100),      # spelling-by-sound
    "rare1":     ("k_rare1", 100),     # one distinctive word shared
    "rare2":     ("k_rare2", 100),     # two rarest words shared
    "addr":      ("k_addr", 50),       # house number + street name
    "addrname":  ("k_addrname", 100),  # address + first 2 chars of name
    "postname":  ("k_postname", 100),  # postcode/PIN + first 4 chars of name
    "postrare":  ("k_postrare", 100),  # postcode/PIN + rarest name token
    # v3: address-rarity keys (list-valued; no house number needed)
    "apair":     ("k_apair", 30),      # any pair of the 3 rarest address words
    "nameaddr":  ("k_nameaddr", 30),   # rare name token x rare address word
    # v4
    "numname":   ("k_numname", 30),    # address number x rare name token
}
# List-valued key columns (k_compact, k_rare2, k_apair, k_nameaddr) are
# unnested automatically: a pair is a candidate if ANY of its keys match.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="dir with keys_s*.parquet")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--memory", default="12GB")
    ap.add_argument("--only", nargs="*", help="run only these blockers")
    ap.add_argument("--no-country", action="store_true",
                    help="do not require equal country (check cross-country share first)")
    ap.add_argument("--tsv-format", choices=["grouped", "long"], default="grouped",
                    help="grouped: s1_id<TAB>id,id,id  |  long: s1_id<TAB>cand_id")
    args = ap.parse_args()

    w = args.work
    con = duckdb.connect(os.path.join(w, "blocking.duckdb"))
    tmp = os.path.join(w, "duckdb_tmp")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")

    con.execute(f"CREATE OR REPLACE VIEW s1 AS SELECT * FROM read_parquet('{w}/keys_s1.parquet')")
    con.execute(f"""CREATE OR REPLACE VIEW v AS
        SELECT * FROM read_parquet(['{w}/keys_s2.parquet', '{w}/keys_s3.parquet'])""")

    names = list(BLOCKERS)
    run = [b for b in names if not args.only or b in args.only]
    stats = {}
    country_join = "" if args.no_country else "AND s.country = v.country"
    country_grp = "" if args.no_country else "v.country,"
    country_blk = "" if args.no_country else "AND v.country = blk.country"

    schema = dict(con.execute("SELECT column_name, column_type FROM (DESCRIBE s1)").fetchall())

    def keyed(tbl: str, col: str, extra: str = "") -> str:
        """one row per (record, key); list columns are unnested"""
        if schema.get(col, "").endswith("[]"):
            return (f"SELECT entity_id, country{extra}, UNNEST({col}) AS key "
                    f"FROM {tbl} WHERE {col} IS NOT NULL")
        return (f"SELECT entity_id, country{extra}, {col} AS key "
                f"FROM {tbl} WHERE {col} IS NOT NULL")

    cj = "" if args.no_country else "AND s.country = v.country"
    cb = "" if args.no_country else "AND v.country = blk.country"
    cg = "" if args.no_country else "country,"

    for b in run:
        col, cap = BLOCKERS[b]
        if col not in schema:
            print(f"{b:10s} skipped: {col} not in keys (rebuild keys with the current build_keys.py)")
            continue
        t0 = time.time()
        out = f"{w}/pairs_{b}.parquet"
        con.execute(f"CREATE OR REPLACE TEMP TABLE vk AS {keyed('v', col, ', source')}")
        con.execute(f"CREATE OR REPLACE TEMP TABLE sk AS {keyed('s1', col)}")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE blk AS
            SELECT {cg} key, count(*) AS n FROM vk GROUP BY ALL""")
        con.execute(f"""
            COPY (
              SELECT DISTINCT s.entity_id AS s1_id, v.entity_id AS cand_id, v.source
              FROM sk s
              JOIN vk v ON s.key = v.key {cj}
              JOIN (SELECT * FROM blk WHERE n <= {cap}) blk ON v.key = blk.key {cb}
            ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
        """)
        dropped = con.execute(f"SELECT count(*), coalesce(sum(n),0) FROM blk WHERE n > {cap}").fetchone()
        n_pairs = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
        stats[b] = {"key": col, "cap": cap, "pairs": n_pairs,
                    "dropped_blocks": dropped[0], "dropped_vendor_records": int(dropped[1]),
                    "seconds": round(time.time() - t0, 1)}
        print(f"{b:10s} pairs={n_pairs:>13,}  dropped_blocks={dropped[0]:>9,}  "
              f"{time.time()-t0:6.0f}s", flush=True)

    # ---- UNION with a bitmask saying which blockers produced each pair ----
    t0 = time.time()
    parts = []
    for b in names:
        p = f"{w}/pairs_{b}.parquet"
        if os.path.exists(p):
            bit = 1 << names.index(b)
            parts.append(f"SELECT s1_id, cand_id, source, {bit}::INTEGER AS bit FROM read_parquet('{p}')")
    con.execute(f"""
        COPY (
          SELECT s1_id, cand_id, any_value(source) AS source, bit_or(bit) AS mask
          FROM ({' UNION ALL '.join(parts)}) GROUP BY s1_id, cand_id
        ) TO '{w}/candidates.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)
    n_union = con.execute(f"SELECT count(*) FROM read_parquet('{w}/candidates.parquet')").fetchone()[0]
    stats["_union"] = {"pairs": n_union, "seconds": round(time.time() - t0, 1),
                       "bit_order": names}
    print(f"{'UNION':10s} pairs={n_union:>13,}  {time.time()-t0:6.0f}s")

    # ---- audited TSV ----
    t0 = time.time()
    if args.tsv_format == "long":
        con.execute(f"""COPY (SELECT s1_id AS source1_entity_id, cand_id AS candidate_entity_id
                        FROM read_parquet('{w}/candidates.parquet') ORDER BY 1, 2)
                        TO '{w}/candidate_pairs.tsv' (DELIMITER '\t', HEADER)""")
    else:
        con.execute(f"""COPY (
              SELECT s.entity_id AS source1_entity_id,
                     coalesce(string_agg(c.cand_id, ',' ORDER BY c.cand_id), '') AS candidate_entity_ids
              FROM s1 s LEFT JOIN read_parquet('{w}/candidates.parquet') c ON s.entity_id = c.s1_id
              GROUP BY s.entity_id ORDER BY 1
            ) TO '{w}/candidate_pairs.tsv' (DELIMITER '\t', HEADER, QUOTE '')""")
    stats["_tsv_seconds"] = round(time.time() - t0, 1)
    try:
        import resource
        stats["_peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)
    except ImportError:  # Windows
        stats["_peak_rss_gb"] = None

    with open(f"{w}/blocking_run_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {w}/candidate_pairs.tsv and blocking_run_stats.json")


if __name__ == "__main__":
    main()
