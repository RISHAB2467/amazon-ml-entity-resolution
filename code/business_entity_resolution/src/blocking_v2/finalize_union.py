"""
Low-memory UNION of the per-blocker pair files (fix for the DuckDB
OutOfMemory at the final union/write step on the 148M-pair test set).

Uses the pairs_<blocker>.parquet files that run_blocking.py ALREADY wrote -
no blocker is re-run. The union is done in N hash buckets of s1_id, so only
1/N of the pairs is in memory at a time. Optionally keeps only the top-K
candidates per S1 (score from trim_weights.json, see trim_eval.py).

Outputs in --work:
  candidates/part_XX.parquet   s1_id, cand_id, source, mask   (read with 'candidates/*.parquet')
  candidate_pairs.tsv          official format, one row per S1 (empty if no candidates)
  finalize_stats.json

Usage (test set, no trim):
  python code/business_entity_resolution/src/blocking_v2/finalize_union.py --work output/blocking_test --memory 4GB
With trim (weights learned on train15):
  python code/business_entity_resolution/src/blocking_v2/finalize_union.py --work output/blocking_test --memory 4GB --topk 50 --weights output/blocking_train15/trim_weights.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import duckdb

from run_blocking import BLOCKERS   # same bit order as the train run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--memory", default="4GB")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--topk", type=int, default=0, help="0 = keep all candidates")
    ap.add_argument("--weights", default=None, help="trim_weights.json (required with --topk)")
    args = ap.parse_args()
    w = args.work.replace("\\", "/")
    t_all = time.time()

    names = list(BLOCKERS)
    parts_in = [(b, 1 << i) for i, b in enumerate(names) if os.path.exists(f"{w}/pairs_{b}.parquet")]
    missing = [b for b in names if not os.path.exists(f"{w}/pairs_{b}.parquet")]
    print(f"blocker files found: {len(parts_in)}  missing: {missing or 'none'}")

    score_sql = "0"
    if args.topk:
        if not args.weights:
            raise SystemExit("--topk needs --weights (run trim_eval.py on a train folder first)")
        wj = json.load(open(args.weights))
        if wj["bit_order"] != names:
            raise SystemExit("bit order in weights file differs from BLOCKERS - rerun trim_eval.py")
        score_sql = " + ".join(f"CASE WHEN (mask & {1 << i}) <> 0 THEN {wj['weights'][b]} ELSE 0 END"
                               for i, b in enumerate(names))

    con = duckdb.connect()
    os.makedirs(f"{w}/duckdb_tmp", exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{w}/duckdb_tmp'")
    con.execute("SET preserve_insertion_order=false")

    out_dir = f"{w}/candidates"
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(f"{out_dir}/part_*.parquet"):
        os.remove(f)
    tsv_parts = []
    B = args.buckets
    total_pairs = 0
    for k in range(B):
        t0 = time.time()
        union = " UNION ALL ".join(
            f"SELECT s1_id, cand_id, source, {bit}::INTEGER AS bit FROM read_parquet('{w}/pairs_{b}.parquet') "
            f"WHERE hash(s1_id) % {B} = {k}" for b, bit in parts_in)
        grouped = f"""SELECT s1_id, cand_id, any_value(source) AS source, bit_or(bit) AS mask
                      FROM ({union}) GROUP BY s1_id, cand_id"""
        if args.topk:
            grouped = f"""SELECT s1_id, cand_id, source, mask FROM ({grouped})
                          QUALIFY row_number() OVER (PARTITION BY s1_id ORDER BY {score_sql} DESC, cand_id) <= {args.topk}"""
        part = f"{out_dir}/part_{k:02d}.parquet"
        con.execute(f"COPY ({grouped}) TO '{part}' (FORMAT parquet, COMPRESSION zstd)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{part}')").fetchone()[0]
        total_pairs += n

        # official TSV rows for the S1s of this bucket (all S1, empty if none)
        tp = f"{w}/duckdb_tmp/cp_{k:02d}.tsv"
        con.execute(f"""COPY (
              SELECT s.entity_id AS source1_entity_id,
                     coalesce(string_agg(c.cand_id, ',' ORDER BY c.cand_id), '') AS candidate_entity_ids
              FROM (SELECT entity_id FROM read_parquet('{w}/keys_s1.parquet') WHERE hash(entity_id) % {B} = {k}) s
              LEFT JOIN read_parquet('{part}') c ON s.entity_id = c.s1_id
              GROUP BY s.entity_id
            ) TO '{tp}' (DELIMITER '\t', HEADER false, QUOTE '')""")
        tsv_parts.append(tp)
        print(f"bucket {k+1:>2}/{B}: {n:>12,} pairs  {time.time()-t0:5.0f}s", flush=True)

    # stitch the TSV (streaming, constant memory)
    with open(f"{w}/candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("source1_entity_id\tcandidate_entity_ids\n")
        for tp in tsv_parts:
            with open(tp, "r", encoding="utf-8") as fi:
                for line in fi:
                    fo.write(line)
            os.remove(tp)

    n_s1, n_zero = con.execute(f"""
        SELECT count(*), sum(CASE WHEN c.s1_id IS NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{w}/keys_s1.parquet') s
        LEFT JOIN (SELECT DISTINCT s1_id FROM read_parquet('{out_dir}/*.parquet')) c ON s.entity_id = c.s1_id""").fetchone()
    stats = {"pairs": total_pairs, "s1": n_s1, "s1_zero_candidates": int(n_zero or 0),
             "avg_per_s1": round(total_pairs / max(n_s1, 1), 2), "topk": args.topk,
             "buckets": B, "seconds": round(time.time() - t_all, 1)}
    json.dump(stats, open(f"{w}/finalize_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {out_dir}/part_*.parquet and {w}/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
