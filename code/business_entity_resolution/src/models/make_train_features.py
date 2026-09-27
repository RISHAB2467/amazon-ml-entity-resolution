"""
Step 1 (Person B): features + labels + folds for the TRAIN sample, written in buckets.

  python code/business_entity_resolution/src/models/make_train_features.py --work output/blocking_train15 --gt dataset/train/train_ground_truth.tsv --out output/model/train_feats --buckets 32

Writes output/model/train_feats/feat_XX.parquet with: s1_id, cand_id, fold, y, <features>
fold = hash(s1_id) % 5  (S1-grouped: an S1 is never in train and validation of the same fold)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_lib import PAIR_SQL, build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="output/model/train_feats")
    ap.add_argument("--buckets", type=int, default=32)
    ap.add_argument("--memory", default="2GB")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    w = args.work.replace("\\", "/")
    os.makedirs(args.out, exist_ok=True)
    cands = f"{w}/candidates/*.parquet" if os.path.isdir(f"{w}/candidates") else f"{w}/candidates.parquet"

    con = duckdb.connect()
    os.makedirs("output/duckdb_tmp_b", exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute("SET temp_directory='output/duckdb_tmp_b'")
    con.execute(f"""CREATE TABLE gt AS
        SELECT DISTINCT trim(source1_entity_id) AS s1_id, trim(m) AS cand_id
        FROM read_csv('{args.gt.replace(chr(92), '/')}', delim='\t', header=true, all_varchar=true, quote=''),
             unnest(string_split(coalesce(matched_entity_ids, ''), ',')) AS t(m)
        WHERE trim(m) <> ''""")

    t_all = time.time()
    tot = pos = 0
    for k in range(args.buckets):
        out = f"{args.out}/feat_{k:02d}.parquet"
        if os.path.exists(out):
            print(f"bucket {k}: exists, skipped"); continue
        t0 = time.time()
        q = f"""SELECT b.*, (g.s1_id IS NOT NULL)::TINYINT AS y, (hash(b.s1_id) % 5)::TINYINT AS fold
                FROM ({PAIR_SQL.format(cands=cands, nb=args.buckets, k=k, w=w)}) b
                LEFT JOIN gt g ON b.s1_id = g.s1_id AND b.cand_id = g.cand_id"""
        df = con.execute(q).fetchdf()
        X, names = build_features(df)
        tbl = pa.table({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy(),
                        "fold": df["fold"].to_numpy(np.int8), "y": df["y"].to_numpy(np.int8),
                        **{n: X[:, i] for i, n in enumerate(names)}})
        pq.write_table(tbl, out + ".tmp", compression="zstd")
        os.replace(out + ".tmp", out)
        tot += len(df); pos += int(df["y"].sum())
        print(f"bucket {k+1:>2}/{args.buckets}: {len(df):>9,} pairs, {int(df['y'].sum()):>7,} positives, "
              f"{time.time()-t0:5.0f}s", flush=True)
        del df, X, tbl
    print(f"done: {tot:,} pairs ({pos:,} positives) in {time.time()-t_all:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
