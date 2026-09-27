"""
Step 3 (Person B): score every TEST candidate (features computed on the fly, not stored:
148M pairs of features would not fit on disk comfortably). Resumable: finished buckets are skipped.

  python code/business_entity_resolution/src/models/predict_test.py --work output/blocking_test --models output/model --out output/model/test_preds --buckets 96

Writes output/model/test_preds/pred_XX.parquet  (s1_id, cand_id, p = mean of the 5 fold models)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_lib import PAIR_SQL, build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="output/blocking_test")
    ap.add_argument("--models", default="output/model")
    ap.add_argument("--out", default="output/model/test_preds")
    ap.add_argument("--buckets", type=int, default=96)
    ap.add_argument("--memory", default="2GB")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--from-bucket", type=int, default=0, help="split work across laptops: A runs 48-95, B runs 0-47")
    ap.add_argument("--to-bucket", type=int, default=None, help="last bucket, inclusive")
    args = ap.parse_args()
    w = args.work.replace("\\", "/")
    os.makedirs(args.out, exist_ok=True)
    cands = f"{w}/candidates/*.parquet" if os.path.isdir(f"{w}/candidates") else f"{w}/candidates.parquet"

    names = open(f"{args.models}/feature_names.txt").read().split("\n")
    models = [lgb.Booster(model_file=f"{args.models}/lgbm_fold{k}.txt") for k in range(5)]
    con = duckdb.connect()
    os.makedirs("output/duckdb_tmp_b", exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute("SET temp_directory='output/duckdb_tmp_b'")

    t_all = time.time()
    total = 0
    last = args.buckets - 1 if args.to_bucket is None else args.to_bucket
    for k in range(args.from_bucket, last + 1):
        out = f"{args.out}/pred_{k:02d}.parquet"
        if os.path.exists(out):
            print(f"bucket {k}: exists, skipped"); continue
        t0 = time.time()
        df = con.execute(PAIR_SQL.format(cands=cands, nb=args.buckets, k=k, w=w)).fetchdf()
        X, fnames = build_features(df)
        assert fnames == names, "feature list differs from training - use the same features_lib.py"
        p = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
        t = pa.table({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy(), "p": p})
        pq.write_table(t, out + ".tmp", compression="zstd")
        os.replace(out + ".tmp", out)
        total += len(df)
        el = time.time() - t_all
        print(f"bucket {k+1:>3}/{args.buckets}: {len(df):>9,} pairs {time.time()-t0:5.0f}s  "
              f"(elapsed {el/60:.0f} min)", flush=True)
        del df, X
    print(f"done: {total:,} pairs -> {args.out}  next: decide.py --write")


if __name__ == "__main__":
    main()
