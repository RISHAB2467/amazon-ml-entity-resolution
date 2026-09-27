"""
Step 2 (Person B): LightGBM, 5 folds grouped by S1, memory-safe.

  python code/business_entity_resolution/src/models/train.py --feats output/model/train_feats --out output/model

Training rows per fold = all positives + a random 20% of negatives (weight 5), from the other 4 folds.
Writes:
  output/model/lgbm_fold{0..4}.txt
  output/model/oof_preds.parquet    s1_id, cand_id, p   (EVERY train pair, predicted by the fold that never saw its S1)
  output/model/feature_names.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

META = ["s1_id", "cand_id", "fold", "y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="output/model/train_feats")
    ap.add_argument("--out", default="output/model")
    ap.add_argument("--neg-frac", type=float, default=0.2)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(f"{args.feats}/feat_*.parquet"))
    rng = np.random.default_rng(args.seed)

    # --- load a sampled training table (all positives + neg_frac of negatives) ---
    parts, names = [], None
    for f in files:
        df = pq.read_table(f).to_pandas()
        names = [c for c in df.columns if c not in META]
        keep = (df["y"].to_numpy() == 1) | (rng.random(len(df)) < args.neg_frac)
        parts.append(df.loc[keep, ["fold", "y"] + names])
        print(f"loaded {os.path.basename(f)}: kept {keep.sum():,}/{len(df):,}", flush=True)
        del df
    S = pd.concat(parts, ignore_index=True); del parts
    S[names] = S[names].astype(np.float32)
    w = np.where(S["y"].to_numpy() == 1, 1.0, 1.0 / args.neg_frac).astype(np.float32)
    print(f"sampled training table: {len(S):,} rows, {int(S['y'].sum()):,} positives, {len(names)} features")
    with open(f"{args.out}/feature_names.txt", "w") as fo:
        fo.write("\n".join(names))

    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=args.seed, verbose=-1, num_threads=0)
    fold = S["fold"].to_numpy()
    for k in range(5):
        t0 = time.time()
        tr, va = fold != k, fold == k
        dtr = lgb.Dataset(S.loc[tr, names], S.loc[tr, "y"], weight=w[tr], free_raw_data=True)
        dva = lgb.Dataset(S.loc[va, names], S.loc[va, "y"], weight=w[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        m.save_model(f"{args.out}/lgbm_fold{k}.txt", num_iteration=m.best_iteration)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(S.loc[va, "y"], m.predict(S.loc[va, names], num_iteration=m.best_iteration))
        print(f"fold {k}: best_iter={m.best_iteration} AUC={auc:.5f} ({time.time()-t0:.0f}s)", flush=True)
        del dtr, dva, m
    del S

    # --- OOF predictions for EVERY pair (chunked by feature file) ---
    models = [lgb.Booster(model_file=f"{args.out}/lgbm_fold{k}.txt") for k in range(5)]
    writer = None
    for f in files:
        df = pq.read_table(f).to_pandas()
        p = np.zeros(len(df), np.float32)
        X = df[names].to_numpy(np.float32)
        fd = df["fold"].to_numpy()
        for k in range(5):
            idx = fd == k
            if idx.any():
                p[idx] = models[k].predict(X[idx])
        t = pa.table({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy(), "p": p})
        if writer is None:
            writer = pq.ParquetWriter(f"{args.out}/oof_preds.parquet", t.schema, compression="zstd")
        writer.write_table(t)
        print(f"OOF {os.path.basename(f)} done", flush=True)
        del df, X
    writer.close()

    imp = pd.Series(models[0].feature_importance("gain"), index=names).sort_values(ascending=False)
    print("\nTop 15 features (gain, fold 0):")
    print(imp.head(15).round(0).to_string())
    print(f"\nwrote {args.out}/oof_preds.parquet  -> next: decide.py --tune")


if __name__ == "__main__":
    main()
