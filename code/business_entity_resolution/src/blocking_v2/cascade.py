"""
Cascade: shrink the candidate set with a learned filter, then a final model on the survivors.

  blocking (~86/S1)  ->  stage-1 = Person B's LightGBM p1  ->  keep top-K per S1 with p1 >= eps
                     ->  candidate_pairs.tsv  (the set the FINAL model runs on, ~K/S1)
                     ->  stage-2 LightGBM on survivors (p1 + within-S1 context)  ->  decide.py

Modes (run from the repo root):
  1) eval   : choose K / eps on train15 OOF (recall kept vs set size)
     python code/business_entity_resolution/src/blocking_v2/cascade.py eval --oof output/model/oof_preds.parquet --s1-keys output/blocking_train15/keys_s1.parquet --gt dataset/train/train_ground_truth.tsv
  2) train  : stage-2 model on OOF survivors -> output/model/stage2/ (models + oof_stage2.parquet)
     python code/business_entity_resolution/src/blocking_v2/cascade.py train --oof output/model/oof_preds.parquet --s1-keys output/blocking_train15/keys_s1.parquet --gt dataset/train/train_ground_truth.tsv --k 10 --eps 0.01
  3) apply  : test -> output/candidate_pairs.tsv (small) + output/model/stage2/test_stage2/*.parquet
     python code/business_entity_resolution/src/blocking_v2/cascade.py apply --preds "output/model/test_preds/*.parquet" --s1-keys output/blocking_test/keys_s1.parquet --k 10 --eps 0.01
Then: decide.py --tune on oof_stage2.parquet, and decide.py --write on test_stage2/*.parquet.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import duckdb
import numpy as np

S2DIR = "output/model/stage2"
FEATS = ["p1", "rk", "gap", "ratio", "best_p", "second_p", "n_surv", "sum_p", "n_hi", "n_blocked"]


def connect(memory: str, threads: int):
    con = duckdb.connect()
    os.makedirs("output/duckdb_tmp_cascade", exist_ok=True)
    con.execute(f"SET memory_limit='{memory}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET temp_directory='output/duckdb_tmp_cascade'")
    con.execute("SET preserve_insertion_order=false")
    return con


def survivors_sql(src: str, k: int, eps: float, where: str = "TRUE") -> str:
    """Top-K per S1 by p1 (and p1 >= eps) + within-S1 context features, computed over
    ALL of the S1's blocking candidates (n_blocked) and over survivors."""
    return f"""
    WITH r AS (
        SELECT s1_id, cand_id, p AS p1,
               row_number() OVER (PARTITION BY s1_id ORDER BY p DESC, cand_id) AS rk,
               max(p) OVER (PARTITION BY s1_id) AS best_p,
               count(*) OVER (PARTITION BY s1_id) AS n_blocked,
               sum(CASE WHEN p >= 0.5 THEN 1 ELSE 0 END) OVER (PARTITION BY s1_id) AS n_hi
        FROM read_parquet('{src}') WHERE {where}),
    s AS (SELECT * FROM r WHERE rk <= {k} AND p1 >= {eps})
    SELECT s1_id, cand_id, p1, rk::FLOAT AS rk,
           (best_p - p1) AS gap, p1 / nullif(best_p, 0) AS ratio, best_p,
           coalesce(nth_value(p1, 2) OVER (PARTITION BY s1_id ORDER BY rk
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING), 0) AS second_p,
           count(*) OVER (PARTITION BY s1_id)::FLOAT AS n_surv,
           sum(p1) OVER (PARTITION BY s1_id) AS sum_p,
           n_hi::FLOAT AS n_hi, n_blocked::FLOAT AS n_blocked
    FROM s"""


def load_gt(con, gt: str, s1_keys: str):
    con.execute(f"CREATE VIEW s1 AS SELECT entity_id FROM read_parquet('{s1_keys}')")
    con.execute(f"""CREATE TABLE gt AS
        SELECT DISTINCT trim(source1_entity_id) AS s1_id, trim(m) AS cand_id
        FROM read_csv('{gt}', delim='\t', header=true, all_varchar=true, quote=''),
             unnest(string_split(coalesce(matched_entity_ids, ''), ',')) AS t(m)
        WHERE trim(m) <> '' AND trim(source1_entity_id) IN (SELECT entity_id FROM s1)""")


def cmd_eval(a):
    con = connect(a.memory, a.threads)
    load_gt(con, a.gt, a.s1_keys)
    n_gt = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    n_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    con.execute(f"""CREATE TABLE r AS
        SELECT o.s1_id, o.cand_id, o.p,
               row_number() OVER (PARTITION BY o.s1_id ORDER BY o.p DESC, o.cand_id) AS rk,
               (g.s1_id IS NOT NULL) AS y
        FROM read_parquet('{a.oof}') o LEFT JOIN gt g USING (s1_id, cand_id)""")
    base_tp, base_n = con.execute("SELECT sum(y::INT), count(*) FROM r").fetchone()
    print(f"blocking only: {base_n:,} pairs = {base_n/n_s1:.1f}/S1, recall {100*base_tp/n_gt:.2f}%")
    print(f"{'K':>4} {'eps':>6} {'pairs':>12} {'per S1':>7} {'recall %':>9} {'lost pt':>8} {'reduction':>10}")
    for k in a.ks:
        for eps in a.epss:
            tp, n = con.execute(f"SELECT sum(y::INT), count(*) FROM r WHERE rk <= {k} AND p >= {eps}").fetchone()
            tp = tp or 0
            print(f"{k:>4} {eps:>6} {n:>12,} {n/n_s1:>7.2f} {100*tp/n_gt:>9.2f} {100*(base_tp-tp)/n_gt:>8.2f} {base_n/max(n,1):>9.1f}x")


SIB = ["sib_support_n", "sib_support_a", "sib_top_n", "sib_top_a", "sib_max_n", "sib_n_close", "sib_top_p"]


def bucket_frame(con, src: str, k: int, eps: float, where: str, keys: str | None):
    """Survivors of one bucket (+ sibling features if keys given).

    Sibling features: how similar this candidate is to the OTHER survivors of
    the same S1 (a business has ~3.5 vendor records that resemble each other),
    weighted by the other survivor's stage-1 probability.
    """
    import pandas as pd
    from rapidfuzz import fuzz, process
    df = con.execute(survivors_sql(src, k, eps, where)).fetchdf()
    if not keys or df.empty:
        return df
    kd = keys.replace("\\", "/")
    con.register("sv_df", df[["s1_id", "cand_id", "p1", "rk"]])
    pairs = con.execute(f"""
        WITH s AS (SELECT sv.s1_id, sv.cand_id, sv.p1, sv.rk,
                          coalesce(v.name_norm,'') AS n, coalesce(v.address_norm,'') AS a
                   FROM sv_df sv JOIN read_parquet(['{kd}/keys_s2.parquet', '{kd}/keys_s3.parquet']) v
                        ON sv.cand_id = v.entity_id)
        SELECT x.s1_id, x.cand_id, y.p1 AS pb, y.rk AS rkb, x.n AS na, y.n AS nb, x.a AS aa, y.a AS ab
        FROM s x JOIN s y ON x.s1_id = y.s1_id AND x.cand_id <> y.cand_id""").fetchdf()
    con.unregister("sv_df")
    if pairs.empty:
        for c in SIB:
            df[c] = np.nan
        return df
    ns = process.cpdist(pairs["na"].tolist(), pairs["nb"].tolist(), scorer=fuzz.token_set_ratio,
                        workers=-1, dtype=np.float32)
    as_ = process.cpdist(pairs["aa"].tolist(), pairs["ab"].tolist(), scorer=fuzz.token_set_ratio,
                         workers=-1, dtype=np.float32)
    empty = (pairs["aa"].to_numpy() == "") | (pairs["ab"].to_numpy() == "")
    as_[empty] = np.nan
    pb = pairs["pb"].to_numpy(np.float32)
    P = pd.DataFrame({"s1_id": pairs["s1_id"], "cand_id": pairs["cand_id"], "rkb": pairs["rkb"].to_numpy(),
                      "pb": pb, "ns": ns, "as": as_, "sn": pb * ns / 100, "sa": pb * np.nan_to_num(as_) / 100,
                      "close": (ns >= 90).astype(np.float32)})
    g = P.groupby(["s1_id", "cand_id"], sort=False)
    agg = g.agg(sib_support_n=("sn", "max"), sib_support_a=("sa", "max"), sib_max_n=("ns", "max"),
                sib_n_close=("close", "sum")).reset_index()
    top = P.sort_values("rkb").drop_duplicates(["s1_id", "cand_id"])[["s1_id", "cand_id", "ns", "as", "pb"]]
    top.columns = ["s1_id", "cand_id", "sib_top_n", "sib_top_a", "sib_top_p"]
    df = df.merge(agg, on=["s1_id", "cand_id"], how="left").merge(top, on=["s1_id", "cand_id"], how="left")
    return df


def feats_for(keys):
    return FEATS + (SIB if keys else [])


def cmd_train(a):
    import lightgbm as lgb
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    con = connect(a.memory, a.threads)
    load_gt(con, a.gt, a.s1_keys)
    parts = []
    for b in range(a.buckets):
        t0 = time.time()
        parts.append(bucket_frame(con, a.oof, a.k, a.eps, f"hash(s1_id) % {a.buckets} = {b}", a.keys))
        print(f"  bucket {b+1}/{a.buckets}: {len(parts[-1]):,} survivors {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(parts, ignore_index=True); del parts
    con.register("dfk", df[["s1_id", "cand_id"]])
    lab = con.execute("""SELECT d.s1_id, d.cand_id, (g.s1_id IS NOT NULL)::INT AS y, (hash(d.s1_id) % 5)::INT AS fold
                         FROM dfk d LEFT JOIN gt g USING (s1_id, cand_id)""").fetchdf()
    df = df.merge(lab, on=["s1_id", "cand_id"], how="left")
    F = feats_for(a.keys)
    X = df[F].astype(np.float32).to_numpy()
    y = df["y"].to_numpy()
    print(f"stage-2 training rows (OOF survivors): {len(df):,}, positives {y.sum():,}, features {len(F)}")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1)
    p2 = np.zeros(len(df), np.float32)
    fold = df["fold"].to_numpy()
    for k in range(5):
        tr, va = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 2000,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        m.save_model(f"{out}/stage2_fold{k}.txt", num_iteration=m.best_iteration)
        p2[va] = m.predict(X[va], num_iteration=m.best_iteration)
        print(f"fold {k}: iters {m.best_iteration}, AUC stage1 {roc_auc_score(y[va], X[va, 0]):.5f} -> stage2 {roc_auc_score(y[va], p2[va]):.5f}")
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.table({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy(), "p": p2}),
                   f"{out}/oof_stage2.parquet")
    with open(f"{out}/cascade_params.txt", "w") as f:
        f.write(f"k={a.k}\neps={a.eps}\nsiblings={bool(a.keys)}\nfeatures={','.join(F)}\n")
    print(f"wrote {out}/oof_stage2.parquet -> decide.py --tune on it")


def cmd_apply(a):
    import lightgbm as lgb
    import pyarrow as pa, pyarrow.parquet as pq
    con = connect(a.memory, a.threads)
    S2 = a.out_dir
    models = [lgb.Booster(model_file=f"{S2}/stage2_fold{k}.txt") for k in range(5)]
    F = feats_for(a.keys)
    if models[0].num_feature() != len(F):
        raise SystemExit(f"model expects {models[0].num_feature()} features, got {len(F)}: use the same --keys setting as in train")
    out_dir = f"{S2}/test_stage2"
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(f"{out_dir}/*.parquet"):
        os.remove(f)
    src = a.preds.replace("\\", "/")
    s1k = a.s1_keys.replace("\\", "/")
    tsv_parts, total = [], 0
    for b in range(a.buckets):
        t0 = time.time()
        df = bucket_frame(con, src, a.k, a.eps, f"hash(s1_id) % {a.buckets} = {b}", a.keys)
        X = df[F].astype(np.float32).to_numpy() if len(df) else np.zeros((0, len(F)), np.float32)
        p2 = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32) if len(df) else np.zeros(0, np.float32)
        part = f"{out_dir}/part_{b:02d}.parquet"
        pq.write_table(pa.table({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy(), "p": p2}), part)
        tp = f"output/duckdb_tmp_cascade/cp_{b:02d}.tsv"
        con.execute(f"""COPY (
            SELECT s.entity_id AS source1_entity_id,
                   coalesce(string_agg(c.cand_id, ',' ORDER BY c.cand_id), '') AS candidate_entity_ids
            FROM (SELECT entity_id FROM read_parquet('{s1k}') WHERE hash(entity_id) % {a.buckets} = {b}) s
            LEFT JOIN read_parquet('{part}') c ON s.entity_id = c.s1_id
            GROUP BY s.entity_id) TO '{tp}' (DELIMITER '\t', HEADER false, QUOTE '')""")
        tsv_parts.append(tp)
        total += len(df)
        print(f"bucket {b+1:>2}/{a.buckets}: {len(df):>10,} survivors  {time.time()-t0:4.0f}s", flush=True)
    with open(a.candidate_out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("source1_entity_id\tcandidate_entity_ids\n")
        for tp in tsv_parts:
            with open(tp, encoding="utf-8") as fi:
                for line in fi:
                    fo.write(line)
            os.remove(tp)
    n_s1 = con.execute(f"SELECT count(*) FROM read_parquet('{s1k}')").fetchone()[0]
    print(f"survivors: {total:,} = {total/n_s1:.2f} per S1 -> {a.candidate_out}; stage-2 scores in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "train", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--memory", default="2GB")
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--s1-keys", required=True)
        if name in ("eval", "train"):
            p.add_argument("--oof", default="output/model/oof_preds.parquet")
            p.add_argument("--gt", required=True)
        if name == "eval":
            p.add_argument("--ks", nargs="*", type=int, default=[3, 5, 8, 10, 15, 20, 30])
            p.add_argument("--epss", nargs="*", type=float, default=[0.0, 0.005, 0.02])
        else:
            p.add_argument("--k", type=int, default=10)
            p.add_argument("--eps", type=float, default=0.01)
        if name in ("train", "apply"):
            p.add_argument("--keys", default=None,
                           help="blocking folder with keys_s2/keys_s3.parquet -> enables sibling features "
                                "(train: output/blocking_train15, apply: output/blocking_test)")
            p.add_argument("--out-dir", default=S2DIR)
            p.add_argument("--buckets", type=int, default=8 if name == "train" else 32)
        if name == "apply":
            p.add_argument("--preds", default="output/model/test_preds/*.parquet")
            p.add_argument("--candidate-out", default="output/candidate_pairs.tsv")
    a = ap.parse_args()
    for attr in ("oof", "gt", "s1_keys", "preds", "keys", "out_dir"):
        if getattr(a, attr, None):
            setattr(a, attr, getattr(a, attr).replace("\\", "/"))
    {"eval": cmd_eval, "train": cmd_train, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
