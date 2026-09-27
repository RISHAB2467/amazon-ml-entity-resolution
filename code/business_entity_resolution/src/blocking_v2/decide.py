"""
Decision layer: model probabilities -> matching_results.tsv (the A/B hand-off).

Person B delivers probabilities as Parquet with columns  s1_id, cand_id, p
  - OOF on train15:  output/model/oof_preds.parquet      (plus labels are recomputed from GT here)
  - test:            output/model/test_preds/*.parquet

Rule (tuned on OOF):  keep a pair if  p >= t   OR  (it is the S1's best candidate AND p >= t_best)
then optionally the one-owner rule: each vendor record kept only for its highest-p S1.

1) Tune on OOF (macro F0.5 over ALL S1 of the sample; recall counts blocking misses):
   python code/business_entity_resolution/src/blocking_v2/decide.py --preds output/model/oof_preds.parquet --s1-keys output/blocking_train15/keys_s1.parquet --gt dataset/train/train_ground_truth.tsv --tune
2) Write the test submission with the chosen values:
   python code/business_entity_resolution/src/blocking_v2/decide.py --preds "output/model/test_preds/*.parquet" --s1-keys output/blocking_test/keys_s1.parquet --t 0.55 --t-best 0.35 --one-owner --write output/matching_results.tsv
"""
from __future__ import annotations

import argparse
import os

import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="parquet file or glob with s1_id, cand_id, p")
    ap.add_argument("--s1-keys", required=True, help="keys_s1.parquet = the S1 universe (every S1 gets a row)")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--t-best", type=float, default=None, help="default = t (no special top-1 rule)")
    ap.add_argument("--one-owner", action="store_true")
    ap.add_argument("--write", default=None)
    ap.add_argument("--memory", default="2GB")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    con = duckdb.connect()
    tmp = "output/duckdb_tmp_decide"
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")

    con.execute(f"CREATE VIEW s1 AS SELECT entity_id FROM read_parquet('{args.s1_keys.replace(chr(92), '/')}')")
    # rank each S1's candidates once, to disk
    con.execute(f"""COPY (
        SELECT s1_id, cand_id, p,
               row_number() OVER (PARTITION BY s1_id ORDER BY p DESC, cand_id) AS rk
        FROM read_parquet('{args.preds.replace(chr(92), '/')}')
      ) TO '{tmp}/ranked.parquet' (FORMAT parquet)""")
    con.execute(f"CREATE VIEW ranked AS SELECT * FROM read_parquet('{tmp}/ranked.parquet')")
    n = con.execute("SELECT count(*), count(DISTINCT s1_id) FROM ranked").fetchone()
    print(f"predictions: {n[0]:,} pairs for {n[1]:,} S1")

    if args.gt:
        con.execute(f"""CREATE TABLE gt AS
            SELECT DISTINCT trim(source1_entity_id) AS s1_id, trim(m) AS cand_id
            FROM read_csv('{args.gt.replace(chr(92), '/')}', delim='\t', header=true, all_varchar=true, quote=''),
                 unnest(string_split(coalesce(matched_entity_ids, ''), ',')) AS t(m)
            WHERE trim(m) <> '' AND trim(source1_entity_id) IN (SELECT entity_id FROM s1)""")
        con.execute("CREATE TABLE ntrue AS SELECT s1_id, count(*) nt FROM gt GROUP BY 1")

    def select(t: float, tb: float, one_owner: bool) -> str:
        q = f"SELECT s1_id, cand_id, p FROM ranked WHERE p >= {t} OR (rk = 1 AND p >= {tb})"
        if one_owner:
            q = f"""SELECT s1_id, cand_id, p FROM ({q})
                    QUALIFY row_number() OVER (PARTITION BY cand_id ORDER BY p DESC, s1_id) = 1"""
        return q

    def score(t: float, tb: float, one_owner: bool) -> tuple:
        con.execute(f"CREATE OR REPLACE TABLE pred AS {select(t, tb, one_owner)}")
        return con.execute("""
            WITH p AS (SELECT p.s1_id, count(*) np, sum(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) tp
                       FROM pred p LEFT JOIN gt g USING (s1_id, cand_id) GROUP BY 1),
                 e AS (SELECT coalesce(p.np,0) np, coalesce(p.tp,0) tp, coalesce(t.nt,0) nt
                       FROM s1 s LEFT JOIN p ON s.entity_id = p.s1_id LEFT JOIN ntrue t ON s.entity_id = t.s1_id)
            SELECT avg(CASE WHEN nt = 0 AND np = 0 THEN 1.0
                            WHEN nt = 0 OR np = 0 OR tp = 0 THEN 0.0
                            ELSE 1.25 * (tp/np) * (tp/nt) / (0.25 * (tp/np) + (tp/nt)) END),
                   sum(tp) / nullif(sum(np), 0), sum(tp) / nullif(sum(nt), 0)
            FROM e""").fetchone()

    if args.tune:
        if not args.gt:
            raise SystemExit("--tune needs --gt")
        best = (-1, None)
        print(f"{'t':>5} {'t_best':>6} {'1-owner':>7} {'F0.5':>8} {'prec%':>6} {'rec%':>6}")
        for oo in (False, True):
            for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                for d in (0.0, 0.15, 0.3):
                    tb = round(max(t - d, 0.05), 2)
                    f, pr, rc = score(t, tb, oo)
                    print(f"{t:>5} {tb:>6} {str(oo):>7} {f:>8.4f} {100*(pr or 0):>6.1f} {100*(rc or 0):>6.1f}", flush=True)
                    if f > best[0]:
                        best = (f, (t, tb, oo))
        # refine around the best t in 0.02 steps
        t0, tb0, oo = best[1]
        for t in [round(t0 + k * 0.02, 2) for k in range(-4, 5)]:
            for tb in sorted({round(max(tb0 + k * 0.05, 0.05), 2) for k in range(-2, 3)} | {t}):
                if tb > t:
                    continue
                f, pr, rc = score(t, tb, oo)
                if f > best[0]:
                    best = (f, (t, tb, oo))
        f, (t, tb, oo) = best
        print(f"\nBEST: macro F0.5 = {f:.4f}  with  --t {t} --t-best {tb}" + ("  --one-owner" if oo else ""))

    elif args.gt:
        tb = args.t_best if args.t_best is not None else args.t
        f, pr, rc = score(args.t, tb, args.one_owner)
        print(f"macro F0.5 = {f:.4f}   pair precision = {100*(pr or 0):.1f}%   pair recall = {100*(rc or 0):.1f}%")

    if args.write:
        tb = args.t_best if args.t_best is not None else args.t
        con.execute(f"COPY ({select(args.t, tb, args.one_owner)}) TO '{tmp}/final.parquet' (FORMAT parquet)")
        con.execute(f"""COPY (
            SELECT s.entity_id AS source1_entity_id,
                   coalesce(string_agg(p.cand_id, ',' ORDER BY p.cand_id), '') AS matched_entity_ids
            FROM s1 s LEFT JOIN read_parquet('{tmp}/final.parquet') p ON s.entity_id = p.s1_id
            GROUP BY s.entity_id ORDER BY 1
          ) TO '{args.write.replace(chr(92), '/')}' (DELIMITER '\t', HEADER, QUOTE '')""")
        n = con.execute(f"SELECT count(*), count(DISTINCT s1_id) FROM read_parquet('{tmp}/final.parquet')").fetchone()
        print(f"wrote {args.write}: {n[0]:,} matches, {n[1]:,} S1 non-empty")


if __name__ == "__main__":
    main()
