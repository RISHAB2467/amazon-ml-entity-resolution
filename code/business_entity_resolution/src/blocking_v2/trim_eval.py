"""
Measure how much recall a top-K trim would cost, using ONLY the blocker mask.

Each blocker gets a weight = log((true pairs it found + 1) / (false pairs + 1)),
learned on a labelled work folder (train sample). A candidate's score is the
sum of the weights of the blockers that found it. Keeping the top K per S1 is
then a pure integer operation - no string work - so it is cheap on 150M pairs.

Writes <work>/trim_weights.json (used by finalize_union.py --topk) and prints
recall retained / pairs kept for several K.

Usage:
  python code/business_entity_resolution/src/blocking_v2/trim_eval.py --work output/blocking_train15 --gt dataset/train/train_ground_truth.tsv --memory 4GB
"""
from __future__ import annotations

import argparse
import json
import math
import os

import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--memory", default="4GB")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--k", nargs="*", type=int, default=[20, 30, 40, 50, 60, 80, 100, 150])
    args = ap.parse_args()
    w = args.work.replace("\\", "/")

    con = duckdb.connect()
    os.makedirs(f"{w}/duckdb_tmp", exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{w}/duckdb_tmp'")
    con.execute("SET preserve_insertion_order=false")

    names = json.load(open(f"{w}/blocking_run_stats.json"))["_union"]["bit_order"]
    con.execute(f"CREATE VIEW s1 AS SELECT entity_id FROM read_parquet('{w}/keys_s1.parquet')")
    con.execute(f"""CREATE TABLE gt AS
        SELECT DISTINCT trim(source1_entity_id) AS s1_id, trim(m) AS cand_id
        FROM read_csv('{args.gt.replace(chr(92), '/')}', delim='\t', header=true, all_varchar=true, quote=''),
             unnest(string_split(coalesce(matched_entity_ids, ''), ',')) AS t(m)
        WHERE trim(m) <> '' AND trim(source1_entity_id) IN (SELECT entity_id FROM s1)""")
    n_gt = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    con.execute(f"""CREATE TABLE cl AS
        SELECT c.s1_id, c.cand_id, c.mask, (g.s1_id IS NOT NULL) AS y
        FROM read_parquet('{w}/candidates.parquet') c LEFT JOIN gt g USING (s1_id, cand_id)""")

    # per-blocker weights
    weights = {}
    for i, b in enumerate(names):
        tp, fp = con.execute(f"""SELECT sum(y::INT), sum((NOT y)::INT) FROM cl
                                 WHERE (mask & {1 << i}) <> 0""").fetchone()
        tp, fp = int(tp or 0), int(fp or 0)
        weights[b] = round(math.log((tp + 1) / (fp + 1)) + 10.0, 4)   # +10 keeps all weights positive
        print(f"  {b:10s} tp={tp:>10,} fp={fp:>11,} weight={weights[b]}")
    json.dump({"bit_order": names, "weights": weights},
              open(f"{w}/trim_weights.json", "w"), indent=2)

    score = " + ".join(f"CASE WHEN (mask & {1 << i}) <> 0 THEN {weights[b]} ELSE 0 END"
                       for i, b in enumerate(names))
    con.execute(f"""CREATE TABLE ranked AS
        SELECT y, row_number() OVER (PARTITION BY s1_id ORDER BY {score} DESC, cand_id) AS rk
        FROM cl""")
    n_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    base_tp, base_pairs = con.execute("SELECT sum(y::INT), count(*) FROM ranked").fetchone()
    print(f"\nNo trim: pairs={base_pairs:,}  recall={100*base_tp/n_gt:.2f}%  avg/S1={base_pairs/n_s1:.1f}")
    print(f"{'K':>5} {'pairs':>13} {'avg/S1':>7} {'recall %':>9} {'recall lost':>12}")
    for k in args.k:
        tp, pairs = con.execute(f"SELECT sum(y::INT), count(*) FROM ranked WHERE rk <= {k}").fetchone()
        print(f"{k:>5} {pairs:>13,} {pairs/n_s1:>7.1f} {100*tp/n_gt:>9.2f} {100*(base_tp-tp)/n_gt:>11.2f}pt")
    print(f"\nwrote {w}/trim_weights.json")


if __name__ == "__main__":
    main()
