"""
Step 3: measure blocking against the training ground truth.

Answers, with numbers:
  * recall of each blocker alone, and of the UNION
  * how many true pairs ONLY this blocker finds (what you lose if you drop it)
  * candidates per S1: mean, p50, p90, p99, max
  * all-match coverage: share of matched S1s whose every true match is a candidate
  * ground-truth facts: one-owner rule, cross-country pairs, missing addresses
  * what the still-missed true pairs look like (categories + a sample to read)

Usage:
  python evaluate_blocking_v2.py --work output/blocking_v2 \
      --gt dataset/train/train_ground_truth.tsv
Writes blocking_report.md and blocking_report.json and missed_sample.tsv in --work.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import duckdb
from rapidfuzz import fuzz

GT_S1_COL = "source1_entity_id"
GT_MATCH_COL = "matched_entity_ids"


def q1(con, sql):
    return con.execute(sql).fetchone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--memory", default="12GB")
    ap.add_argument("--missed-sample", type=int, default=3000)
    args = ap.parse_args()
    w = args.work
    t_all = time.time()

    con = duckdb.connect()
    tmp = os.path.join(w, "duckdb_tmp")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET temp_directory='{tmp}'")

    run_stats = json.load(open(f"{w}/blocking_run_stats.json"))
    names = run_stats["_union"]["bit_order"]

    con.execute(f"CREATE VIEW s1 AS SELECT entity_id, country, name_norm, address_norm, hnum FROM read_parquet('{w}/keys_s1.parquet')")
    con.execute(f"""CREATE VIEW v AS SELECT entity_id, source, country, name_norm, address_norm, hnum
                    FROM read_parquet(['{w}/keys_s2.parquet','{w}/keys_s3.parquet'])""")
    con.execute(f"CREATE VIEW cand AS SELECT * FROM read_parquet('{w}/candidates.parquet')")

    # ---- ground truth, exploded to pairs --------------------------------
    con.execute(f"""
        CREATE TABLE gt AS
        SELECT DISTINCT trim(CAST({GT_S1_COL} AS VARCHAR)) AS s1_id, trim(m) AS cand_id
        FROM read_csv('{args.gt}', delim='\t', header=true, all_varchar=true, quote=''),
             unnest(string_split(coalesce({GT_MATCH_COL}, ''), ',')) AS t(m)
        WHERE trim(m) <> ''
          AND trim(CAST({GT_S1_COL} AS VARCHAR)) IN (SELECT entity_id FROM s1)   -- supports S1 samples
    """)
    n_s1 = q1(con, "SELECT count(*) FROM s1")[0]
    n_gt = q1(con, "SELECT count(*) FROM gt")[0]
    n_s1_matched = q1(con, "SELECT count(DISTINCT s1_id) FROM gt")[0]
    print(f"S1={n_s1:,}  true pairs={n_gt:,}  S1 with >=1 match={n_s1_matched:,}")

    # ---- ground-truth facts that decide later design choices ------------
    facts = {}
    r = q1(con, """SELECT count(*), sum(CASE WHEN n>1 THEN 1 ELSE 0 END), max(n)
                   FROM (SELECT cand_id, count(DISTINCT s1_id) n FROM gt GROUP BY 1)""")
    facts["vendor_records_in_gt"] = r[0]
    facts["vendor_records_with_2plus_owners"] = int(r[1] or 0)
    facts["max_owners_per_vendor_record"] = r[2]
    facts["one_owner_rule_holds"] = (r[1] or 0) == 0
    r = q1(con, """SELECT count(*),
                          sum(CASE WHEN s.country <> v.country THEN 1 ELSE 0 END),
                          sum(CASE WHEN v.address_norm IS NULL OR v.address_norm = '' THEN 1 ELSE 0 END),
                          sum(CASE WHEN s.hnum IS NOT NULL AND v.hnum IS NOT NULL AND s.hnum <> v.hnum THEN 1 ELSE 0 END),
                          sum(CASE WHEN g.cand_id IS NOT NULL AND v.entity_id IS NULL THEN 1 ELSE 0 END)
                   FROM gt g LEFT JOIN s1 s ON g.s1_id = s.entity_id
                             LEFT JOIN v ON g.cand_id = v.entity_id""")
    facts["gt_pairs_joined"] = r[0]
    facts["gt_pairs_cross_country"] = int(r[1] or 0)
    facts["gt_pairs_vendor_address_missing"] = int(r[2] or 0)
    facts["gt_pairs_house_number_conflict"] = int(r[3] or 0)
    facts["gt_ids_not_found_in_vendor_files"] = int(r[4] or 0)

    # ---- label candidates -------------------------------------------------
    con.execute("""CREATE TABLE cl AS
                   SELECT c.s1_id, c.cand_id, c.source, c.mask, (g.s1_id IS NOT NULL) AS is_true
                   FROM cand c LEFT JOIN gt g USING (s1_id, cand_id)""")

    rows = []
    for i, b in enumerate(names):
        bit = 1 << i
        r = q1(con, f"""SELECT count(*), sum(is_true::INT),
                               sum(CASE WHEN mask = {bit} AND is_true THEN 1 ELSE 0 END)
                        FROM cl WHERE (mask & {bit}) <> 0""")
        pairs, tp, unique_tp = r[0], int(r[1] or 0), int(r[2] or 0)
        rs = run_stats.get(b, {})
        rows.append({
            "blocker": b, "pairs": pairs,
            "pair_recall_%": round(100 * tp / n_gt, 2) if n_gt else 0,
            "unique_recall_%": round(100 * unique_tp / n_gt, 2) if n_gt else 0,
            "precision_%": round(100 * tp / pairs, 2) if pairs else 0,
            "avg_cand_per_s1": round(pairs / n_s1, 2),
            "seconds": rs.get("seconds"),
            "dropped_blocks": rs.get("dropped_blocks"),
        })

    # union
    r = q1(con, "SELECT count(*), sum(is_true::INT) FROM cl")
    u_pairs, u_tp = r[0], int(r[1] or 0)
    dist = q1(con, f"""
        WITH per AS (SELECT s.entity_id, count(c.cand_id) n
                     FROM s1 s LEFT JOIN cand c ON s.entity_id = c.s1_id GROUP BY 1)
        SELECT avg(n), quantile_cont(n, 0.5), quantile_cont(n, 0.9), quantile_cont(n, 0.99),
               max(n), sum(CASE WHEN n = 0 THEN 1 ELSE 0 END), sum(CASE WHEN n > 200 THEN 1 ELSE 0 END)
        FROM per""")
    cov = q1(con, """
        WITH per AS (SELECT g.s1_id, count(*) n_true, sum(CASE WHEN c.s1_id IS NOT NULL THEN 1 ELSE 0 END) n_hit
                     FROM gt g LEFT JOIN cand c USING (s1_id, cand_id) GROUP BY 1)
        SELECT sum(CASE WHEN n_hit = n_true THEN 1 ELSE 0 END), sum(CASE WHEN n_hit = 0 THEN 1 ELSE 0 END) FROM per""")
    by_src = con.execute("""
        SELECT v.source, count(*), sum(CASE WHEN c.s1_id IS NOT NULL THEN 1 ELSE 0 END)
        FROM gt g JOIN v ON g.cand_id = v.entity_id LEFT JOIN cand c USING (s1_id, cand_id)
        GROUP BY 1 ORDER BY 1""").fetchall()
    by_cty = con.execute("""
        WITH r AS (SELECT s.country, count(*) t, sum(CASE WHEN c.s1_id IS NOT NULL THEN 1 ELSE 0 END) h
                   FROM gt g JOIN s1 s ON g.s1_id = s.entity_id LEFT JOIN cand c USING (s1_id, cand_id) GROUP BY 1),
             n AS (SELECT s.country, count(DISTINCT s.entity_id) ns, count(c.cand_id) nc
                   FROM s1 s LEFT JOIN cand c ON s.entity_id = c.s1_id GROUP BY 1)
        SELECT n.country, r.t, r.h, n.ns, n.nc FROM n LEFT JOIN r USING (country) ORDER BY 1""").fetchall()
    union = {
        "pairs": u_pairs,
        "by_country": {c: {"recall_%": round(100 * (h or 0) / t, 2) if t else None,
                           "avg_cand_per_s1": round(nc / ns, 1) if ns else None}
                       for c, t, h, ns, nc in by_cty},
        "pair_recall_%": round(100 * u_tp / n_gt, 3),
        "precision_%": round(100 * u_tp / u_pairs, 2) if u_pairs else 0,
        "avg_cand_per_s1": round(dist[0], 2), "p50": dist[1], "p90": dist[2],
        "p99": dist[3], "max": dist[4],
        "s1_with_zero_candidates": int(dist[5]), "s1_with_over_200": int(dist[6]),
        "all_match_coverage_%": round(100 * cov[0] / n_s1_matched, 2) if n_s1_matched else 0,
        "matched_s1_with_no_true_candidate": int(cov[1]),
        "recall_by_source_%": {f"S{s}": round(100 * h / t, 2) for s, t, h in by_src},
    }

    # ---- missed true pairs: categories + readable sample -----------------
    con.execute("""CREATE TABLE missed AS
        SELECT g.s1_id, g.cand_id, v.source,
               s.country AS s1_country, v.country AS v_country,
               s.name_norm AS s1_name, v.name_norm AS v_name,
               s.address_norm AS s1_addr, v.address_norm AS v_addr,
               s.hnum AS s1_hnum, v.hnum AS v_hnum
        FROM gt g LEFT JOIN cand c USING (s1_id, cand_id)
                  LEFT JOIN s1 s ON g.s1_id = s.entity_id
                  LEFT JOIN v ON g.cand_id = v.entity_id
        WHERE c.s1_id IS NULL""")
    n_missed = q1(con, "SELECT count(*) FROM missed")[0]
    sample = con.execute(f"SELECT * FROM missed USING SAMPLE {min(args.missed_sample, max(n_missed,1))} ROWS (reservoir, 42)").fetchdf()
    cats = {"cross_country": 0, "vendor_addr_missing": 0, "same_house_number": 0,
            "name_sim>=90": 0, "name_sim_70_90": 0, "name_sim_50_70": 0, "name_sim<50": 0}
    sims = []
    for r in sample.itertuples():
        a, b = r.s1_name or "", r.v_name or ""
        sim = fuzz.token_set_ratio(a, b)
        sims.append(sim)
        if r.s1_country != r.v_country: cats["cross_country"] += 1
        if not r.v_addr: cats["vendor_addr_missing"] += 1
        if r.s1_hnum and r.s1_hnum == r.v_hnum: cats["same_house_number"] += 1
        if sim >= 90: cats["name_sim>=90"] += 1
        elif sim >= 70: cats["name_sim_70_90"] += 1
        elif sim >= 50: cats["name_sim_50_70"] += 1
        else: cats["name_sim<50"] += 1
    sample["name_token_set_ratio"] = sims
    sample.sort_values("name_token_set_ratio", ascending=False).to_csv(
        f"{w}/missed_sample.tsv", sep="\t", index=False)
    ns = max(len(sample), 1)
    missed = {"missed_pairs": n_missed, "sample_size": len(sample),
              "categories_%_of_sample": {k: round(100 * v / ns, 1) for k, v in cats.items()}}

    report = {"n_s1": n_s1, "n_true_pairs": n_gt, "n_s1_matched": n_s1_matched,
              "gt_facts": facts, "blockers": rows, "union": union, "missed": missed,
              "eval_seconds": round(time.time() - t_all, 1)}
    json.dump(report, open(f"{w}/blocking_report.json", "w"), indent=2, default=str)

    # ---- markdown report ---------------------------------------------------
    L = ["# Blocking report", "",
         f"S1 = {n_s1:,} · true pairs = {n_gt:,} · S1 with a match = {n_s1_matched:,}", "",
         "## Per blocker", "",
         "| Blocker | Pairs | Recall % | Only-this % | Precision % | Avg cand/S1 | Seconds | Dropped blocks |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['blocker']} | {r['pairs']:,} | {r['pair_recall_%']} | {r['unique_recall_%']} | "
                 f"{r['precision_%']} | {r['avg_cand_per_s1']} | {r['seconds']} | {r['dropped_blocks']} |")
    u = union
    L += ["", "## Union", "",
          f"- Pair recall: **{u['pair_recall_%']}%** (exact-name baseline: 21.85%)",
          f"- Recall by source: {u['recall_by_source_%']}",
          f"- By country (recall %, avg candidates/S1): {u['by_country']}",
          f"- All-match coverage: **{u['all_match_coverage_%']}%** (baseline: 2.8%)",
          f"- Pairs: {u['pairs']:,} · precision {u['precision_%']}%",
          f"- Candidates per S1: mean {u['avg_cand_per_s1']}, p50 {u['p50']}, p90 {u['p90']}, p99 {u['p99']}, max {u['max']}",
          f"- S1 with zero candidates: {u['s1_with_zero_candidates']:,} · with >200: {u['s1_with_over_200']:,}",
          f"- Matched S1 with no true candidate at all: {u['matched_s1_with_no_true_candidate']:,}",
          "", "## Ground-truth facts", ""]
    for k, v in facts.items():
        L.append(f"- {k}: {v:,}" if isinstance(v, int) and not isinstance(v, bool) else f"- {k}: {v}")
    L += ["", f"## Missed true pairs: {n_missed:,}", "",
          f"Categories in a sample of {len(sample):,} (a pair can be in several):", ""]
    for k, v in missed["categories_%_of_sample"].items():
        L.append(f"- {k}: {v}%")
    L += ["", "Read `missed_sample.tsv` (sorted by name similarity) to decide the next blocker.",
          "", f"_Evaluation time: {report['eval_seconds']}s_"]
    open(f"{w}/blocking_report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
