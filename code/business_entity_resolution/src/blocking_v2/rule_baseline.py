"""
Safety-net submission: match = candidate pairs whose blocker mask satisfies a
high-precision rule. No model, runs in minutes.

1) Score several rules on a labelled folder (macro F0.5 over ALL S1, recall
   denominator = ALL true matches incl. those blocking missed):
   python code/business_entity_resolution/src/blocking_v2/rule_baseline.py --work output/blocking_train15 --gt dataset/train/train_ground_truth.tsv --memory 4GB
2) Write the test submission with the best rule:
   python code/business_entity_resolution/src/blocking_v2/rule_baseline.py --work output/blocking_test --rule R3 --write output/matching_results.tsv --memory 4GB
"""
from __future__ import annotations

import argparse
import os

import duckdb

from run_blocking import BLOCKERS

NAMES = list(BLOCKERS)


def bit(name: str) -> str:
    return f"((mask & {1 << NAMES.index(name)}) <> 0)"


def nbits() -> str:
    return "bit_count(mask)"


# rules = SQL boolean expressions over the blocker mask
RULES = {
    "R1": f"{bit('addrname')}",
    "R2": f"{bit('addrname')} OR {bit('postname')} OR {bit('postrare')}",
    "R3": f"{bit('addrname')} OR {bit('postname')} OR {bit('postrare')} "
          f"OR ({bit('compact')} AND ({bit('numname')} OR {bit('apair')} OR {bit('addr')}))",
    "R4": f"{bit('addrname')} OR {bit('postname')} OR {bit('postrare')} "
          f"OR ({bit('compact')} AND ({bit('numname')} OR {bit('apair')} OR {bit('addr')})) "
          f"OR ({bit('sorted')} AND {bit('nameaddr')} AND {bit('numname')})",
    "R5": f"{nbits()} >= 6",
    "R6": f"{nbits()} >= 5",
    "R7": f"({bit('addrname')} OR {bit('postname')} OR {bit('postrare')}) OR {nbits()} >= 6",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--rule", default=None, help="evaluate/write only this rule")
    ap.add_argument("--write", default=None, help="path for matching_results.tsv")
    ap.add_argument("--one-owner", action="store_true",
                    help="if a vendor record is matched to 2+ S1, keep the one with most blockers")
    ap.add_argument("--memory", default="2GB")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    w = args.work.replace("\\", "/")

    con = duckdb.connect()
    os.makedirs(f"{w}/duckdb_tmp", exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{w}/duckdb_tmp'")
    con.execute("SET preserve_insertion_order=false")

    src = f"{w}/candidates/*.parquet" if os.path.isdir(f"{w}/candidates") else f"{w}/candidates.parquet"
    con.execute(f"CREATE VIEW cand AS SELECT * FROM read_parquet('{src}')")
    con.execute(f"CREATE VIEW s1 AS SELECT entity_id FROM read_parquet('{w}/keys_s1.parquet')")
    rules = {args.rule: RULES[args.rule]} if args.rule else RULES

    if args.gt:
        con.execute(f"""CREATE TABLE gt AS
            SELECT DISTINCT trim(source1_entity_id) AS s1_id, trim(m) AS cand_id
            FROM read_csv('{args.gt.replace(chr(92), '/')}', delim='\t', header=true, all_varchar=true, quote=''),
                 unnest(string_split(coalesce(matched_entity_ids, ''), ',')) AS t(m)
            WHERE trim(m) <> '' AND trim(source1_entity_id) IN (SELECT entity_id FROM s1)""")
        con.execute("CREATE TABLE ntrue AS SELECT s1_id, count(*) nt FROM gt GROUP BY 1")

    tmp = f"{w}/duckdb_tmp"

    def build_pred(expr: str):
        """Streaming filter to disk first (148M rows never held in RAM), then
        one-owner resolution on the much smaller predicted set."""
        con.execute(f"""COPY (SELECT s1_id, cand_id, bit_count(mask) AS nb FROM cand WHERE {expr})
                        TO '{tmp}/pred_raw.parquet' (FORMAT parquet)""")
        if args.one_owner:
            con.execute(f"""COPY (SELECT s1_id, cand_id FROM read_parquet('{tmp}/pred_raw.parquet')
                              QUALIFY row_number() OVER (PARTITION BY cand_id ORDER BY nb DESC, s1_id) = 1)
                            TO '{tmp}/pred.parquet' (FORMAT parquet)""")
        else:
            con.execute(f"""COPY (SELECT s1_id, cand_id FROM read_parquet('{tmp}/pred_raw.parquet'))
                            TO '{tmp}/pred.parquet' (FORMAT parquet)""")
        con.execute(f"CREATE OR REPLACE VIEW pred AS SELECT * FROM read_parquet('{tmp}/pred.parquet')")
        n = con.execute("SELECT count(*) FROM pred").fetchone()[0]
        print(f"  predicted pairs: {n:,}", flush=True)

    for name, expr in rules.items():
        build_pred(expr)
        if args.gt:
            r = con.execute("""
                WITH p AS (SELECT p.s1_id, count(*) np,
                                  sum(CASE WHEN g.s1_id IS NOT NULL THEN 1 ELSE 0 END) tp
                           FROM pred p LEFT JOIN gt g USING (s1_id, cand_id) GROUP BY 1),
                     e AS (SELECT s.entity_id, coalesce(p.np,0) np, coalesce(p.tp,0) tp, coalesce(t.nt,0) nt
                           FROM s1 s LEFT JOIN p ON s.entity_id = p.s1_id LEFT JOIN ntrue t ON s.entity_id = t.s1_id)
                SELECT avg(CASE WHEN nt = 0 AND np = 0 THEN 1.0
                                WHEN nt = 0 OR np = 0 OR tp = 0 THEN 0.0
                                ELSE 1.25 * (tp/np) * (tp/nt) / (0.25 * (tp/np) + (tp/nt)) END),
                       sum(tp) / nullif(sum(np), 0), sum(tp) / nullif(sum(nt), 0),
                       avg(CASE WHEN np = 0 THEN 1.0 ELSE 0.0 END)
                FROM e""").fetchone()
            print(f"{name}: macro F0.5 = {r[0]:.4f}   pair precision = {100*(r[1] or 0):.1f}%   "
                  f"pair recall = {100*(r[2] or 0):.1f}%   S1 predicted empty = {100*r[3]:.1f}%")
        if args.write:
            con.execute(f"""COPY (
                SELECT s.entity_id AS source1_entity_id,
                       coalesce(string_agg(p.cand_id, ',' ORDER BY p.cand_id), '') AS matched_entity_ids
                FROM s1 s LEFT JOIN pred p ON s.entity_id = p.s1_id
                GROUP BY s.entity_id ORDER BY 1
              ) TO '{args.write.replace(chr(92), '/')}' (DELIMITER '\t', HEADER, QUOTE '')""")
            print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
