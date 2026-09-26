"""
Low-memory submission checker (DuckDB, ~2 GB), for laptops where the official
stdlib validator runs out of RAM on a 148M-ID candidate_pairs.tsv.

Checks the official rules:
  matching_results.tsv : exact header; one row per test S1 (none missing, no
                         extras, no duplicate rows); IDs only S2-/S3- that exist
                         in the test files; no duplicate IDs within a list;
                         every matched ID is a candidate of that S1
  candidate_pairs.tsv  : exact header; one row per test S1
Run the official utils/validate_submission.py too if a machine with enough
RAM is available.

Usage:
  python code/business_entity_resolution/src/blocking_v2/check_submission.py --matching output/matching_results.tsv --candidates output/blocking_test/candidates --candidate-tsv output/candidate_pairs.tsv --test-dir dataset/test
"""
from __future__ import annotations

import argparse
import os

import duckdb


def rd(p: str, cols: str = "") -> str:
    return (f"read_csv('{p.replace(chr(92), '/')}', delim='\t', header=true, quote='', escape='', "
            f"all_varchar=true, null_padding=true)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidates", required=True, help="folder with part_*.parquet (or a .parquet file)")
    ap.add_argument("--candidate-tsv", default=None)
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--memory", default="2GB")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    con = duckdb.connect()
    tmp = os.path.join(os.path.dirname(args.matching) or ".", "duckdb_tmp").replace("\\", "/")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")

    problems = []
    td = args.test_dir.replace("\\", "/")

    # header
    with open(args.matching, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
    if header != "source1_entity_id\tmatched_entity_ids":
        problems.append(f"matching header is {header!r}")

    con.execute(f"CREATE TABLE s1 AS SELECT entity_id FROM {rd(td + '/test_source1.tsv')}")
    con.execute(f"CREATE TABLE m AS SELECT * FROM {rd(args.matching)}")
    n_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    n_rows, n_dist = con.execute("SELECT count(*), count(DISTINCT source1_entity_id) FROM m").fetchone()
    missing = con.execute("SELECT count(*) FROM s1 WHERE entity_id NOT IN (SELECT source1_entity_id FROM m)").fetchone()[0]
    extra = con.execute("SELECT count(*) FROM m WHERE source1_entity_id NOT IN (SELECT entity_id FROM s1)").fetchone()[0]
    print(f"test S1: {n_s1:,} | matching rows: {n_rows:,} | distinct: {n_dist:,} | missing: {missing:,} | extra: {extra:,}")
    if n_rows != n_dist: problems.append(f"{n_rows - n_dist:,} duplicate source1 rows")
    if missing: problems.append(f"{missing:,} test S1 missing from matching")
    if extra: problems.append(f"{extra:,} rows for S1 ids not in the test set")

    con.execute("""CREATE TABLE mx AS
        SELECT source1_entity_id AS s1_id, trim(x) AS cand_id
        FROM m, unnest(string_split(coalesce(matched_entity_ids, ''), ',')) AS t(x)
        WHERE trim(x) <> ''""")
    n_ids = con.execute("SELECT count(*) FROM mx").fetchone()[0]
    dup = con.execute("SELECT count(*) FROM (SELECT s1_id, cand_id FROM mx GROUP BY ALL HAVING count(*) > 1)").fetchone()[0]
    badpref = con.execute("SELECT count(*) FROM mx WHERE NOT (starts_with(cand_id,'S2-') OR starts_with(cand_id,'S3-'))").fetchone()[0]
    con.execute(f"""CREATE TABLE vend AS
        SELECT entity_id FROM {rd(td + '/test_source2.tsv')} UNION ALL
        SELECT entity_id FROM {rd(td + '/test_source3.tsv')}""")
    unknown = con.execute("SELECT count(*) FROM mx WHERE cand_id NOT IN (SELECT entity_id FROM vend)").fetchone()[0]
    c = args.candidates.replace("\\", "/")
    csrc = f"{c}/*.parquet" if os.path.isdir(args.candidates) else c
    not_cand = con.execute(f"""SELECT count(*) FROM mx ANTI JOIN read_parquet('{csrc}') p
                               ON mx.s1_id = p.s1_id AND mx.cand_id = p.cand_id""").fetchone()[0]
    s1_nonempty = con.execute("SELECT count(DISTINCT s1_id) FROM mx").fetchone()[0]
    print(f"matched IDs: {n_ids:,} | S1 with >=1 match: {s1_nonempty:,} ({100*s1_nonempty/max(n_s1,1):.1f}%) "
          f"| avg per S1: {n_ids/max(n_s1,1):.2f}")
    if dup: problems.append(f"{dup:,} duplicate IDs inside lists")
    if badpref: problems.append(f"{badpref:,} IDs without S2-/S3- prefix")
    if unknown: problems.append(f"{unknown:,} IDs not in test S2/S3")
    if not_cand: problems.append(f"{not_cand:,} matched IDs not among that S1's candidates")

    if args.candidate_tsv:
        with open(args.candidate_tsv, encoding="utf-8") as f:
            h2 = f.readline().rstrip("\r\n")
        if h2 != "source1_entity_id\tcandidate_entity_ids":
            problems.append(f"candidate header is {h2!r}")
        r = con.execute(f"""SELECT count(*), count(DISTINCT source1_entity_id)
                            FROM read_csv('{args.candidate_tsv.replace(chr(92), '/')}', delim='\t', header=true,
                                          quote='', escape='', all_varchar=true, null_padding=true,
                                          columns={{'source1_entity_id':'VARCHAR','candidate_entity_ids':'VARCHAR'}})""").fetchone()
        print(f"candidate_pairs.tsv rows: {r[0]:,} | distinct S1: {r[1]:,}")
        if r[0] != n_s1 or r[1] != n_s1:
            problems.append("candidate_pairs.tsv does not have exactly one row per test S1")

    print()
    if problems:
        print("FAIL")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        raise SystemExit(1)
    print("PASS (low-memory check of all official rules)")


if __name__ == "__main__":
    main()
