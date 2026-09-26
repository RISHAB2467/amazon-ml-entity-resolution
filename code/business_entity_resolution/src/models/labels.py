import argparse
import csv
import os

import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    con = duckdb.connect()

    # Read candidate pairs.
    candidates = f"""
        SELECT
            CAST(s1_id AS VARCHAR) AS s1_id,
            CAST(cand_id AS VARCHAR) AS cand_id,
            CAST(source AS INTEGER) AS source,
            CAST(mask AS INTEGER) AS mask
        FROM read_parquet('{args.candidates}')
    """

    # Read ground truth.
    #
    # matched_entity_ids is comma-separated.
    # Empty means this S1 has no true match.
    gt = f"""
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS s1_id,
            CAST(matched_entity_ids AS VARCHAR) AS matched_entity_ids
        FROM read_csv(
            '{args.ground_truth}',
            delim='\\t',
            header=true,
            quote='',
            escape='',
            columns={{
                'source1_entity_id': 'VARCHAR',
                'matched_entity_ids': 'VARCHAR'
            }}
        )
    """

    # Explode the ground truth into one row per true pair.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_pairs AS
        SELECT
            s1_id,
            TRIM(x) AS cand_id
        FROM ({gt}),
        UNNEST(string_split(matched_entity_ids, ',')) AS t(x)
        WHERE TRIM(x) <> ''
    """)

    # Label every candidate pair.
    con.execute(f"""
        COPY (
            SELECT
                c.s1_id,
                c.cand_id,
                c.source,
                c.mask,
                CASE
                    WHEN g.cand_id IS NOT NULL THEN 1
                    ELSE 0
                END AS label
            FROM ({candidates}) c
            LEFT JOIN gt_pairs g
                ON c.s1_id = g.s1_id
               AND c.cand_id = g.cand_id
        )
        TO '{args.out}'
        (FORMAT PARQUET)
    """)

    candidate_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{args.out}')"
    ).fetchone()[0]

    positives = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{args.out}') WHERE label = 1"
    ).fetchone()[0]

    total_true_pairs = con.execute(
        "SELECT COUNT(*) FROM gt_pairs"
    ).fetchone()[0]

    ceiling = (
        positives / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    print()
    print("=== LABEL REPORT ===")
    print(f"Candidate pairs : {candidate_count:,}")
    print(f"Positive pairs  : {positives:,}")
    print(f"All true pairs  : {total_true_pairs:,}")
    print(f"Blocking recall ceiling: {ceiling:.4%}")
    print()
    print(f"Wrote: {args.out}")

    con.close()


if __name__ == "__main__":
    main()