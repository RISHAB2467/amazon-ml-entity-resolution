# blocking_v2: profiling, blocking and candidate generation (Person A)

Reads the **raw** challenge TSVs (`entity_id, business_name, business_address, country`) directly. Normalisation happens on the fly, and the raw files are never modified. Works on Windows, Linux and macOS.

Put this folder at `code/business_entity_resolution/src/blocking_v2/` and run everything from the repo root.

```powershell
pip install "duckdb>=1.1" jellyfish rapidfuzz pyarrow pandas psutil
```

## Steps (PowerShell, one line each)

```powershell
# 0. Profile the data (5-15 min): IDs, missing values, countries, ground-truth distribution, real examples
python code/business_entity_resolution/src/blocking_v2/profile_data.py --data dataset --out output/profile

# 1. Blocking keys for TRAIN (about 15-25 min)
python code/business_entity_resolution/src/blocking_v2/build_keys.py --s1 dataset/train/train_source1.tsv --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv --out output/blocking_train

# 2. Run the blockers + union (DuckDB spills to disk; memory = ~50% of your RAM)
python code/business_entity_resolution/src/blocking_v2/run_blocking.py --work output/blocking_train --memory 6GB

# 3. Measure recall against ground truth
python code/business_entity_resolution/src/blocking_v2/evaluate_blocking_v2.py --work output/blocking_train --gt dataset/train/train_ground_truth.tsv --memory 6GB

# 4. Same blocking for TEST (no evaluation: no labels)
python code/business_entity_resolution/src/blocking_v2/build_keys.py --s1 dataset/test/test_source1.tsv --s2 dataset/test/test_source2.tsv --s3 dataset/test/test_source3.tsv --out output/blocking_test
python code/business_entity_resolution/src/blocking_v2/run_blocking.py --work output/blocking_test --memory 6GB
```

### Dev sample for Person B (5% of S1, all vendor records)

```powershell
python code/business_entity_resolution/src/blocking_v2/build_keys.py --s1 dataset/train/train_source1.tsv --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv --out output/blocking_dev --s1-sample 0.05
python code/business_entity_resolution/src/blocking_v2/run_blocking.py --work output/blocking_dev --memory 6GB
python code/business_entity_resolution/src/blocking_v2/evaluate_blocking_v2.py --work output/blocking_dev --gt dataset/train/train_ground_truth.tsv --memory 6GB
```

The sample is deterministic (hash of `entity_id`), so A and B get identical files.

## Handoff files (in each `--work` folder)

| File | Columns | Used by |
|---|---|---|
| `candidates.parquet` | s1_id, cand_id, source (2/3), mask (bit per blocker) | B: features |
| `keys_s1.parquet`, `keys_s2.parquet`, `keys_s3.parquet` | entity_id, source, country, name_norm, address_norm, hnum, postcode, keys | B: features (normalised text) |
| `candidate_pairs.tsv` | source1_entity_id, candidate_entity_ids | official submission file |
| `blocking_report.md` | recall per blocker and union | documentation |

`mask` bit order is listed in `blocking_run_stats.json` → `_union.bit_order`. "Which blockers found this pair" makes a useful feature.

## Blockers

| Blocker | Key (within the same country) | Catches |
|---|---|---|
| compact | Core name without spaces, legal suffix stripped (also glued "…limited"), digit look-alikes fixed | suffix, spacing, inc/incorporated, re1iable |
| sorted | Sorted set of singularised tokens | word order, plurals |
| prefix / suffix | First / last 8 characters of the compact name | one typo anywhere |
| phonetic | Metaphone of each token | spelling by sound |
| rare1 / rare2 | Rarest name token / two rarest tokens | distinctive shared words |
| addr / addrname | House number + street name (+ first 2 letters of name) | renamed businesses at the same site |
| postname / postrare | Postcode or PIN + first 4 letters / rarest token | landmark addresses ("Near SBI ATM, Pune 411001") |

France (test only): accents folded, SARL/SAS/SASU/EURL/SCI/SNC/Cie treated as legal forms, "rue/bd/chemin" and "de/la/du" skipped in street names, 5-digit postcodes recognised. Country is only compared for equality, so new country labels work unchanged.

Block caps (in `BLOCKERS` at the top of `run_blocking.py`) drop over-generic keys. A higher cap gives more recall and more candidates. Change one cap at a time and re-measure with step 3.

## Low-memory final union + optional trim (for the 148M-pair test set)

`run_blocking.py` writes one `pairs_<blocker>.parquet` per blocker, then unions them. On the full test set that union can exhaust RAM. `finalize_union.py` redoes ONLY the union, in hash buckets of s1_id, from the existing pair files (no blocker is re-run):

```powershell
# 1. learn blocker weights + see recall vs top-K on the labelled train sample
python code/business_entity_resolution/src/blocking_v2/trim_eval.py --work output/blocking_train15 --gt dataset/train/train_ground_truth.tsv --memory 4GB
# 2a. test union, no trim
python code/business_entity_resolution/src/blocking_v2/finalize_union.py --work output/blocking_test --memory 4GB
# 2b. or with trim (apply the SAME K to train15 so B trains on the same distribution)
python code/business_entity_resolution/src/blocking_v2/finalize_union.py --work output/blocking_train15 --memory 4GB --topk 50 --weights output/blocking_train15/trim_weights.json
python code/business_entity_resolution/src/blocking_v2/finalize_union.py --work output/blocking_test --memory 4GB --topk 50 --weights output/blocking_train15/trim_weights.json
```
Output: `candidates/part_XX.parquet` (read as `candidates/*.parquet`) and the official `candidate_pairs.tsv` with every S1.
