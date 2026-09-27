# ML Challenge 2026: Business Entity Resolution, Solution

**Team Name:** [Your Team Name]
**Team Members:** [Person A name], [Person B name]
**Submission Date:** [Date]

> Placeholders marked **«TODO»** are filled once the model run finishes (validation score, cascade K, final candidate count).

---

## 1. Executive Summary

We solve Source 1 → Source 2/3 entity resolution with a three-stage pipeline built for scale on a single 8 GB laptop:
1. **Multi-key blocking.** 14 complementary blockers, run as out-of-core DuckDB equi-joins within each country.
2. **A learned candidate filter.** A LightGBM model on 49 string, address and blocking-evidence features; only the top-K candidates per Source 1 entity survive.
3. **A final classifier with a decision rule tuned for macro F0.5.** It enforces the "one owner per vendor record" property, which we measured to hold in 100% of the training labels.

Key innovations:
- **Address-rarity blocking keys** that do not depend on house numbers (12.6% of true pairs have conflicting numbers). These raised blocking recall from 85.3% to 95.7%.
- **Blocker-provenance bits** used as model features.
- **A metric-aware decision rule** (a threshold plus a "best candidate" rule, then one-owner resolution) tuned on out-of-fold predictions.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA)

All statistics were measured on the provided data with DuckDB, streaming from disk.

| Item | Value |
|---|---|
| Train S1 / S2 / S3 records | 2,206,821 / 5,034,616 / 5,285,603 |
| Test S1 / S2 / S3 records | 1,732,544 / 4,887,273 / 5,082,316 |
| True S1 → S2/S3 pairs (train) | 7,638,365 (mean 3.46 per S1, max 11) |
| S1 with no match (singletons) | 123,247 (5.6%) |
| Vendor records matching nothing | ~26% (pure distractors) |
| Vendor records owned by more than one S1 | **0**, so the one-owner property holds exactly |
| Cross-country true pairs | **0** of 7,638,365 |
| Countries | Train: US, India. Test adds **France** (259,452 of 1,732,544 test S1; no labels) |
| Missing names | S1 0; S2 2; S3 13 |
| Missing addresses | S1 0; S2 168,967; S3 175,916 |
| True pairs with conflicting house numbers | 12.6% |
| Exact normalised-name blocking (first attempt) | pair recall 21.85%; all matches found for only 2.8% of S1 |

**Noise patterns observed in missed true pairs.** We inspected the missed pairs after every blocking iteration and built each fix from real examples:

- **Legal forms anywhere and shuffled:** "laxmi foods private limited" ↔ "private laxmi foods limited"; glued forms ("limitedlimited", "ashishserviceslimited").
- **Truncated vendor names:** "royal impex private limited" ↔ "impex"; "shiv healthcare private limited" ↔ "healthcare".
- **Character noise:** digit look-alikes ("re1iable"), typos, duplicated tokens ("a1pha alpha"), honorifics ("sri", "shri"), appended words ("enterprises", "partners"), phone numbers and "www … com" inside names, DBA names ("solquo dba delhi producer").
- **Address noise:** corrupted house numbers (4600 ↔ 4598, 8770 ↔ 877), corrupted ordinals ("217th" ↔ "217nd", "6th" ↔ "sixth"), "street" → "saint", leading zeros ("005319"), "null" tokens, component reordering, Indian landmark addresses ("near SBI ATM") with PIN codes, missing vendor addresses.
- **Generic names reused by hundreds of businesses** ("urgent care clinic", "commission on finance") and often paired with a missing vendor address. These are effectively unresolvable and bound the achievable recall.
- **French test records** (from a sample): SAS / SASU / SARL legal forms, accents, "rue / allée / bis" address structure, 5-digit postcodes.

### 2.2 Solution Strategy

**Approach type:** Hybrid: rule-based multi-key blocking, then a learned candidate filter (LightGBM), then a final LightGBM classifier with a metric-aware decision rule and one-owner post-processing.

**Core innovation:** Blocking keys built from the *rarest address words* and *address number × rare name word* combinations. They survive corrupted house numbers, truncated vendor names and reordered addresses, and they lifted recall on the hardest country (India) from 76.6% to over 90%. Combined with blocker-provenance features and exploiting the exact one-owner property, this gives a strong, fully reproducible pipeline that runs out-of-core on commodity hardware.

**Design principles:**
- Recall first at blocking; precision at decision time. F0.5 weights precision 2×.
- Every decision is validated on held-out data grouped by S1: 5 folds, `fold = hash(s1_id) % 5`.
- **Country-agnostic modelling.** Country is used only as an equality condition in blocking (0 cross-country pairs). It is never a model feature, so French records are scored by the same logic as US and Indian ones.
- No external data, APIs or geocoding. Only the provided files and MIT/Apache-licensed libraries (LightGBM is MIT). No pretrained language model is used.

---

## 3. Candidate Generation (Blocking)

### 3.1 Normalisation (per record, streamed in 500k-row chunks)

- **Text:** lowercase, accents folded (NFKD), "&" → "and", punctuation removed.
- **Name tokens:** digit look-alikes fixed inside mixed tokens (1→l, 0→o, 3→e, …), abbreviations expanded (intl, svc, mgmt, ctr/centre, …). We drop stop words (English and French), honorifics, single letters, phone and ID numbers, and web tokens. Legal forms are removed **anywhere** in the name (US, India and France sets), including when glued onto a token. Duplicate tokens are dropped, and "x dba y" yields keys for both parts.
- **Addresses:** Indian PINs "411 001" joined, postcode detected (last 5- or 6-digit token), house number (first number up to 5 digits, leading zeros stripped, not the postcode), ordinals normalised ("217nd" → 217, "sixth" → 6), junk tokens removed. The street-name token skips street types, directions, unit words and French articles ("12 bis rue de la paix" → 12 + paix).
- **Rarity:** two document-frequency tables over all three sources: name tokens and alphabetic address words.

### 3.2 Blocking keys (14 blockers, union)

Every blocker is an equi-join of S1 against S2 ∪ S3 on *(country, key)*. Blocks larger than a cap (vendor records per key) are dropped, which removes over-generic keys. List-valued keys are unnested: a pair is a candidate if any key matches.

Measured on a 15% sample of train S1 (331,398 S1, 1,147,448 true pairs):

| Blocker | Key | Cap | Pairs | Recall % | Precision % | Cand./S1 |
|---|---|---:|---:|---:|---:|---:|
| compact | Core name without spaces (+ both DBA parts) | 200 | 7,186,444 | 56.92 | 9.09 | 21.7 |
| sorted | Sorted set of singularised core tokens | 200 | 6,950,442 | 53.17 | 8.78 | 21.0 |
| prefix | First 8 characters of the compact name | 100 | 3,329,852 | 37.41 | 12.89 | 10.0 |
| suffix | Last 8 characters of the compact name | 100 | 1,974,969 | 17.93 | 10.42 | 6.0 |
| phonetic | Metaphone of each core token | 100 | 4,106,014 | 50.16 | 14.02 | 12.4 |
| rare1 | Rarest name token (DF ≤ 2,000) | 100 | 3,707,176 | 19.42 | 6.01 | 11.2 |
| rare2 | Every pair among the 3 rarest name tokens | 100 | 5,040,514 | 51.69 | 11.77 | 15.2 |
| addr | House number + street-name token | 50 | 1,678,132 | 55.21 | 37.75 | 5.1 |
| addrname | addr + first 2 letters of the name | 100 | 698,203 | 50.75 | 83.40 | 2.1 |
| postname | Postcode/PIN + first 4 letters of the name | 100 | 7,290 | 0.52 | 82.63 | 0.02 |
| postrare | Postcode/PIN + rarest name token | 100 | 6,098 | 0.45 | 84.04 | 0.02 |
| **apair** | Every pair among the 3 rarest address words | 30 | 3,776,921 | 62.55 | 19.00 | 11.4 |
| **nameaddr** | 3 rarest name tokens × 2 rarest address words | 30 | 5,126,875 | **74.06** | 16.57 | 15.5 |
| **numname** | First 3 address numbers × 2 rarest name tokens | 30 | 2,950,538 | 58.33 | 22.68 | 8.9 |
| **Union** | | | **28,315,968** | **95.66** | 3.88 | **85.4** |

**Blocking iterations** (each driven by an analysis of the missed true pairs):

| Version | Change | Pair recall | India | US | Cand./S1 |
|---|---|---:|---:|---:|---:|
| v0 | Exact normalised name | 21.85% | | | 9.9 |
| v1 | 11 name/address/postcode blockers | 85.34% | 76.62% | 91.10% | 52.9 |
| v3 | + address-rarity keys (apair, nameaddr), legal forms anywhere, glued forms, ordinals | 94.85% | 90.74% | 97.57% | 79.2 |
| v4 | + numname, leading zeros, "saint", phone/web tokens, appended words | **95.66%** | | | 85.4 |

(v1 and v3 were measured on a 5% sample, v4 on a 15% sample.)

### 3.3 Candidate pairs generated

- **Test blocking output:** 148,561,977 pairs for 1,732,544 S1 (85.75 per S1). All S1 were processed; 216 had no candidate and get an empty list.
- **Reduction ratio at blocking:** 1 − 148.6M / (1.73M × 9.97M) = **99.99914%** of all S1 × vendor pairs eliminated.
- **Final candidate set (`candidate_pairs.tsv`):** after the learned filter (Section 4.4), **«TODO: N» pairs = «TODO: x» per S1**, keeping top-K = «TODO» per S1 with p1 ≥ «TODO». This is exactly the set the final classifier runs inference on.

### 3.4 How we ensured true matches were not lost

- **Recall measured, not assumed:** every blocker and the union were evaluated against the ground truth (pair recall, recall by country and source, share of S1 with all matches found, recall found only by that blocker), on deterministic S1 samples (hash-based, 5% and 15%).
- **Missed-pair analysis after each iteration** (a sample sorted by name similarity, with categories: house-number agreement, missing vendor address, name-similarity bands). Every new key targets an observed failure mode.
- **Caps drop only over-generic blocks,** and every generic name is still reachable through name × address keys.
- **The learned filter's K is chosen on out-of-fold predictions** so that it costs at most «TODO» recall points.

### 3.5 Scalability engineering (8 GB RAM laptop)

- **Key building:** chunked pandas, 1.8 GB peak RSS, about 60 min for train S1 sample + all train vendors.
- **Blocking:** out-of-core DuckDB joins (memory-limited, spilling to disk), one Parquet file per blocker.
- **Union:** memory-safe, in 16 hash buckets of `s1_id` (`finalize_union.py`, 36 min for 148.6M pairs). The single-pass union ran out of memory.
- **Submission writing and checking:** also streaming and bucketed (`rule_baseline.py`, `decide.py`, `check_submission.py`, about 2 GB).

---

## 4. Matching Model

### 4.1 Features (49; identical code for train and test; no country feature)

- **Name (15):** ratio, token-set, token-sort, partial ratio and Jaro-Winkler on normalised names; ratio, token-set and partial on *core* names (legal and stop words removed); compact-name equality; core-token subset flag (truncated names); core-token Jaccard; first-token equality; token counts; length difference.
- **Address (9):** vendor address missing; token-set, ratio and partial on addresses; house number equal / conflict / missing; postcode equal / conflict; Jaccard of all numbers in the address.
- **Blocking evidence (15):** one bit per blocker that produced the pair, plus the number of blockers.
- **Per-S1 context (9):** number of candidates; rank of and gap to the S1's best name score, core-name score and address score; number of strongly similar candidates.
- **Other (1):** source (S2 / S3).

### 4.2 Model type

- **LightGBM binary classifier** (MIT license): num_leaves 127, learning rate 0.05, min_data_in_leaf 100, feature/bagging fraction 0.8, early stopping (100 rounds).
- **Validation:** 5-fold GroupKFold by S1 (`hash(s1_id) % 5`), trained on the 15% train sample (331,398 S1, 28.3M pairs).
- **Sampling:** all positives plus 20% of negatives with weight 5, which keeps memory in bounds without biasing probabilities.
- **Out-of-fold predictions** for every training pair: each is scored by the model that never saw its S1.

### 4.3 Decision rule (threshold selection)

Tuned on out-of-fold predictions for **macro F0.5 over all S1, including singletons**. The recall denominator includes true pairs lost at blocking.

1. Keep a pair if `p ≥ t`, or if it is the S1's best candidate and `p ≥ t_best`. This handles single-match S1 without flooding multi-candidate S1.
2. **One-owner resolution:** a vendor record predicted for several S1 is kept only for the highest-probability S1. On the rule baseline this raised precision from 81.0% to 84.2% with no recall loss.
3. `t` and `t_best` come from a grid search followed by refinement (`decide.py --tune`). Chosen: t = «TODO», t_best = «TODO».

### 4.4 Cascade (candidate-set reduction)

- **Stage 1:** the LightGBM above scores all blocking candidates. The top-K per S1 with p1 ≥ ε survive. K and ε are chosen on out-of-fold predictions (`cascade.py eval`), with recall loss of «TODO» points.
- **Stage 2:** a small LightGBM (p1, rank, gap and ratio to the best, second-best p1, number of survivors, sum of p1, number of high-confidence candidates, blocking candidate count), trained on out-of-fold survivors with the same folds. It produces the final probabilities; the decision rule of 4.3 is then applied.
- `candidate_pairs.tsv` = the stage-2 input set.

---

## 5. Results & Error Analysis

| System (15% train sample, out-of-fold, macro F0.5) | F0.5 | Pair precision | Pair recall |
|---|---:|---:|---:|
| Rule baseline R4 (high-precision blocker combinations) | 0.7596 | 81.0% | 68.5% |
| R4 + one-owner | **0.7631** | 84.2% | 68.5% |
| LightGBM + tuned decision rule + one-owner | «TODO» | «TODO» | «TODO» |
| Cascade (stage 2) + decision rule + one-owner | «TODO» | «TODO» | «TODO» |
| Blocking recall ceiling | | | 95.66% |

**F0.5 score (macro, best validation):** «TODO»
**Public leaderboard:** R4 + one-owner = «TODO»; model = «TODO»

**Common false positives (wrong merges):**
- Different businesses at the same address (shared buildings, "we work" style addresses).
- Businesses with the same generic name in the same city ("urgent care clinic").
- Branches of one brand at nearby numbers.
- Mitigations: one-owner resolution, and house-number / postcode conflict features.

**Common false negatives (missed matches):**
- Pairs never generated by blocking (4.3%): mostly generic, reused names combined with a missing vendor address; the text can't disambiguate them among hundreds of identical names.
- Heavily truncated vendor names ("energy", "healthcare") when the address is also rewritten.
- Glued or garbled tokens ("srand co", "coaditya and").

«TODO: 2–3 lines from error analysis of the worst out-of-fold S1»

---

## 6. Conclusion

A recall-first, evidence-driven blocking design (14 complementary keys, with new address-rarity keys that are robust to corrupted house numbers) reached a 95.7% blocking recall ceiling. It runs entirely out-of-core on an 8 GB laptop and cuts the S1 × vendor space by 99.999%. A LightGBM matcher with blocker-provenance and per-S1 context features, a learned filter that shrinks the candidate set, a decision rule tuned for macro F0.5 and exact one-owner resolution turn those candidates into precise matches. Lessons: measure recall per blocker and read the misses after each iteration; exploit structural properties of the data (one owner, same country); design every step to stream through memory from the start.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── README.md, requirements.txt
└── src/
    ├── blocking_v2/
    │   ├── profile_data.py          # EDA / data profile (DuckDB)
    │   ├── build_keys.py            # normalisation + blocking keys -> keys_s{1,2,3}.parquet
    │   ├── run_blocking.py          # 14 blockers (DuckDB joins) -> pairs_<blocker>.parquet
    │   ├── finalize_union.py        # memory-safe bucketed union -> candidates/*.parquet
    │   ├── evaluate_blocking_v2.py  # recall per blocker / union, missed-pair analysis
    │   ├── trim_eval.py             # blocker weights (analysis)
    │   ├── rule_baseline.py         # rule-based fallback submission
    │   ├── cascade.py               # learned filter (top-K) + stage-2 model
    │   ├── decide.py                # decision rule tuning + matching_results.tsv writer
    │   └── check_submission.py      # low-memory check of all submission rules
    └── models/
        ├── features_lib.py          # 49 pair features (shared by train and test)
        ├── make_train_features.py   # train features + labels + folds
        ├── train.py                 # 5-fold LightGBM + out-of-fold predictions
        └── predict_test.py          # test scoring (bucketed, resumable)
```

**Reproduce end to end** (from the repository root; every step streams and runs within about 2–6 GB of RAM):
```
# 1. Blocking: train sample (15%) and full test
python src/blocking_v2/build_keys.py --s1 dataset/train/train_source1.tsv --s2 dataset/train/train_source2.tsv --s3 dataset/train/train_source3.tsv --out output/blocking_train15 --s1-sample 0.15
python src/blocking_v2/run_blocking.py --work output/blocking_train15 --memory 4GB
python src/blocking_v2/build_keys.py --s1 dataset/test/test_source1.tsv --s2 dataset/test/test_source2.tsv --s3 dataset/test/test_source3.tsv --out output/blocking_test
python src/blocking_v2/run_blocking.py --work output/blocking_test --memory 4GB
python src/blocking_v2/finalize_union.py --work output/blocking_test --memory 4GB
# 2. Model
python src/models/make_train_features.py --work output/blocking_train15 --gt dataset/train/train_ground_truth.tsv --out output/model/train_feats --buckets 32
python src/models/train.py --feats output/model/train_feats --out output/model
python src/models/predict_test.py --work output/blocking_test --models output/model --out output/model/test_preds --buckets 96
# 3. Cascade + decision
python src/blocking_v2/cascade.py train --oof output/model/oof_preds.parquet --s1-keys output/blocking_train15/keys_s1.parquet --gt dataset/train/train_ground_truth.tsv --k «K» --eps «EPS»
python src/blocking_v2/decide.py --preds output/model/stage2/oof_stage2.parquet --s1-keys output/blocking_train15/keys_s1.parquet --gt dataset/train/train_ground_truth.tsv --tune
python src/blocking_v2/cascade.py apply --preds "output/model/test_preds/*.parquet" --s1-keys output/blocking_test/keys_s1.parquet --k «K» --eps «EPS»      # -> output/candidate_pairs.tsv
python src/blocking_v2/decide.py --preds "output/model/stage2/test_stage2/*.parquet" --s1-keys output/blocking_test/keys_s1.parquet --t «T» --t-best «TB» --one-owner --write output/matching_results.tsv
python src/blocking_v2/check_submission.py --matching output/matching_results.tsv --candidates output/model/stage2/test_stage2 --candidate-tsv output/candidate_pairs.tsv --test-dir dataset/test
```

### B. Additional Results

**Runtime and memory (Windows laptop, 7.75 GB RAM):**

| Step | Data | Time | Peak memory |
|---|---|---:|---:|
| build_keys (train, 15% S1 + all vendors) | 10.65M records | 60 min | 1.8 GB |
| run_blocking (train15) | 28.3M pairs | minutes | ≤ 4 GB (DuckDB limit) |
| finalize_union (test) | 148.6M pairs, 16 buckets | 36 min | ≤ 4 GB |
| rule_baseline / check_submission (test) | 148.6M candidates | ~10 min | ~2 GB |
| Features / training / test prediction | 28.3M / 148.6M pairs | «TODO» | «TODO» |

**Ground-truth properties used:** 0 cross-country pairs (country-equality blocking is safe); 0 vendor records with more than one owner (one-owner resolution); 5.6% singletons (an empty prediction is a valid, rewarded decision).
