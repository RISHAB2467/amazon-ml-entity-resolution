# models/: Person B pipeline (features, LightGBM, test prediction)

Put these files in `code/business_entity_resolution/src/models/`. Run from the repo root, one at a time.

```powershell
pip install lightgbm scikit-learn rapidfuzz duckdb pyarrow pandas numpy

# 1. train features + labels + folds (about 15-30 min, 32 buckets, resumable)
python code/business_entity_resolution/src/models/make_train_features.py --work output/blocking_train15 --gt dataset/train/train_ground_truth.tsv --out output/model/train_feats --buckets 32

# 2. 5-fold LightGBM + OOF predictions for all 28.3M train pairs (about 20-40 min)
python code/business_entity_resolution/src/models/train.py --feats output/model/train_feats --out output/model

# 3. tune the decision rule on OOF: the printed BEST line is your score to compare with 0.763
python code/business_entity_resolution/src/blocking_v2/decide.py --preds output/model/oof_preds.parquet --s1-keys output/blocking_train15/keys_s1.parquet --gt dataset/train/train_ground_truth.tsv --tune

# 4. score every test pair (1-2 h, 96 buckets, resumable: rerun the same command after a crash)
python code/business_entity_resolution/src/models/predict_test.py --work output/blocking_test --models output/model --out output/model/test_preds --buckets 96

# 5. write + check the submission with the BEST values from step 3
python code/business_entity_resolution/src/blocking_v2/decide.py --preds "output/model/test_preds/*.parquet" --s1-keys output/blocking_test/keys_s1.parquet --t <T> --t-best <TB> --one-owner --write output/matching_results.tsv
python code/business_entity_resolution/src/blocking_v2/check_submission.py --matching output/matching_results.tsv --candidates output/blocking_test/candidates --test-dir dataset/test
```

Low RAM (8 GB): add `--neg-frac 0.1` to train.py. Features: 49 (name x10, core-name x8, address x9, blocker bits x15, per-S1 context x8, source). No country feature, on purpose (France is test-only).
