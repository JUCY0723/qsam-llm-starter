# Replication Package (Anonymized) — For Review

This package reproduces two pipelines without changing any existing files:
(1) sentence-level sentiment scoring + calibration
(2) holder/target identification

> Double-anonymized: no author/affiliation, no public GitHub/DOI links in this stage.

## Quickstart (Windows / PowerShell)

# (optional) new virtual env
python -m venv .venv
.\.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# ① Sentiment scoring → calibration  (TRAIN & TEST)
python .\scripts\Qwen.py               --input .\data_min\sentiment_train.csv --output .\outputs\sentiment_train_with_qwen_pred.csv
python .\scripts\Qwen.py               --input .\data_min\sentiment_test.csv  --output .\outputs\sentiment_test_with_qwen_pred.csv

python .\scripts\Qwen_correction.py --train-pred data_min\sentiment_train_with_qwen_pred.csv --test-pred data_min\sentiment_test_with_qwen_pred.csv --human data_min\sentiment_test.csv --merge-key sentence --out-dir outputs --test-out sentiment_test_super_calibrated.csv

# ② Holder / Target identification
python .\scripts\qwen_h.py             --input .\data_min\holder_target_test.csv  --output .\outputs\holder_target_test_with_qwen_pred.csv

## Expected inputs (already in data_min/)
- sentiment_train.csv, sentiment_test.csv  # columns: id, text, ...
- holder_target_train.csv, holder_target_test.csv  # columns: id, text, ...

## Outputs (will be written to outputs/)
- sentiment_train_with_qwen_pred.csv,  sentiment_test_with_qwen_pred.csv
- sentiment_test_super_calibrated.csv
- holder_target_test_with_qwen_pred.csv

Notes
- No copyrighted full texts included. Remove personal paths/metadata before zipping.
- Public DOI/GitHub links will be added after acceptance (final version).
