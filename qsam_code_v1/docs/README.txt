QSAM-LLM Code (v1)

- code/: Python scripts for sentiment & holder-target; no full news texts.
- models/: optional calibrators (.model/.pkl)
- docs/: licenses and this README

Environment
- Tested with Python ≥3.8 on Windows.
- If you have a requirements.txt, run: pip install -r requirements.txt

Quickstart (Windows / PowerShell)
1) python -m venv .venv && .\.venv\Scripts\activate
2) pip install -r requirements.txt   (if this file is absent, install as your scripts suggest)
3) example:
   python code\roberta1.py --input ..\qsam_data_v1\derived\sentiment_test.csv --out .\metrics_roberta.csv

Notes
- No copyrighted full news texts are included.
- Code under MIT; derived data (see companion dataset) under CC BY 4.0.
