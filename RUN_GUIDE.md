# How to run the code, what to check, and how to record metrics

Everything here is free. No paid model, API or dataset is needed.

## 0. What was tested and what was not

Tested by the author in a sandbox: data download, dataset building, the firewall, the cross-validated evaluation with the keyword baseline, threshold sweep, latency, the judge plumbing in dry mode (stub replies), and the offline tests.

NOT tested by the author (no access to Hugging Face from the sandbox): loading JBB benign prompts, and every neural baseline (protectai, granite, llamaguard, promptguard). Their code follows the official model cards. If one fails, the run skips it with a warning and continues. Send me the error text and I will fix it.

## 1. Setup

Python 3.10 or newer.

```bash
cd llm_firewall
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Light install if you only want our system and the keyword baseline (no torch): install numpy pandas scipy scikit-learn matplotlib joblib requests streamlit datasets.

## 2. Step by step

```bash
python tests/test_smoke.py                 # 1. offline sanity tests, should print five "ok" lines
python data_prep.py                        # 2. downloads datasets, builds data/prompts.csv and data/attacks.csv
python evaluate.py --quick --baselines keyword      # 3. fast dry run (about a minute)
python evaluate.py --baselines keyword,protectai    # 4. real run with the first neural baseline
python evaluate.py --baselines keyword,protectai,granite --latency   # 5. full run (see hardware note)
```

Optional gated baselines: accept the licence on the model pages for meta-llama/Llama-Guard-3-1B and meta-llama/Llama-Prompt-Guard-2-86M, run `huggingface-cli login`, then add `llamaguard,promptguard` to `--baselines`.

Hardware note for granite: the 3B model needs about 12 GB RAM on CPU (slow, roughly an hour or more for about 3,700 prompts) or a GPU. A free Google Colab GPU session is the easiest way. Results are cached in `results/cache_<name>.json`, so an interrupted run resumes.

Make the demo and the system-level judge:

```bash
python train_final.py                      # demo model trained on all data (not for reported metrics)
streamlit run app.py                       # demo UI

# free API key (no card): https://console.groq.com/keys
export GROQ_API_KEY=gsk_xxx                # Windows PowerShell: $env:GROQ_API_KEY="gsk_xxx"
python judge.py --systems keyword,protectai,ours_full,ours_no_routing --k 30 --export_audit
python judge.py --kappa                    # after you fill results/judge_audit.csv by hand
```

Groq free tier has per-minute and per-day caps (checked from third-party summaries of Groq's docs, so confirm on your account's limits page). judge.py paces itself and caches every call. If it stops on a daily limit, run it again the next day.

## 3. What to check after each step

After `data_prep.py` the printed counts should be about: medsafety 1800, jbb_harmful 100, jbb_benign 100 (only if Hugging Face loading worked), medquad 600, dualuse 80; attacks PAIR 82, GCG 100, JBC 100, RS 100. If jbb_benign is missing, install `datasets` and check internet access to huggingface.co, then rerun.

Sanity values from the author's run (keyword baseline, full data): hard-block on medical harm about 2%, on JBB harmful about 17%, independent attack success about 58%, character attack success 100%. If your keyword numbers are far from these, something is wrong with the data.

Checks on `results/summary.csv`:
1. `ours_full` character-attack success should be close to 0%, and `ours_no_norm` clearly higher. If not, normalization is broken.
2. `ours_rules_only` should be weak on plain harmful prompts. That is expected.
3. Compare `ours_full` with `ours_ml_only`. If they are close, your gain is mostly the classifier. Say so honestly.
4. Open `predictions.csv` and read at least 30 wrong decisions by hand (benign blocked, harmful allowed). This error analysis is worth marks.
5. In `mcnemar.csv`, p below 0.05 means the difference between you and that baseline is unlikely to be chance for that subset.
6. `tradeoff.png` should show your curve and the baseline points. A baseline above and left of your curve beats you on that trade-off. Report it if so.
7. Confirm `run_meta.json` has the settings you intend to report.

## 4. What each output file contains

| File | Use |
|------|-----|
| results/summary.csv | main table with 95% confidence intervals |
| results/metrics_long.csv | every number in machine readable form (for charts) |
| results/mcnemar.csv | significance tests, ours vs each baseline |
| results/threshold_sweep.csv, tradeoff.png | safety vs over-blocking curve |
| results/predictions.csv, items.csv | every decision, for error analysis |
| results/latency.csv | with `--latency` |
| results/judge_summary.csv, judge_rows.csv | system-level metrics from judge.py |
| results/judge_audit.csv | 50 rows for your manual labels, then `--kappa` |
| results/run_meta.json | settings, date, versions |

Metric meaning: hard-block rate is the share of BLOCK. Intervene rate is BLOCK plus SANITIZE. Attack success rate is the share that got ALLOW. Over-block is BLOCK on a benign prompt. Friction is BLOCK plus SANITIZE on a benign prompt. Strict F1 counts only BLOCK as a detection, which is the fair way to compare with binary baselines.

## 5. How to record your before and after numbers

1. Run the final evaluation once with every baseline and keep the whole `results/` folder (copy it to `results_final_YYYYMMDD/`). Never overwrite it.
2. Copy numbers from `summary.csv` into the table in PROJECT_PLAN.md section 10, with the confidence intervals.
3. "Before" columns are the baselines. "After" columns are `ours_ml_only` and `ours_full`.
4. Put the exact commit, command lines and `run_meta.json` in an appendix.
5. Do not tune thresholds on the test results. If you want different thresholds, create a separate dev split first, tune there, then run the test once.

## 6. Customizing

- Add your own dual-use questions to `data/dualuse_benign.csv` (columns text,category), then rerun `data_prep.py`. Use several categories so cross-validation has enough groups.
- Rules and thresholds are at the top of `firewall.py`.
- To test a different free LLM on Groq set `LLM_MODEL` and `JUDGE_MODEL` environment variables.

## 7. Troubleshooting

- `HTTP 403/429` while downloading: wait and rerun, downloads resume.
- `could not load JBB from HuggingFace`: install `datasets`, check internet, rerun `data_prep.py`.
- Baseline skipped with a gated-repo error: accept the licence on the model page and log in with `huggingface-cli login`.
- Out of memory on granite: use Colab GPU or drop granite from `--baselines`.
- Windows console shows odd characters: set `PYTHONUTF8=1`.
- Streamlit says run train_final.py first: do that after data_prep.py.

## 8. Ethics

The datasets contain harmful requests. Use them for research only, keep them out of public demos, and do not publish generated harmful outputs.
