# Reproducing and extending DLinear (COMP8240)

Ishraq Ahmed (48937851), Macquarie University, COMP8240 Applications of Data Science, Session 2 2026.

This repository reproduces **DLinear** from *Are Transformers Effective for Time Series Forecasting?*
(Zeng et al., AAAI 2023, [arXiv 2205.13504](https://arxiv.org/abs/2205.13504), code at
[cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)) and tests it on new data:

1. **Replication** of the authors' results on ETTh1, Weather and ILI (`replication/`), including 5 repeated runs.
2. **Existing dataset not in the paper:** Beijing Multi-Site Air Quality, Dongsi station (`beijing/`).
3. **Constructed dataset:** nine controlled synthetic series (trend + daily + weekly cycles + noise, with an
   optional level shift), each with a known noise-free signal (`synthetic/`).
4. **LLM-as-judge:** two LLM judges compare pairs of forecasts, and their picks are compared with MSE and MAE (`judge/`).

The final report is in `report/`.

## Layout
| Path | Contents |
|---|---|
| `setup.sh` | clones the authors' code and applies the one source fix |
| `replication/` | run commands, 5-run script and log, paper-vs-ours results |
| `beijing/` | download, preprocessing and baseline scripts, results |
| `synthetic/` | generator, runner, level-shift analysis, the nine generated CSVs, results and figures |
| `judge/` | pair builder, judge prompt, both judges' verdicts, answer key, scoring scripts, results |
| `report/` | the final report (LaTeX source, bibliography, figures, PDF) |

## Environment
MacBook Air (Apple Silicon), CPU only. Python 3.9.25, PyTorch 2.8.0 (the repo's pinned PyTorch 1.9 does not
install on Apple Silicon), NumPy 2.x, pandas, matplotlib.

Two changes to the authors' setup: `np.Inf` -> `np.inf` in `utils/tools.py` (done by `setup.sh`), and
`--num_workers 0` on every run.

## Reproduce
```bash
conda activate LTSF_Linear
bash setup.sh                      # clone authors' code, apply the fix
# copy the authors' benchmark CSVs into LTSF-Linear/dataset/
bash replication/run_replication.sh           # single runs
bash replication/run_seeds.sh                 # 5 runs each, writes replication/seed_results.csv
bash replication/beijing_run.sh
python synthetic/make_synthetic.py            # regenerates synthetic/data/*.csv (seed 2026)
python synthetic/run_synthetic.py             # trains DLinear + NLinear on all nine scenarios (about 20 min on CPU)
python synthetic/level_shift_breakdown.py     # error breakdown for the level-shift scenario
```
Set `LTSF_DIR` to use an existing checkout of the authors' repo instead of `./LTSF-Linear`.

## Results

### Replication (mean and std over 5 runs, `replication/seed_results.csv`)
| Data | Paper MSE / MAE | Ours MSE (mean, std) | Ours MAE (mean, std) |
|---|---|---|---|
| ETTh1 (96) | 0.375 / 0.399 | 0.3748 (0.0055) | 0.3965 (0.0051) |
| Weather (96) | 0.176 / 0.237 | 0.1747 (0.0004) | 0.2352 (0.0015) |
| ILI (24) | 2.215 / 1.081 | 2.2515 (0.0199) | 1.0454 (0.0122) |

The first single runs gave ETTh1 0.384 / 0.405, Weather 0.178 / 0.244 and ILI 2.280 / 1.061 (`replication/replication_results.csv`).
The authors' code prints test metrics to the log but does not save them, so `summarise_seeds.py` reads `replication/seeds_log.txt`.

### Beijing Dongsi (single run, 336 in, 96 out)
DLinear MSE 0.754, MAE 0.491; predict-the-mean 1.171; repeat-last 1.325.

### Synthetic (single runs)
DLinear stays close to the noise floor in eight scenarios; on the level-shift series MSE is 0.424
against a floor of 0.028, and 12.7% of test windows (those touching the shift) carry 93.7% of the error.
Full tables are in `synthetic/results/`.

### LLM-as-judge (`judge/`)
`make_pairs.py` builds 180 anonymised forecast pairs (10 windows x 9 synthetic scenarios x 2 comparisons:
DLinear vs NLinear, DLinear vs repeat-last). The judge sees the last 96 input values, the true next 96 values and
forecasts A/B in random order, never the model names or errors. `judge_prompt.md` is the instruction given to the judges.
Two judges (Claude Sonnet, `verdicts/`; Claude Haiku, `verdicts_haiku/`) judged all 180 pairs, one fresh instance per batch of 30.
`key.csv` holds the hidden labels, `score_judge.py` compares verdicts with the MSE and MAE winner, and
`compare_judges.py` compares the two judges with each other.

- DLinear vs repeat-last (large error gap): both judges match the MSE winner in 95 to 98% of the pairs where they pick a side (Sonnet 83/85, Haiku 82/86), and agree with each other on 82 of 83.
- DLinear vs NLinear (gap under 20%): both judges say "tie" in 89 or 90 of 90 pairs.

Limits: two judges from one model family, 10 windows per scenario, synthetic data only, no human check.
