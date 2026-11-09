# Reproducing and extending DLinear (COMP8240)

Ishraq Ahmed (48937851), Macquarie University, COMP8240 Applications of Data Science, Session 2 2026.

This repository reproduces **DLinear** from *Are Transformers Effective for Time Series Forecasting?*
(Zeng et al., AAAI 2023, [arXiv 2205.13504](https://arxiv.org/abs/2205.13504), code at
[cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)) and tests it on new data:

1. **Replication** of the authors' results on ETTh1, Weather and ILI (`replication/`).
2. **Existing dataset not in the paper:** Beijing Multi-Site Air Quality, Dongsi station (`beijing/`).
3. **Constructed dataset:** nine controlled synthetic series (trend + daily + weekly cycles + noise, with an
   optional level shift), each with a known noise-free signal (`synthetic/`).

## Layout
| Folder | Contents |
|---|---|
| `setup.sh` | clones the authors' code and applies the one source fix |
| `replication/` | run commands and the paper-vs-ours results table |
| `beijing/` | download, preprocessing and baseline scripts, results |
| `synthetic/` | generator, runner, level-shift analysis, the nine generated CSVs, results and figures |
| `report/` | the final report |

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
bash replication/run_replication.sh
bash replication/beijing_run.sh
python synthetic/make_synthetic.py            # regenerates synthetic/data/*.csv (seed 2026)
python synthetic/run_synthetic.py             # trains DLinear + NLinear on all nine scenarios (about 20 min on CPU)
python synthetic/level_shift_breakdown.py     # error breakdown for the level-shift scenario
```
Set `LTSF_DIR` to use an existing checkout of the authors' repo instead of `./LTSF-Linear`.

## Results so far (single runs)
| Dataset | Horizon | Paper MSE / MAE | Ours MSE / MAE | MSE diff |
|---|---|---|---|---|
| ETTh1 | 96 | 0.375 / 0.399 | 0.384 / 0.405 | +2.4% |
| Weather | 96 | 0.176 / 0.237 | 0.178 / 0.244 | +1.1% |
| ILI | 24 | 2.215 / 1.081 | 2.280 / 1.061 | +2.9% |

Beijing Dongsi (336 in, 96 out): DLinear MSE 0.754, MAE 0.491; predict-the-mean 1.171; repeat-last 1.325.

Synthetic: DLinear stays close to the noise floor in eight scenarios; on the level-shift series MSE is 0.424
against a floor of 0.028, and 12.7% of test windows (those touching the shift) carry 93.7% of the error.
Full tables are in `synthetic/results/`.
