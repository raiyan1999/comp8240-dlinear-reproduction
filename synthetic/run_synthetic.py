"""Run DLinear / NLinear on every synthetic scenario and collect MSE/MAE.

Also computes the 'noise floor': the MSE of a perfect forecaster that knows the noise-free
signal (standardised with train-split statistics, as LTSF-Linear does), so results can be
read against the best achievable error.
"""
import os, re, subprocess, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from make_synthetic import SCENARIOS, generate

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.environ.get("LTSF_DIR", os.path.join(REPO, "LTSF-Linear"))  # the cloned authors' repo
DATA = os.path.join(REPO, "synthetic", "data")
MODELS = ["DLinear", "NLinear"]
SEQ_LEN, PRED_LEN = 336, 96


def noise_floor(override):
    df, clean, p = generate(override)
    y = df["OT"].values
    n = len(y)
    n_train = int(n * 0.7)
    mu, sd = y[:n_train].mean(), y[:n_train].std()
    test_start = n - int(n * 0.2)
    yt = (y[test_start:] - mu) / sd
    ct = (clean[test_start:] - mu) / sd
    return float(np.mean((yt - ct) ** 2)), float(np.mean(np.abs(yt - ct)))


def run(model, name):
    csv = f"syn_{name}.csv"
    cmd = [sys.executable, "run_longExp.py", "--is_training", "1", "--root_path", DATA + "/",
           "--data_path", csv, "--model_id", f"syn_{name}", "--model", model, "--data", "custom",
           "--features", "S", "--target", "OT", "--seq_len", str(SEQ_LEN), "--pred_len", str(PRED_LEN),
           "--enc_in", "1", "--des", "Exp", "--itr", "1", "--batch_size", "32",
           "--learning_rate", "0.005", "--num_workers", "0", "--use_gpu", "False"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
    m = re.findall(r"mse:([\d.eE+-]+), mae:([\d.eE+-]+)", out)
    if not m:
        print(out[-800:])
        raise RuntimeError(f"no result for {model} {name}")
    return float(m[-1][0]), float(m[-1][1])


rows = []
for name, override in SCENARIOS.items():
    floor_mse, floor_mae = noise_floor(override)
    for model in MODELS:
        mse, mae = run(model, name)
        rows.append(dict(scenario=name, model=model, mse=mse, mae=mae,
                         floor_mse=floor_mse, floor_mae=floor_mae))
        print(f"{name:18s} {model:8s} MSE={mse:.4f} MAE={mae:.4f} (floor MSE={floor_mse:.4f})", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(REPO, "synthetic", "results", "synthetic_results.csv"), index=False)
print(res.pivot(index="scenario", columns="model", values="mse").round(4))
