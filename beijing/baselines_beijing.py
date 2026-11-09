"""Two naive baselines for the Beijing Dongsi forecast, scored on exactly the windows and standardisation
used by LTSF-Linear (70/10/20 chronological split, z-score with train statistics, 336 in / 96 out)."""
import os
import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LTSF = os.environ.get("LTSF_DIR", os.path.join(REPO, "LTSF-Linear"))
data = pd.read_csv(os.path.join(LTSF, "dataset", "beijing_Dongsi.csv")).drop(columns="date").values.astype(float)
n, seq, pred = len(data), 336, 96
n_train, n_test = int(n * 0.7), int(n * 0.2)
z = (data - data[:n_train].mean(0)) / data[:n_train].std(0)
start = n - n_test - seq
rep, avg = [], []
for i in range(n_test - pred + 1):
    x = z[start + i: start + i + seq]
    y = z[start + i + seq: start + i + seq + pred]
    rep.append(((y - x[-1]) ** 2).mean())
    avg.append((y ** 2).mean())
print("windows:", len(rep))
print("repeat-last MSE:", round(float(np.mean(rep)), 4))
print("predict-train-mean MSE:", round(float(np.mean(avg)), 4))
