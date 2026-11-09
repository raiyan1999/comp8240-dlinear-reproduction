"""Build anonymised forecast pairs for the LLM-as-judge experiment.

For each synthetic scenario we pick N_WIN forecast windows (fixed seed) and make
two comparisons: DLinear vs NLinear, and DLinear vs repeat-last-value.
The judge sees only: input history, true future, and two forecasts labelled A/B
(random order). Model names and error values are kept in key.csv, never shown.

Usage: python judge/make_pairs.py --preds <folder with results/syn_*/pred.npy>
"""
import os, glob, json, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SEQ, PRED, N_WIN, SEED = 336, 96, 10, 7
SCEN = ["baseline","noise_low","noise_high","noise_extreme","no_seasonality",
        "weak_seasonality","strong_trend","no_trend","level_shift"]

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
a = ap.parse_args()
rng = np.random.default_rng(SEED)
r2 = lambda v: [round(float(x), 2) for x in v]

pairs, key = [], []
for s in SCEN:
    y = pd.read_csv(os.path.join(REPO, "synthetic/data", f"syn_{s}.csv"))["OT"].values.astype(float)
    N = len(y); ntr = int(N*0.7)
    z = (y - y[:ntr].mean())/y[:ntr].std()
    b1 = N - int(N*0.2) - SEQ
    P = {}
    for m in ["DLinear", "NLinear"]:
        f = glob.glob(os.path.join(a.preds, "results", f"syn_{s}_{m}_*", "pred.npy"))[0]
        P[m] = np.load(f).reshape(-1, PRED)
    n = len(P["DLinear"])
    idx = np.sort(rng.choice(n, N_WIN, replace=False))
    for k in idx:
        hist = z[b1+k : b1+k+SEQ]; truth = z[b1+k+SEQ : b1+k+SEQ+PRED]
        P["Last"] = np.full((n, PRED), 0.0); P["Last"][k] = hist[-1]
        for other in ["NLinear", "Last"]:
            fa, fb = ("DLinear", other) if rng.random() < .5 else (other, "DLinear")
            pid = f"{s}-{k}-{other}"
            pairs.append(dict(id=pid, history_last96=r2(hist[-96:]), truth=r2(truth),
                              A=r2(P[fa][k]), B=r2(P[fb][k])))
            e = lambda m: (float(((P[m][k]-truth)**2).mean()), float(np.abs(P[m][k]-truth).mean()))
            key.append(dict(id=pid, scenario=s, window=int(k), A=fa, B=fb,
                            mseA=e(fa)[0], mseB=e(fb)[0], maeA=e(fa)[1], maeB=e(fb)[1]))

with open(os.path.join(HERE, "pairs", "pairs.jsonl"), "w") as f:
    for p in pairs: f.write(json.dumps(p) + "\n")
pd.DataFrame(key).to_csv(os.path.join(HERE, "key.csv"), index=False)
print(len(pairs), "pairs written")
