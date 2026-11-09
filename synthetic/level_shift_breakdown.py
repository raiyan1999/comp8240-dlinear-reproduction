"""Where does the level-shift error come from?  Groups the test windows of the level_shift
scenario by where the shift falls relative to each window's input and forecast, and reports each
group's share of the total DLinear error.  Uses the saved pred.npy and rebuilds the truth from the CSV
(the authors' code does not save true.npy)."""
import os
import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LTSF = os.environ.get("LTSF_DIR", os.path.join(REPO, "LTSF-Linear"))
d = os.path.join(LTSF, "results",
    "syn_level_shift_DLinear_custom_ftS_sl336_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0")
p = np.load(os.path.join(d, "pred.npy")).reshape(-1, 96)
y = pd.read_csv(os.path.join(REPO, "synthetic", "data", "syn_level_shift.csv"))["OT"].values.astype(float)
N, SEQ, PRED = len(y), 336, 96
ntr = int(N * 0.7)
z = (y - y[:ntr].mean()) / y[:ntr].std()          # standardise with train statistics
b1 = N - int(N * 0.2) - SEQ                         # start of the test region
i = np.arange(len(p))
t = np.stack([z[b1 + k + SEQ: b1 + k + SEQ + PRED] for k in i])
e = ((p - t) ** 2).mean(axis=1)
shift = int(0.85 * N)
in0, in1 = b1 + i, b1 + i + SEQ
f0, f1 = in1, in1 + PRED
groups = {"A: all pre-shift": f1 <= shift,
          "B: shift inside forecast horizon": (f0 <= shift) & (shift < f1),
          "C: shift inside input window": (in0 < shift) & (shift < in1),
          "D: all post-shift": in0 >= shift}
print("windows", len(e), "overall MSE", round(e.mean(), 4))
rows = [dict(cat=k, count=int(m.sum()), mean_mse=e[m].mean(), share_of_total_error_pct=e[m].sum() / e.sum() * 100)
        for k, m in groups.items()]
out = pd.DataFrame(rows); print(out.round(3).to_string(index=False))
bc = groups["B: shift inside forecast horizon"] | groups["C: shift inside input window"]
print(f"shift windows B+C: {bc.sum()} ({bc.mean()*100:.1f}% of windows) carry {e[bc].sum()/e.sum()*100:.1f}% of the error")
out.to_csv(os.path.join(REPO, "synthetic", "results", "level_shift_error_breakdown.csv"), index=False)
