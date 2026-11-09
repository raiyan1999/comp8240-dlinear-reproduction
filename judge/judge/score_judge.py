"""Compare the LLM judge's verdicts with the MSE / MAE ranking.
Usage: python judge/score_judge.py [key.csv]   (verdicts read from judge/verdicts/*.jsonl)
"""
import os, sys, glob, json
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
key = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "key.csv"))
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(HERE, "verdicts", "*.jsonl")))
        for l in open(f) if l.strip()]
v = pd.DataFrame(rows)[["id", "verdict"]].drop_duplicates("id")
d = key.merge(v, on="id")
d["comparison"] = d.id.str.split("-").str[-1]
d["mse_win"] = (d.mseA <= d.mseB).map({True: "A", False: "B"})
d["mae_win"] = (d.maeA <= d.maeB).map({True: "A", False: "B"})
d["gap"] = (d.mseA - d.mseB).abs() / d[["mseA", "mseB"]].min(axis=1)
d["tie"] = d.verdict == "tie"
d["agree_mse"] = d.verdict == d.mse_win
d["agree_mae"] = d.verdict == d.mae_win
d["mse_mae_agree"] = d.mse_win == d.mae_win
d["gap_bin"] = pd.cut(d.gap, [0, .05, .2, 1, 1e9], labels=["<5%", "5-20%", "20-100%", ">100%"])
def summ(g):
    return pd.Series(dict(n=len(g), ties=g.tie.sum(),
        agree_MSE_nonties=g[~g.tie].agree_mse.mean() if (~g.tie).any() else float('nan'),
        agree_MSE=g.agree_mse.mean(), agree_MAE=g.agree_mae.mean(),
        MSE_vs_MAE=g.mse_mae_agree.mean()))
out = os.path.join(HERE, "results"); os.makedirs(out, exist_ok=True)
d.to_csv(os.path.join(out, "judge_merged.csv"), index=False)
for name, g in [("comparison", d.groupby("comparison")), ("gap_bin", d[d.comparison == "NLinear"].groupby("gap_bin", observed=True)),
                ("scenario", d.groupby("scenario"))]:
    t = g.apply(summ).round(3); t.to_csv(os.path.join(out, f"judge_by_{name}.csv")); print(name); print(t, "\n")
print("overall"); print(summ(d).round(3))
