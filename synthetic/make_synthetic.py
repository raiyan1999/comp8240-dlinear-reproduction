"""Controlled synthetic time series for probing DLinear's trend/seasonal decomposition.

Each series = trend + daily seasonality + weekly seasonality + Gaussian noise (+ optional level shift).
Output CSVs use the LTSF-Linear "custom" format: a `date` column then value column(s), target last (`OT`).
Hourly frequency, 17,420 steps (same length as ETTh1) so results are comparable in scale.

Usage:  python synthetic/make_synthetic.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 17420          # same length as ETTh1
SEED = 2026
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# Default generating parameters (one knob changed per scenario)
BASE = dict(
    trend_slope=0.0005,    # units per hour (linear trend)
    daily_amp=3.0,         # amplitude of 24h cycle
    weekly_amp=1.5,        # amplitude of 168h cycle
    noise_std=0.5,         # std of Gaussian noise
    shift_size=0.0,        # abrupt level shift
    shift_at=0.85,         # fraction of series where shift happens (0.85 -> inside the test split)
)

SCENARIOS = {
    "baseline":        dict(),
    "noise_low":       dict(noise_std=0.25),
    "noise_high":      dict(noise_std=2.0),
    "noise_extreme":   dict(noise_std=4.0),
    "no_seasonality":  dict(daily_amp=0.0, weekly_amp=0.0),
    "weak_seasonality": dict(daily_amp=0.5, weekly_amp=0.25),
    "strong_trend":    dict(trend_slope=0.003),
    "no_trend":        dict(trend_slope=0.0),
    "level_shift":     dict(shift_size=8.0),
}


def generate(params, n=N, seed=SEED):
    p = {**BASE, **params}
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = p["trend_slope"] * t
    daily = p["daily_amp"] * np.sin(2 * np.pi * t / 24)
    weekly = p["weekly_amp"] * np.sin(2 * np.pi * t / 168 + 1.0)
    noise = rng.normal(0, p["noise_std"], n)
    shift = np.where(t >= int(p["shift_at"] * n), p["shift_size"], 0.0)
    y = 10 + trend + daily + weekly + noise + shift
    clean = 10 + trend + daily + weekly + shift  # noise-free signal
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({"date": dates, "OT": y})
    return df, clean, p


def main():
    rows = []
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(11, 2.0 * len(SCENARIOS)), sharex=True)
    for ax, (name, override) in zip(axes, SCENARIOS.items()):
        df, clean, p = generate(override)
        df.to_csv(os.path.join(OUT, f"syn_{name}.csv"), index=False)
        ax.plot(df["OT"].values[:1500], lw=0.6, label="observed")
        ax.plot(clean[:1500], lw=1.0, label="clean signal")
        ax.set_ylabel(name, rotation=0, ha="right", fontsize=8)
        rows.append(dict(scenario=name, **p, n=len(df)))
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("hour (first 1,500 steps shown)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scenarios_overview.png"), dpi=130)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "scenario_parameters.csv"), index=False)
    print("wrote", len(SCENARIOS), "datasets to", OUT)


if __name__ == "__main__":
    main()
