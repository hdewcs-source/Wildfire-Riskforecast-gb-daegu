"""Recompute FFDRI_new and evaluate simple baselines for the 7-day forecast task.

What this script does
---------------------
1. Rebuilds FFDRI_new from the published Ridge formula and checks it against the
   summary statistics in the report (mean, std, quartiles) and the risk-grade cut points.
2. Reproduces the forecasting protocol of src/total_code.py: 14-day lookback, 7-day horizon,
   one model per season value in the `season` column, train < 2023, validation 2023, test 2024.
3. Evaluates two baselines on the 2024 test sequences:
   - persistence: every horizon repeats the last observed FFDRI_new
   - Ridge: a linear model on the 14 past values of FFDRI_new

Run
---
python analysis/verify_and_baselines.py --data data/lstm_dataset.csv --out results
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

LOOKBACK, HORIZON = 14, 7

# Ridge coefficients reported in the final report (Step 4 of the FFDRI_new design)
COEF = {"const": 5.3012, "DWI": 6.6232, "FMI": 0.7657, "TMI": 1.0994,
        "sunlight_era5": 0.0906, "NDVI": -10.6796}


def add_ffdri_new(df):
    out = df.copy()
    out["FFDRI_new"] = COEF["const"] + sum(COEF[k] * out[k] for k in COEF if k != "const")
    return out


def make_sequences(sub, feature="FFDRI_new"):
    """Sliding windows per coordinate, in the same way as the original training code.

    Rows are taken as consecutive within the given subset. No check is made that the
    dates are adjacent, which is also the behaviour of the original code.
    """
    xs, ys, last = [], [], []
    for _, g in sub.groupby("pid"):
        g = g.sort_values("date").reset_index(drop=True)
        f, t = g[feature].values, g["FFDRI_new"].values
        for i in range(len(g) - LOOKBACK - HORIZON + 1):
            xs.append(f[i:i + LOOKBACK])
            ys.append(t[i + LOOKBACK:i + LOOKBACK + HORIZON])
            last.append(t[i + LOOKBACK - 1])
    return np.array(xs), np.array(ys), np.array(last)


def scores(y, p):
    return {
        "rmse_t1": float(np.sqrt(np.mean((y[:, 0] - p[:, 0]) ** 2))),
        "mae_t1": float(np.mean(np.abs(y[:, 0] - p[:, 0]))),
        "corr_t1": float(np.corrcoef(y[:, 0], p[:, 0])[0, 1]),
        "rmse_1_7": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae_1_7": float(np.mean(np.abs(y - p))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/lstm_dataset.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["date"])
    df = add_ffdri_new(df)
    lines = []

    stats = df["FFDRI_new"].describe(percentiles=[0.25, 0.5, 0.75]).round(3)
    q90 = float(df["FFDRI_new"].quantile(0.9))
    lines += [
        "## FFDRI_new recomputed from the published coefficients",
        "",
        "| Statistic | Recomputed | Report |",
        "| --- | --- | --- |",
        f"| count | {int(stats['count'])} | 32880 |",
        f"| mean | {stats['mean']:.3f} | 14.970 |",
        f"| std | {stats['std']:.3f} | 5.229 |",
        f"| min | {stats['min']:.3f} | 1.807 |",
        f"| 25 percent | {stats['25%']:.3f} | 11.257 |",
        f"| median | {stats['50%']:.3f} | 14.916 |",
        f"| 75 percent | {stats['75%']:.3f} | 18.751 |",
        f"| max | {stats['max']:.3f} | 30.128 |",
        "",
        f"Pearson correlation between FFDRI_new and the national FFDRI over all rows: "
        f"{df['FFDRI_new'].corr(df['FFDRI']):.3f}. 90th percentile (start of the Extreme grade): {q90:.2f}.",
        "",
        "## Season definition in the dataset",
        "",
    ]
    for s in sorted(df["season"].unique()):
        months = sorted(df.loc[df["season"] == s, "date"].dt.month.unique())
        lines.append(f"- `season == {s}` contains months {months[0]} to {months[-1]}")
    lines.append("")

    lines += [
        "## 7-day forecast baselines, test year 2024",
        "",
        "Protocol: 14-day lookback, 7-day horizon, train before 2023, test 2024, one evaluation per "
        "season value of the dataset. RMSE and MAE are in FFDRI_new units.",
        "",
        "| Season | Method | t+1 RMSE | t+1 MAE | t+1 corr | 1-7 day RMSE | 1-7 day MAE | test sequences |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for season in sorted(df["season"].unique()):
        sub = df[df["season"] == season].sort_values(["date", "pid"])
        train = sub[sub["date"] < "2023-01-01"]
        test = sub[sub["date"] >= "2024-01-01"]
        x_tr, y_tr, _ = make_sequences(train)
        x_te, y_te, last = make_sequences(test)
        persist = np.repeat(last[:, None], HORIZON, axis=1)
        ridge = Ridge(alpha=1.0).fit(x_tr, y_tr).predict(x_te)
        for name, pred in [("Persistence", persist), ("Ridge on 14 lags", ridge)]:
            s = scores(y_te, pred)
            lines.append(
                f"| {season} | {name} | {s['rmse_t1']:.3f} | {s['mae_t1']:.3f} | {s['corr_t1']:.3f} | "
                f"{s['rmse_1_7']:.3f} | {s['mae_1_7']:.3f} | {len(y_te)} |"
            )
    text = "\n".join(lines) + "\n"
    (out_dir / "baseline_results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
