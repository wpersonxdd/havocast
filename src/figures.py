"""Make the charts used in the README from outputs/test_predictions.csv.

Usage: python src/figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import CITY
from features import THRESHOLD

OUT = Path(__file__).resolve().parent.parent / "outputs"
MODEL = "boosting (today+forecast) real forecast"


def main():
    p = pd.read_csv(OUT / "test_predictions.csv", parse_dates=["date"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(p["date"], p["actual"], color="#334155", lw=1.4, label="what happened (CAMS daily mean)")
    ax.plot(p["date"], p[MODEL], color="#2563eb", lw=1.2, label="our forecast, made the day before")
    ax.plot(p["date"], p["persistence"], color="#94a3b8", lw=0.9, ls="--", label="baseline: tomorrow = today")
    ax.axhline(THRESHOLD, color="#dc2626", lw=1, ls=":", label=f"{THRESHOLD:.0f} µg/m³ daily limit")
    ax.set_ylabel("PM2.5, µg/m³")
    ax.set_title(f"{CITY}, test year the model never saw (Nov 2025 – Nov 2026)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "test_year.png", dpi=130)

    p["month"] = p["date"].dt.month
    g = p.groupby("month").apply(lambda s: pd.Series({
        "our model": (s[MODEL] - s["actual"]).abs().mean(),
        "baseline": (s["persistence"] - s["actual"]).abs().mean(),
    }), include_groups=False)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    g.plot.bar(ax=ax, color=["#2563eb", "#94a3b8"], width=0.75)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                       rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("mean absolute error, µg/m³")
    ax.set_title("Error by month (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "error_by_month.png", dpi=130)
    print("saved test_year.png and error_by_month.png")


if __name__ == "__main__":
    main()
