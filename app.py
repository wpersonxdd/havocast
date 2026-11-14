"""Gradio demo: tomorrow's air in Tashkent, as advice for a school PE teacher.

Tab 1 forecasts tomorrow from live data. Tab 2 replays any day of the test year (Nov 2025 - Nov 2026),
which the model never saw, and shows what it predicted next to what really happened.

Usage: python app.py
"""
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from features import THRESHOLD  # noqa: E402
from forecast_tomorrow import advice, forecast  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
MODEL_COL = "boosting (today+forecast) real forecast"
DISCLAIMER = ("Prototype for a school project. PM2.5 values are CAMS model estimates from Open-Meteo, "
              "not station measurements, and are likely lower than real levels in Tashkent.")


def load_test():
    p = pd.read_csv(OUT / "test_predictions.csv", parse_dates=["date"])
    import joblib
    return p, joblib.load(OUT / "model.joblib")["warn_at"]


TEST, WARN_AT = load_test()


def live():
    return advice(forecast())


def chart(highlight=None):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(TEST["date"], TEST["actual"], color="#334155", lw=1.4, label="what happened")
    ax.plot(TEST["date"], TEST[MODEL_COL], color="#2563eb", lw=1.2, label="our forecast (day before)")
    ax.plot(TEST["date"], TEST["persistence"], color="#94a3b8", lw=0.9, ls="--", label="baseline: tomorrow = today")
    ax.axhline(THRESHOLD, color="#dc2626", lw=1, ls=":", label=f"{THRESHOLD:.0f} µg/m³ limit")
    if highlight is not None:
        ax.axvline(highlight, color="#f59e0b", lw=2)
    ax.set_ylabel("PM2.5, µg/m³")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Test year the model never saw: Nov 2025 – Nov 2026")
    fig.tight_layout()
    return fig


def replay(date_str):
    day = pd.Timestamp(date_str)
    row = TEST[TEST["date"] == day]
    if row.empty:
        return f"No data for {date_str}. Pick a date between {TEST['date'].min().date()} and {TEST['date'].max().date()}.", chart()
    r = row.iloc[0]
    pred, actual, base = r[MODEL_COL], r["actual"], r["persistence"]
    said = "move PE indoors" if pred > WARN_AT else "outdoor PE is fine"
    truth = "a smog day" if actual > THRESHOLD else "an acceptable day"
    verdict = ("correct" if (pred > WARN_AT) == (actual > THRESHOLD)
               else ("missed smog day" if actual > THRESHOLD else "false alarm"))
    text = (f"For {day.date()}, the evening before, our model forecast {pred:.1f} µg/m³ → advice: {said}.\n"
            f"The baseline (tomorrow = today) said {base:.1f} µg/m³.\n"
            f"What really happened: {actual:.1f} µg/m³, {truth}. Our advice was: {verdict}.")
    return text, chart(day)


with gr.Blocks(title="Tashkent Air for Schools") as demo:
    gr.Markdown("# Tashkent Air for Schools\nShould tomorrow's PE class be outdoors? "
                "A next-day PM2.5 forecast for Tashkent.\n\n*" + DISCLAIMER + "*")
    with gr.Tab("Tomorrow"):
        btn = gr.Button("Forecast tomorrow", variant="primary")
        out = gr.Textbox(label="Advice for the PE teacher", lines=5)
        btn.click(live, outputs=out)
    with gr.Tab("Replay a past day"):
        date_in = gr.Textbox(label="Date (YYYY-MM-DD)", value="2025-11-30")
        go = gr.Button("Show forecast vs reality")
        res = gr.Textbox(label="Result", lines=4)
        plot = gr.Plot(value=chart())
        go.click(replay, inputs=date_in, outputs=[res, plot])

if __name__ == "__main__":
    demo.launch()
