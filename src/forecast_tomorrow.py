"""Forecast tomorrow's PM2.5 for Tashkent from live data and turn it into advice for a school.

Fetches the last 8 days of PM2.5 (Open-Meteo / CAMS), recent weather and tomorrow's weather forecast
(Open-Meteo Forecast API), builds the same features as in training and applies outputs/model.joblib.

Usage: python src/forecast_tomorrow.py
"""
from pathlib import Path

import joblib
import pandas as pd
import requests

from features import THRESHOLD, daily_table
from fetch_data import AIR_URL, LAT, LON, TZ

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MODEL = Path(__file__).resolve().parent.parent / "outputs" / "model.joblib"
WEATHER = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
           "wind_speed_10m", "wind_direction_10m", "boundary_layer_height"]


def _hourly(url, params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def live_hourly() -> pd.DataFrame:
    common = {"latitude": LAT, "longitude": LON, "timezone": TZ, "past_days": 8, "forecast_days": 2}
    air = _hourly(AIR_URL, {**common, "hourly": "pm2_5,pm10"})
    weather = _hourly(FORECAST_URL, {**common, "hourly": ",".join(WEATHER)})
    return air.join(weather, how="inner")


def forecast(today=None) -> dict:
    bundle = joblib.load(MODEL)
    days = daily_table(live_hourly())
    today = pd.Timestamp(today).normalize() if today else pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    row = days.loc[[today], bundle["features"]]
    pred = float(bundle["model"].predict(row)[0])
    warn_at = bundle["warn_at"]
    r = days.loc[today]
    return {
        "today": today.date().isoformat(),
        "tomorrow": (today + pd.Timedelta(days=1)).date().isoformat(),
        "predicted_pm25": round(pred, 1),
        "warn": pred > warn_at,
        "warn_at": warn_at,
        "threshold": THRESHOLD,
        "today_pm25": round(float(r["pm_today"]), 1),
        "week_pm25": round(float(r["pm_7day"]), 1),
        "tomorrow_wind": round(float(r["forecast_wind_mean"]), 1),
        "tomorrow_rain": round(float(r["forecast_rain"]), 1),
        "tomorrow_temp": round(float(r["forecast_temp_mean"]), 1),
        "heating_season": bool(r["heating_season"]),
    }


def advice(f: dict) -> str:
    if f["warn"]:
        head = (f"Move outdoor PE indoors tomorrow ({f['tomorrow']}). "
                f"Expected PM2.5: {f['predicted_pm25']} ug/m3, close to or above the {f['threshold']:.0f} ug/m3 "
                f"daily limit.")
    else:
        head = (f"Outdoor PE looks fine tomorrow ({f['tomorrow']}). "
                f"Expected PM2.5: {f['predicted_pm25']} ug/m3, below the {f['threshold']:.0f} ug/m3 daily limit.")
    why = [f"today's average was {f['today_pm25']} ug/m3 (7-day average {f['week_pm25']})",
           f"tomorrow's forecast wind is {f['tomorrow_wind']} km/h"
           + (" (calm air lets smog build up)" if f["tomorrow_wind"] < 6 else ""),
           f"forecast rain {f['tomorrow_rain']} mm" + (" (rain washes particles out)" if f["tomorrow_rain"] >= 1 else "")]
    if f["heating_season"]:
        why.append("it is heating season, when Tashkent's air is usually worst")
    return head + "\nWhy: " + "; ".join(why) + "."


if __name__ == "__main__":
    print(advice(forecast()))
