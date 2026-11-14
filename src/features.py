"""Turn hourly PM2.5 and weather into one row per day, with the features known by the evening of that day.

Target: the next day's mean PM2.5 (µg/m³). Everything in a row uses only data up to 23:00 of that day,
except the optional "forecast_*" columns, which hold the next day's weather. In real use those come from a
weather forecast; here we use the observed ERA5 weather as a stand-in, which makes them slightly too good.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
THRESHOLD = 35.0  # µg/m³, US EPA 24-hour PM2.5 standard
# Next-day weather a forecast can provide. Mixing height is left out: archived forecasts don't have it.
FORECAST_SOURCE = ["temp_mean", "rain", "wind_mean", "humidity"]


def load_hourly() -> pd.DataFrame:
    air = pd.read_csv(DATA / "pm25_hourly.csv", parse_dates=["time"]).set_index("time")
    weather = pd.read_csv(DATA / "weather_hourly.csv", parse_dates=["time"]).set_index("time")
    return air.join(weather, how="inner")


def daily_table(h: pd.DataFrame) -> pd.DataFrame:
    rad = np.deg2rad(h["wind_direction_10m"])
    h = h.assign(wind_u=-h["wind_speed_10m"] * np.sin(rad), wind_v=-h["wind_speed_10m"] * np.cos(rad))
    evening = h[h.index.hour >= 18]

    d = pd.DataFrame({
        "pm_today": h["pm2_5"].resample("D").mean(),
        "pm_max_today": h["pm2_5"].resample("D").max(),
        "pm_evening": evening["pm2_5"].resample("D").mean(),
        "temp_mean": h["temperature_2m"].resample("D").mean(),
        "temp_min": h["temperature_2m"].resample("D").min(),
        "humidity": h["relative_humidity_2m"].resample("D").mean(),
        "rain": h["precipitation"].resample("D").sum(),
        "pressure": h["surface_pressure"].resample("D").mean(),
        "wind_mean": h["wind_speed_10m"].resample("D").mean(),
        "wind_max": h["wind_speed_10m"].resample("D").max(),
        "wind_u": h["wind_u"].resample("D").mean(),
        "wind_v": h["wind_v"].resample("D").mean(),
        "mixing_height": h["boundary_layer_height"].resample("D").mean(),
    })
    d["pm_yesterday"] = d["pm_today"].shift(1)
    d["pm_3day"] = d["pm_today"].rolling(3).mean()
    d["pm_7day"] = d["pm_today"].rolling(7).mean()
    d["pressure_change"] = d["pressure"].diff()

    tomorrow = d.index + pd.Timedelta(days=1)
    doy = tomorrow.dayofyear
    d["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["heating_season"] = tomorrow.month.isin([11, 12, 1, 2, 3]).astype(int)
    d["tomorrow_weekend"] = (tomorrow.dayofweek >= 5).astype(int)

    for col in FORECAST_SOURCE:
        d[f"forecast_{col}"] = d[col].shift(-1)

    d["target"] = d["pm_today"].shift(-1)
    return d


TODAY_FEATURES = [
    "pm_today", "pm_max_today", "pm_evening", "pm_yesterday", "pm_3day", "pm_7day",
    "temp_mean", "temp_min", "humidity", "rain", "pressure", "pressure_change",
    "wind_mean", "wind_max", "wind_u", "wind_v", "mixing_height",
    "season_sin", "season_cos", "heating_season", "tomorrow_weekend",
]
FORECAST_FEATURES = [f"forecast_{c}" for c in FORECAST_SOURCE]


def build() -> pd.DataFrame:
    return daily_table(load_hourly())


def real_forecast_features() -> pd.DataFrame:
    """Tomorrow's weather as it was actually forecast one day ahead, indexed by 'today'.

    Same column names as FORECAST_FEATURES, so a model trained on observed weather can be
    tested on what a forecast would really have told the school the evening before.
    """
    f = pd.read_csv(DATA / "forecast_dayahead_hourly.csv", parse_dates=["time"]).set_index("time")
    daily = pd.DataFrame({
        "forecast_temp_mean": f["temperature_2m"].resample("D").mean(),
        "forecast_rain": f["precipitation"].resample("D").sum(min_count=1),
        "forecast_wind_mean": f["wind_speed_10m"].resample("D").mean(),
        "forecast_humidity": f["relative_humidity_2m"].resample("D").mean(),
    })
    daily.index = daily.index - pd.Timedelta(days=1)  # forecast for day d+1 belongs to row d
    return daily
