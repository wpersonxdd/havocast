"""Download hourly PM2.5 and weather for Tashkent from Open-Meteo and save them as CSV.

PM2.5: Open-Meteo Air Quality API (CAMS global model estimates, not station measurements).
Weather: Open-Meteo Historical Weather API (ERA5 reanalysis).
Day-ahead weather forecasts: Open-Meteo Previous Runs API (what forecasts said one day before).
All are free and need no API key.

Usage: python src/fetch_data.py
"""
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

LAT, LON = 41.3111, 69.2797          # central Tashkent
TZ = "Asia/Tashkent"
START = "2022-10-01"                 # first full month of CAMS data for this point
DATA = Path(__file__).resolve().parent.parent / "data"

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_START = "2024-01-01"        # archived day-ahead forecasts exist from 2024
FORECAST_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "boundary_layer_height",
]


def get_hourly(url: str, params: dict) -> pd.DataFrame:
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def main():
    DATA.mkdir(exist_ok=True)
    end = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    air = get_hourly(AIR_URL, {
        "latitude": LAT, "longitude": LON, "timezone": TZ,
        "hourly": "pm2_5,pm10", "start_date": START, "end_date": end,
    })
    air.to_csv(DATA / "pm25_hourly.csv", index=False)
    print(f"PM2.5: {len(air)} hours, {air['pm2_5'].notna().sum()} with values, {air['time'].min()} .. {air['time'].max()}")

    weather = get_hourly(WEATHER_URL, {
        "latitude": LAT, "longitude": LON, "timezone": TZ,
        "hourly": ",".join(WEATHER_VARS), "start_date": START, "end_date": end,
    })
    weather.to_csv(DATA / "weather_hourly.csv", index=False)
    filled = weather.dropna(subset=["temperature_2m"])
    print(f"Weather: {len(weather)} hours, last value {filled['time'].max()}")
    print("Missing share per weather column:")
    print(weather[WEATHER_VARS].isna().mean().round(3).to_string())

    # Real day-ahead forecasts: the value each model run predicted ~24 hours before the hour.
    fc_vars = [f"{v}_previous_day1" for v in FORECAST_VARS]
    forecast = get_hourly(PREVIOUS_RUNS_URL, {
        "latitude": LAT, "longitude": LON, "timezone": TZ,
        "hourly": ",".join(fc_vars), "start_date": FORECAST_START, "end_date": end,
    })
    forecast = forecast.rename(columns={f"{v}_previous_day1": v for v in FORECAST_VARS})
    forecast.to_csv(DATA / "forecast_dayahead_hourly.csv", index=False)
    print(f"Day-ahead forecasts: {forecast['temperature_2m'].notna().sum()} hours with values")


if __name__ == "__main__":
    main()
