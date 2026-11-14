"""The city the project forecasts for. Change CITY to switch, then re-run fetch_data.py, train.py and figures.py.

Each city needs its own data and its own trained model: a model trained on Tashkent is not valid elsewhere.
To add a city, add an entry with its coordinates, time zone and heating-season months.
"""

CITIES = {
    "Tashkent":  {"lat": 41.3111, "lon": 69.2797, "tz": "Asia/Tashkent", "heating_months": [11, 12, 1, 2, 3]},
    "Samarkand": {"lat": 39.6542, "lon": 66.9597, "tz": "Asia/Samarkand", "heating_months": [11, 12, 1, 2, 3]},
    "Andijan":   {"lat": 40.7821, "lon": 72.3442, "tz": "Asia/Tashkent", "heating_months": [11, 12, 1, 2, 3]},
    "Nukus":     {"lat": 42.4600, "lon": 59.6100, "tz": "Asia/Samarkand", "heating_months": [11, 12, 1, 2, 3]},
}

CITY = "Tashkent"

LAT = CITIES[CITY]["lat"]
LON = CITIES[CITY]["lon"]
TZ = CITIES[CITY]["tz"]
HEATING_MONTHS = CITIES[CITY]["heating_months"]
