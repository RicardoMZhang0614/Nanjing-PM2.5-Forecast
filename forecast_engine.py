from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

LATITUDE = 32.0603
LONGITUDE = 118.7969
TIMEZONE = "Asia/Shanghai"
LOCATION_NAME = "Nanjing, Jiangsu, China"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_COLUMNS = [
    "temp_mean",
    "temp_max",
    "temp_min",
    "humidity_mean",
    "dew_point_mean",
    "precipitation_sum",
    "rain_sum",
    "pressure_mean",
    "wind_speed_mean",
    "wind_speed_max",
    "wind_direction_mean",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
]

OPEN_METEO_WEATHER_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
]

CALENDAR_COLUMNS = [
    "year",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "is_public_holiday",
    "is_pre_or_post_holiday",
    "is_workday_proxy",
    "school_term_proxy",
    "holiday_travel_proxy",
    "cold_season_energy_proxy",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]

AIR_LAG_COLUMNS = [
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "pm25_lag7",
    "pm25_lag14",
    "pm25_roll7_mean",
    "pm25_roll14_mean",
    "pm25_roll30_mean",
    "pm25_roll7_std",
]

HOLIDAY_RANGES = [
    ("2022-01-01", "2022-01-03", "New Year"),
    ("2022-01-31", "2022-02-06", "Spring Festival"),
    ("2022-04-03", "2022-04-05", "Qingming Festival"),
    ("2022-04-30", "2022-05-04", "Labor Day"),
    ("2022-06-03", "2022-06-05", "Dragon Boat Festival"),
    ("2022-09-10", "2022-09-12", "Mid-Autumn Festival"),
    ("2022-10-01", "2022-10-07", "National Day Golden Week"),
    ("2023-01-01", "2023-01-02", "New Year"),
    ("2023-01-21", "2023-01-27", "Spring Festival"),
    ("2023-04-05", "2023-04-05", "Qingming Festival"),
    ("2023-04-29", "2023-05-03", "Labor Day"),
    ("2023-06-22", "2023-06-24", "Dragon Boat Festival"),
    ("2023-09-29", "2023-10-06", "Mid-Autumn and National Day"),
    ("2024-01-01", "2024-01-01", "New Year"),
    ("2024-02-10", "2024-02-17", "Spring Festival"),
    ("2024-04-04", "2024-04-06", "Qingming Festival"),
    ("2024-05-01", "2024-05-05", "Labor Day"),
    ("2024-06-08", "2024-06-10", "Dragon Boat Festival"),
    ("2024-09-15", "2024-09-17", "Mid-Autumn Festival"),
    ("2024-10-01", "2024-10-07", "National Day Golden Week"),
    ("2025-01-01", "2025-01-01", "New Year"),
    ("2025-01-28", "2025-02-04", "Spring Festival"),
    ("2025-04-04", "2025-04-06", "Qingming Festival"),
    ("2025-05-01", "2025-05-05", "Labor Day"),
    ("2025-05-31", "2025-06-02", "Dragon Boat Festival"),
    ("2025-10-01", "2025-10-08", "National Day and Mid-Autumn"),
    ("2026-01-01", "2026-01-03", "New Year"),
    ("2026-02-15", "2026-02-23", "Spring Festival"),
    ("2026-04-04", "2026-04-06", "Qingming Festival"),
    ("2026-05-01", "2026-05-05", "Labor Day"),
    ("2026-06-19", "2026-06-21", "Dragon Boat Festival"),
    ("2026-09-25", "2026-09-27", "Mid-Autumn Festival"),
    ("2026-10-01", "2026-10-07", "National Day Golden Week"),
]


@dataclass
class ForecastResult:
    target_date: date
    generated_at: datetime
    location: str
    primary_prediction: float
    conservative_prediction: float | None
    expected_low: float
    expected_high: float
    category: str
    category_color: str
    uncertainty_mae: float
    primary_model_name: str
    conservative_model_name: str | None
    latest_pm25_date: date
    latest_pm25: float
    latest_pm25_hours: int | None
    data_mode: str
    data_note: str
    weather_note: str
    feature_row: pd.DataFrame
    history: pd.DataFrame
    weather_daily: pd.DataFrame
    primary_mae: float
    conservative_mae: float | None
    is_demo: bool = False
    is_degraded: bool = False
    forecast_source: str = "Research ML model"


class ForecastInputError(RuntimeError):
    pass


def classify_pm25(value: float) -> str:
    if math.isnan(value):
        return "Unknown"
    if value <= 35:
        return "Good"
    if value <= 75:
        return "Moderate"
    if value <= 115:
        return "Unhealthy for Sensitive Groups"
    if value <= 150:
        return "Unhealthy"
    if value <= 250:
        return "Very Unhealthy"
    return "Hazardous"


def category_color(category: str) -> str:
    return {
        "Good": "#227567",
        "Moderate": "#A86817",
        "Unhealthy for Sensitive Groups": "#B85F22",
        "Unhealthy": "#9C3434",
        "Very Unhealthy": "#6E3A70",
        "Hazardous": "#432935",
    }.get(category, "#465466")


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ForecastInputError(f"Open-Meteo returned an unexpected payload for {url}.")
    return payload


def _hourly_payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def _aggregate_air_quality(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    frame["date"] = frame["time"].dt.floor("D")
    daily = (
        frame.groupby("date")
        .agg(
            pm25=("pm2_5", "mean"),
            pm25_hour_count=("pm2_5", "count"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def _aggregate_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    frame["date"] = frame["time"].dt.floor("D")
    daily = (
        frame.groupby("date")
        .agg(
            temp_mean=("temperature_2m", "mean"),
            temp_max=("temperature_2m", "max"),
            temp_min=("temperature_2m", "min"),
            humidity_mean=("relative_humidity_2m", "mean"),
            dew_point_mean=("dew_point_2m", "mean"),
            precipitation_sum=("precipitation", "sum"),
            rain_sum=("rain", "sum"),
            pressure_mean=("surface_pressure", "mean"),
            wind_speed_mean=("wind_speed_10m", "mean"),
            wind_speed_max=("wind_speed_10m", "max"),
            wind_direction_mean=("wind_direction_10m", "mean"),
            cloud_cover_mean=("cloud_cover", "mean"),
            shortwave_radiation_sum=("shortwave_radiation", "sum"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def fetch_recent_pm25(reference_date: date, lookback_days: int = 80) -> pd.DataFrame:
    start_date = reference_date - timedelta(days=lookback_days)
    relative_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "past_days": min(92, lookback_days),
        "forecast_days": 2,
        "timezone": TIMEZONE,
        "hourly": "pm2_5",
    }
    dated_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": reference_date.isoformat(),
        "timezone": TIMEZONE,
        "hourly": "pm2_5",
    }
    try:
        payload = _request_json(AIR_QUALITY_URL, relative_params)
    except Exception:
        payload = _request_json(AIR_QUALITY_URL, dated_params)
    return _aggregate_air_quality(_hourly_payload_to_frame(payload))


def split_air_quality_forecast(daily: pd.DataFrame, target_date: date) -> tuple[pd.DataFrame, float, int | None]:
    frame = daily.copy()
    frame = frame[frame["pm25"].notna()].sort_values("date").reset_index(drop=True)
    frame["date_only"] = frame["date"].dt.date
    target_rows = frame[frame["date_only"] == target_date]
    if target_rows.empty:
        available = ", ".join(str(value) for value in frame["date_only"].tail(5).tolist())
        raise ForecastInputError(
            f"Open-Meteo did not return a PM2.5 forecast for {target_date}. "
            f"Latest available dates: {available or 'none'}."
        )

    target = target_rows.iloc[-1]
    history = frame[frame["date_only"] < target_date].drop(columns=["date_only"]).reset_index(drop=True)
    if len(history) < 30:
        raise ForecastInputError("Open-Meteo returned fewer than 30 pre-target PM2.5 days.")

    hours = int(target["pm25_hour_count"]) if not pd.isna(target["pm25_hour_count"]) else None
    return history, float(target["pm25"]), hours


def fetch_weather_window(start_date: date, end_date: date) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": TIMEZONE,
        "hourly": ",".join(OPEN_METEO_WEATHER_HOURLY),
    }
    payload = _request_json(WEATHER_FORECAST_URL, params)
    return _aggregate_weather(_hourly_payload_to_frame(payload))


def load_packaged_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / "modeling_dataset.csv", parse_dates=["date"])
    if "pm25_hour_count" not in frame.columns:
        frame["pm25_hour_count"] = 24
    return frame.sort_values("date").reset_index(drop=True)


def load_models() -> tuple[object, object, list[str], list[str]]:
    primary_model = joblib.load(MODELS_DIR / "best_model.joblib")
    conservative_model = joblib.load(MODELS_DIR / "forecast_model.joblib")
    primary_features = json.loads((MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    conservative_features = json.loads((MODELS_DIR / "forecast_feature_columns.json").read_text(encoding="utf-8"))
    return primary_model, conservative_model, primary_features, conservative_features


def load_model_metrics() -> tuple[str, float, str, float]:
    metrics = pd.read_csv(DATA_DIR / "model_metrics.csv").sort_values("mae")
    forecast_metrics = pd.read_csv(DATA_DIR / "forecast_model_metrics.csv").sort_values("mae")
    primary = metrics.iloc[0]
    conservative = forecast_metrics.iloc[0]
    return str(primary["model"]), float(primary["mae"]), str(conservative["model"]), float(conservative["mae"])


def _holiday_dates() -> tuple[set[date], dict[date, str]]:
    dates: set[date] = set()
    names: dict[date, str] = {}
    for start_text, end_text, name in HOLIDAY_RANGES:
        current = pd.to_datetime(start_text).date()
        final = pd.to_datetime(end_text).date()
        while current <= final:
            dates.add(current)
            names[current] = name
            current += timedelta(days=1)
    return dates, names


def calendar_values(target_date: date) -> dict[str, float | int]:
    holiday_dates, _ = _holiday_dates()
    day_of_week = target_date.weekday()
    month = target_date.month
    day_of_year = target_date.timetuple().tm_yday
    is_public_holiday = int(target_date in holiday_dates)
    is_pre_or_post_holiday = int(
        (target_date - timedelta(days=1)) in holiday_dates or (target_date + timedelta(days=1)) in holiday_dates
    )
    is_weekend = int(day_of_week in [5, 6])
    values: dict[str, float | int] = {
        "year": target_date.year,
        "month": month,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "is_weekend": is_weekend,
        "is_public_holiday": is_public_holiday,
        "is_pre_or_post_holiday": is_pre_or_post_holiday,
        "is_workday_proxy": int(is_weekend == 0 and is_public_holiday == 0),
        "school_term_proxy": int(month in [3, 4, 5, 6, 9, 10, 11, 12]),
        "holiday_travel_proxy": int(is_public_holiday == 1 or is_pre_or_post_holiday == 1),
        "cold_season_energy_proxy": int(month in [11, 12, 1, 2, 3]),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
        "doy_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
    }
    return values


def holiday_name(target_date: date) -> str:
    _, names = _holiday_dates()
    return names.get(target_date, "No Holiday")


def choose_live_history(reference_date: date, requested_target: date) -> tuple[pd.DataFrame, date, str]:
    live = fetch_recent_pm25(reference_date)
    live = live[live["pm25"].notna()].sort_values("date").reset_index(drop=True)
    live["date_only"] = live["date"].dt.date

    today_rows = live[live["date_only"] == reference_date]
    complete_rows = live[(live["date_only"] < reference_date) & (live["pm25_hour_count"] >= 18)]

    if not today_rows.empty and int(today_rows.iloc[-1]["pm25_hour_count"]) >= 8:
        usable = pd.concat([complete_rows, today_rows.tail(1)], ignore_index=True).sort_values("date")
        note = "Live mode: today's PM2.5 is used as the latest lag because at least 8 hourly Open-Meteo values were available."
    else:
        if live.empty:
            raise ForecastInputError("Open-Meteo did not return usable recent PM2.5 values.")
        usable = live[live["date_only"] < requested_target].tail(80).copy()
        note = "Live mode: today's complete PM2.5 was not available, so the app uses the latest Open-Meteo PM2.5 values before the target date."

    if usable.empty:
        raise ForecastInputError("No usable recent PM2.5 records were returned by Open-Meteo.")

    latest_date = usable["date"].max().date()
    if latest_date < requested_target - timedelta(days=1):
        raise ForecastInputError(
            "Live PM2.5 data are too old to support a next-day forecast. "
            f"Latest usable PM2.5 date: {latest_date}; required: {requested_target - timedelta(days=1)}."
        )
    target_date = requested_target
    usable = usable.drop(columns=["date_only"]).reset_index(drop=True)
    return usable, target_date, note


def choose_packaged_demo_history() -> tuple[pd.DataFrame, date, str]:
    packaged = load_packaged_dataset()
    target_date = packaged["date"].max().date()
    history = packaged[packaged["date"].dt.date < target_date].copy()
    note = "Packaged demo mode: live APIs were not used; this is a historical one-day-ahead demonstration."
    return history, target_date, note


def lag_values(history: pd.DataFrame, target_date: date) -> dict[str, float]:
    frame = history.copy()
    frame["date_only"] = pd.to_datetime(frame["date"]).dt.date
    pm25_by_date = frame.set_index("date_only")["pm25"].to_dict()
    values: dict[str, float] = {}
    for lag in [1, 2, 3, 7, 14]:
        lag_date = target_date - timedelta(days=lag)
        if lag_date not in pm25_by_date:
            raise ForecastInputError(f"Missing PM2.5 lag data for {lag_date.isoformat()}.")
        values[f"pm25_lag{lag}"] = float(pm25_by_date[lag_date])

    prior = frame[frame["date_only"] < target_date].sort_values("date").tail(30)
    if len(prior) < 30:
        raise ForecastInputError("At least 30 prior daily PM2.5 records are needed for rolling features.")
    values["pm25_roll7_mean"] = float(prior.tail(7)["pm25"].mean())
    values["pm25_roll14_mean"] = float(prior.tail(14)["pm25"].mean())
    values["pm25_roll30_mean"] = float(prior.tail(30)["pm25"].mean())
    values["pm25_roll7_std"] = float(prior.tail(7)["pm25"].std())
    return values


def weather_for_packaged_target(target_date: date) -> pd.DataFrame:
    packaged = load_packaged_dataset()
    window = packaged[packaged["date"].dt.date.isin([target_date - timedelta(days=1), target_date])].copy()
    if len(window) < 2:
        raise ForecastInputError("Packaged dataset does not contain enough weather records for this target date.")
    return window[["date"] + WEATHER_COLUMNS].reset_index(drop=True)


def weather_climatology_for_target(target_date: date) -> pd.DataFrame:
    packaged = load_packaged_dataset()
    packaged["date_only"] = packaged["date"].dt.date
    packaged["day_of_year"] = packaged["date"].dt.dayofyear
    target_doy = target_date.timetuple().tm_yday

    def circular_distance(value: int) -> int:
        raw = abs(value - target_doy)
        return min(raw, 366 - raw)

    packaged["doy_distance"] = packaged["day_of_year"].apply(circular_distance)
    target_weather = packaged.nsmallest(45, "doy_distance")[WEATHER_COLUMNS].mean().to_dict()

    previous_date = target_date - timedelta(days=1)
    previous_doy = previous_date.timetuple().tm_yday

    def previous_distance(value: int) -> int:
        raw = abs(value - previous_doy)
        return min(raw, 366 - raw)

    packaged["previous_distance"] = packaged["day_of_year"].apply(previous_distance)
    previous_weather = packaged.nsmallest(45, "previous_distance")[WEATHER_COLUMNS].mean().to_dict()

    return pd.DataFrame(
        [
            {"date": pd.Timestamp(previous_date), **previous_weather},
            {"date": pd.Timestamp(target_date), **target_weather},
        ]
    )


def pm25_climatology_series(target_date: date, days: int = 80) -> tuple[pd.DataFrame, float]:
    packaged = load_packaged_dataset()
    packaged["day_of_year"] = packaged["date"].dt.dayofyear

    def circular_distance(value: int, target_doy: int) -> int:
        raw = abs(value - target_doy)
        return min(raw, 366 - raw)

    rows = []
    start_date = target_date - timedelta(days=days)
    current = start_date
    while current <= target_date:
        target_doy = current.timetuple().tm_yday
        distances = packaged["day_of_year"].apply(lambda value: circular_distance(value, target_doy))
        values = packaged.loc[distances <= 21, "pm25"]
        if values.empty:
            values = packaged.nsmallest(45, "day_of_year")["pm25"]
        rows.append(
            {
                "date": pd.Timestamp(current),
                "pm25": float(values.mean()),
                "pm25_hour_count": 0,
            }
        )
        current += timedelta(days=1)

    series = pd.DataFrame(rows)
    target_prediction = float(series[series["date"].dt.date == target_date].iloc[0]["pm25"])
    history = series[series["date"].dt.date < target_date].reset_index(drop=True)
    return history, target_prediction


def feature_rows(
    target_date: date,
    history: pd.DataFrame,
    weather_daily: pd.DataFrame,
    primary_features: list[str],
    conservative_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather_daily = weather_daily.copy()
    weather_daily["date_only"] = pd.to_datetime(weather_daily["date"]).dt.date
    target_weather = weather_daily[weather_daily["date_only"] == target_date]
    previous_weather = weather_daily[weather_daily["date_only"] == target_date - timedelta(days=1)]

    if target_weather.empty:
        raise ForecastInputError(f"Missing weather forecast for target date {target_date.isoformat()}.")
    if previous_weather.empty:
        raise ForecastInputError(f"Missing previous-day weather for {target_date - timedelta(days=1)}.")

    base = {}
    base.update(calendar_values(target_date))
    base.update(lag_values(history, target_date))

    primary_row = base.copy()
    primary_row.update(target_weather.iloc[0][WEATHER_COLUMNS].to_dict())
    primary_frame = pd.DataFrame([primary_row])[primary_features].astype(float)

    conservative_row = base.copy()
    previous_values = previous_weather.iloc[0][WEATHER_COLUMNS].to_dict()
    conservative_row.update({f"{key}_lag1": value for key, value in previous_values.items()})
    conservative_frame = pd.DataFrame([conservative_row])[conservative_features].astype(float)
    return primary_frame, conservative_frame


def fallback_primary_feature_row(
    target_date: date,
    history: pd.DataFrame,
    weather_daily: pd.DataFrame,
    primary_features: list[str],
) -> pd.DataFrame:
    weather_daily = weather_daily.copy()
    weather_daily["date_only"] = pd.to_datetime(weather_daily["date"]).dt.date
    target_weather = weather_daily[weather_daily["date_only"] == target_date]
    if target_weather.empty:
        weather_daily = weather_climatology_for_target(target_date)
        weather_daily["date_only"] = pd.to_datetime(weather_daily["date"]).dt.date
        target_weather = weather_daily[weather_daily["date_only"] == target_date]

    row: dict[str, float | int] = {}
    row.update(calendar_values(target_date))

    hist = history.copy()
    hist["date_only"] = pd.to_datetime(hist["date"]).dt.date
    pm25_by_date = hist.set_index("date_only")["pm25"].to_dict()
    latest_value = float(hist.sort_values("date").iloc[-1]["pm25"])
    for lag in [1, 2, 3, 7, 14]:
        row[f"pm25_lag{lag}"] = float(pm25_by_date.get(target_date - timedelta(days=lag), latest_value))

    prior = hist[hist["date_only"] < target_date].sort_values("date").tail(30)
    row["pm25_roll7_mean"] = float(prior.tail(7)["pm25"].mean())
    row["pm25_roll14_mean"] = float(prior.tail(14)["pm25"].mean())
    row["pm25_roll30_mean"] = float(prior.tail(30)["pm25"].mean())
    row["pm25_roll7_std"] = float(prior.tail(7)["pm25"].std())
    row.update(target_weather.iloc[0][WEATHER_COLUMNS].to_dict())
    return pd.DataFrame([row])[primary_features].astype(float)


def predict_pm25(reference_date: date | None = None, use_live: bool = True) -> ForecastResult:
    reference_date = reference_date or date.today()
    requested_target = reference_date + timedelta(days=1)
    primary_model, conservative_model, primary_features, conservative_features = load_models()
    primary_model_name, primary_mae, conservative_model_name, conservative_mae = load_model_metrics()

    if use_live:
        try:
            target_date = requested_target
            air_quality_daily = fetch_recent_pm25(reference_date)
            history, operational_prediction, target_pm25_hours = split_air_quality_forecast(air_quality_daily, target_date)
            data_note = (
                "Primary forecast is the daily mean of Open-Meteo Air Quality hourly PM2.5 forecast "
                f"for {target_date}. Target-day hourly values: {target_pm25_hours or 'n/a'}."
            )
            operational_model_name = "Open-Meteo PM2.5 forecast"
            operational_source = "Open-Meteo Air Quality forecast"
            is_degraded = False
            try:
                weather_daily = fetch_weather_window(target_date - timedelta(days=1), target_date)
                weather_note = "Weather inputs come from the Open-Meteo forecast endpoint for the target day."
                data_mode = "live PM2.5 forecast"
            except Exception as weather_exc:
                weather_daily = weather_climatology_for_target(target_date)
                weather_note = (
                    "Open-Meteo weather forecast was unavailable or rate-limited, so the app used "
                    "seasonal weather climatology from the packaged Nanjing training dataset. "
                    f"Weather API detail: {weather_exc}"
                )
                data_mode = "live PM2.5 with climatology weather fallback"
                is_degraded = True
            is_demo = False
        except Exception as exc:
            target_date = requested_target
            history, operational_prediction = pm25_climatology_series(target_date)
            weather_daily = weather_climatology_for_target(target_date)
            data_note = (
                "Open-Meteo live PM2.5 forecast was unavailable or rate-limited. "
                "The app used a same-season historical PM2.5 baseline from the packaged Nanjing dataset. "
                f"Live API detail: {exc}"
            )
            weather_note = "Weather features use same-season historical weather averages from the packaged Nanjing dataset."
            data_mode = "seasonal PM2.5 fallback"
            target_pm25_hours = None
            operational_model_name = "Seasonal PM2.5 fallback"
            operational_source = "Packaged seasonal PM2.5 baseline"
            is_demo = False
            is_degraded = True
    else:
        history, target_date, data_note = choose_packaged_demo_history()
        weather_daily = weather_for_packaged_target(target_date)
        weather_note = "Weather inputs come from the packaged historical dataset for validation/demo mode."
        data_mode = "historical demo"
        is_demo = True
        is_degraded = False
        target_pm25_hours = None
        operational_prediction = None
        operational_model_name = primary_model_name
        operational_source = "Research ML model"

    primary_frame = fallback_primary_feature_row(target_date, history, weather_daily, primary_features)
    try:
        _, conservative_frame = feature_rows(
            target_date=target_date,
            history=history,
            weather_daily=weather_daily,
            primary_features=primary_features,
            conservative_features=conservative_features,
        )
        conservative_prediction = float(conservative_model.predict(conservative_frame)[0])
    except ForecastInputError:
        conservative_prediction = None

    if use_live and operational_prediction is not None:
        primary_prediction = float(operational_prediction)
        display_model_name = operational_model_name
        display_mae = conservative_mae
        forecast_source = operational_source
    else:
        primary_prediction = float(primary_model.predict(primary_frame)[0])
        display_model_name = primary_model_name
        display_mae = primary_mae
        forecast_source = "Research ML model"

    uncertainty_mae = max(primary_mae, conservative_mae)
    expected_low = max(0.0, primary_prediction - uncertainty_mae)
    expected_high = primary_prediction + uncertainty_mae
    category = classify_pm25(primary_prediction)

    latest = history.sort_values("date").iloc[-1]
    history_tail = history[["date", "pm25"]].sort_values("date").tail(60).reset_index(drop=True)
    latest_hours = int(latest["pm25_hour_count"]) if "pm25_hour_count" in latest and not pd.isna(latest["pm25_hour_count"]) else None

    return ForecastResult(
        target_date=target_date,
        generated_at=datetime.now(),
        location=LOCATION_NAME,
        primary_prediction=primary_prediction,
        conservative_prediction=conservative_prediction,
        expected_low=expected_low,
        expected_high=expected_high,
        category=category,
        category_color=category_color(category),
        uncertainty_mae=uncertainty_mae,
        primary_model_name=display_model_name,
        conservative_model_name=conservative_model_name,
        latest_pm25_date=pd.to_datetime(latest["date"]).date(),
        latest_pm25=float(latest["pm25"]),
        latest_pm25_hours=target_pm25_hours if use_live else latest_hours,
        data_mode=data_mode,
        data_note=data_note,
        weather_note=weather_note,
        feature_row=primary_frame,
        history=history_tail,
        weather_daily=weather_daily,
        primary_mae=display_mae,
        conservative_mae=conservative_mae,
        is_demo=is_demo,
        is_degraded=is_degraded,
        forecast_source=forecast_source,
    )
