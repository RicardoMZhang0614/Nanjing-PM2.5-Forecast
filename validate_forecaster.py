from __future__ import annotations

from datetime import date, timedelta

from forecast_engine import ForecastInputError, predict_pm25


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    result = predict_pm25(reference_date=date(2026, 5, 26), use_live=False)
    check(result.primary_model_name == "XGBoost", "primary model is XGBoost")
    check(result.conservative_model_name == "Forecast Random Forest", "conservative model is Forecast Random Forest")
    check(0 <= result.primary_prediction <= 300, "primary prediction is in a plausible PM2.5 range")
    check(result.expected_low <= result.primary_prediction <= result.expected_high, "prediction lies inside displayed range")
    check(len(result.feature_row.columns) == 39, "primary feature row has 39 columns")
    check(len(result.history) >= 30, "at least 30 recent PM2.5 days are available")
    check(result.actual_pm25 is not None, "historical validation returns an actual PM2.5 value")
    check(result.absolute_error is not None and result.absolute_error >= 0, "historical validation returns absolute error")
    print("VALIDATION PASSED")

    try:
        live = predict_pm25(reference_date=date.today(), use_live=True)
        check(live.target_date == date.today() + timedelta(days=1), "live forecast targets tomorrow")
        print(f"LIVE CHECK PASSED: {live.target_date}, {live.primary_prediction:.2f}, {live.category}")
    except ForecastInputError as exc:
        print(f"LIVE CHECK SKIPPED/FAILED: {exc}")


if __name__ == "__main__":
    main()
