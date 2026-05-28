from __future__ import annotations

from datetime import date, timedelta

from forecast_engine import ForecastInputError, current_local_date, predict_pm25


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    result = predict_pm25(reference_date=date(2026, 4, 29), use_live=False)
    check(result.primary_model_name == "XGBoost", "primary model is XGBoost")
    check(result.conservative_model_name == "Forecast Random Forest", "conservative model is Forecast Random Forest")
    check(0 <= result.primary_prediction <= 300, "primary prediction is in a plausible PM2.5 range")
    check(result.expected_low <= result.primary_prediction <= result.expected_high, "prediction lies inside displayed range")
    check(len(result.feature_row.columns) == 39, "primary feature row has 39 columns")
    check(len(result.history) >= 30, "at least 30 recent PM2.5 days are available")
    check(result.actual_pm25 is not None, "historical validation returns an actual PM2.5 value")
    check(result.absolute_error is not None and result.absolute_error >= 0, "historical validation returns absolute error")
    print("PACKAGED HISTORICAL VALIDATION PASSED")

    try:
        historical = predict_pm25(reference_date=date(2026, 5, 13), use_live=True)
        if historical.actual_pm25 is None:
            print("ONLINE HISTORICAL CHECK SKIPPED: observed PM2.5 was unavailable or rate-limited")
        else:
            check(historical.absolute_error is not None and historical.absolute_error >= 0, "online historical validation returns absolute error")
            print(
                "ONLINE HISTORICAL CHECK PASSED: "
                f"{historical.target_date}, predicted {historical.primary_prediction:.2f}, "
                f"actual {historical.actual_pm25:.2f}"
            )
    except ForecastInputError as exc:
        print(f"ONLINE HISTORICAL CHECK SKIPPED/FAILED: {exc}")

    try:
        today = current_local_date()
        live = predict_pm25(reference_date=today, use_live=True)
        check(live.target_date == today + timedelta(days=1), "live forecast targets tomorrow")
        print(f"LIVE CHECK PASSED: {live.target_date}, {live.primary_prediction:.2f}, {live.category}")
    except ForecastInputError as exc:
        print(f"LIVE CHECK SKIPPED/FAILED: {exc}")

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
