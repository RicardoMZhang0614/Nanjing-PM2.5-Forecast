from __future__ import annotations

from datetime import date

from forecast_engine import predict_pm25


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
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
