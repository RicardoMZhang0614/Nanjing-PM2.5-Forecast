from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecast_engine import ForecastInputError, ForecastResult, holiday_name, predict_pm25


COLOR_TEXT = "#17212F"
COLOR_MUTED = "#566274"
COLOR_PANEL = "#FFFFFF"
COLOR_BORDER = "#D9E0EA"
COLOR_BLUE = "#245A8D"
COLOR_GREEN = "#227567"
COLOR_AMBER = "#A86817"
COLOR_RED = "#9C3434"


def style_page() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #F5F7FA !important;
            color: #17212F !important;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            color: #17212F !important;
        }
        [data-testid="stSidebar"] {
            background: #FFFFFF !important;
            color: #17212F !important;
        }
        [data-testid="stSidebar"] * {
            color: #17212F !important;
        }
        h1, h2, h3 {
            color: #17212F !important;
            letter-spacing: 0;
        }
        .hero-subtitle {
            color: #566274 !important;
            font-size: 1rem;
            line-height: 1.45;
            max-width: 850px;
            margin: 0.35rem 0 1.1rem 0;
        }
        .forecast-card {
            border: 1px solid #D9E0EA;
            border-radius: 8px;
            background: #FFFFFF !important;
            padding: 22px 24px;
            box-shadow: 0 10px 22px rgba(16, 24, 40, 0.06);
            margin-bottom: 1rem;
        }
        .forecast-label {
            color: #566274 !important;
            font-size: 0.84rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .forecast-value {
            color: #17212F !important;
            font-size: 3rem;
            line-height: 1;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .forecast-unit {
            color: #566274 !important;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 10px;
            font-weight: 700;
            font-size: 0.86rem;
            color: #FFFFFF;
        }
        .small-card {
            min-height: 132px;
            border: 1px solid #D9E0EA;
            border-radius: 8px;
            background: #FFFFFF !important;
            padding: 15px 16px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
        }
        .small-label {
            color: #344054 !important;
            font-size: 0.88rem;
            font-weight: 750;
            margin-bottom: 7px;
        }
        .small-value {
            color: #17212F !important;
            font-size: 1.45rem;
            font-weight: 780;
            line-height: 1.18;
            margin-bottom: 8px;
        }
        .small-note {
            color: #566274 !important;
            font-size: 0.82rem;
            line-height: 1.38;
        }
        .section-note {
            color: #566274 !important;
            font-size: 0.94rem;
            line-height: 1.45;
            margin-bottom: 0.65rem;
        }
        .status-box {
            border: 1px solid #D9E0EA;
            border-left: 5px solid #245A8D;
            border-radius: 8px;
            background: #FFFFFF !important;
            padding: 13px 15px;
            color: #344054 !important;
            font-size: 0.9rem;
            line-height: 1.42;
            margin-bottom: 1rem;
        }
        .notice-box {
            border: 1px solid #D8A13A;
            border-left: 5px solid #A86817;
            border-radius: 8px;
            background: #FFF7E0 !important;
            color: #17212F !important;
            padding: 13px 15px;
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0.75rem 0 1rem 0;
        }
        .error-box {
            border: 1px solid #E0A0A0;
            border-left: 5px solid #9C3434;
            border-radius: 8px;
            background: #FFF1F1 !important;
            color: #17212F !important;
            padding: 13px 15px;
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0.75rem 0 1rem 0;
        }
        .notice-box strong, .error-box strong {
            color: #17212F !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #344054 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_prediction(reference_date: date, use_live: bool) -> ForecastResult:
    return predict_pm25(reference_date=reference_date, use_live=use_live)


def render_small_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="small-card">
            <div class="small-label">{label}</div>
            <div class="small-value">{value}</div>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=34, r=28, t=56, b=38),
        paper_bgcolor=COLOR_PANEL,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="Arial", size=13, color=COLOR_TEXT),
        title=dict(font=dict(size=18, color=COLOR_TEXT), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", bordercolor=COLOR_BORDER, font=dict(color=COLOR_TEXT)),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#AEB8C6",
        zeroline=False,
        tickfont=dict(color="#344054", size=12),
        title_font=dict(color="#17212F", size=13),
    )
    fig.update_yaxes(
        gridcolor="#D5DCE7",
        linecolor="#AEB8C6",
        zeroline=False,
        tickfont=dict(color="#344054", size=12),
        title_font=dict(color="#17212F", size=13),
    )
    return fig


def trend_figure(result: ForecastResult) -> go.Figure:
    history = result.history.copy()
    forecast_x = [pd.Timestamp(result.target_date)]
    forecast_y = [result.primary_prediction]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["pm25"],
            mode="lines",
            name="Recent daily PM2.5",
            line=dict(color="rgba(86,98,116,0.72)", width=1.7),
            hovertemplate="%{x|%Y-%m-%d}<br>PM2.5: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["pm25"].rolling(7, min_periods=1).mean(),
            mode="lines",
            name="7-day average",
            line=dict(color=COLOR_BLUE, width=3),
            hovertemplate="%{x|%Y-%m-%d}<br>7-day mean: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode="markers",
            name="Tomorrow forecast",
            marker=dict(size=15, color=result.category_color, line=dict(color="#FFFFFF", width=2)),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[result.expected_high - result.primary_prediction],
                arrayminus=[result.primary_prediction - result.expected_low],
                thickness=1.6,
                color=result.category_color,
            ),
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Recent PM2.5 History and Tomorrow Forecast",
        xaxis_title="",
        yaxis_title="PM2.5 (micrograms per cubic meter)",
    )
    return apply_layout(fig, 455)


def comparison_figure(result: ForecastResult) -> go.Figure:
    rows = [
        {"model": result.primary_model_name, "prediction": result.primary_prediction, "type": "Weather-forecast model"},
    ]
    if result.conservative_prediction is not None and result.conservative_model_name is not None:
        rows.append(
            {
                "model": result.conservative_model_name.replace("Forecast ", ""),
                "prediction": result.conservative_prediction,
                "type": "Conservative lag model",
            }
        )
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="prediction",
        y="type",
        orientation="h",
        text=frame["prediction"].map(lambda value: f"{value:.1f}"),
        color="type",
        color_discrete_sequence=[COLOR_BLUE, COLOR_AMBER],
        labels={"prediction": "Predicted PM2.5", "type": ""},
        title="Prediction Cross-Check",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_layout(fig, 260)


def driver_figure(result: ForecastResult) -> go.Figure:
    row = result.feature_row.iloc[0]
    items = [
        ("Previous-day PM2.5", row["pm25_lag1"], "micrograms/m3"),
        ("Prior 7-day PM2.5 mean", row["pm25_roll7_mean"], "micrograms/m3"),
        ("Forecast mean wind speed", row["wind_speed_mean"], "km/h"),
        ("Forecast precipitation", row["precipitation_sum"], "mm"),
        ("Forecast mean temperature", row["temp_mean"], "C"),
    ]
    frame = pd.DataFrame(items, columns=["factor", "value", "unit"])
    fig = px.bar(
        frame,
        x="value",
        y="factor",
        orientation="h",
        text=frame.apply(lambda item: f"{item['value']:.1f} {item['unit']}", axis=1),
        color="factor",
        color_discrete_sequence=["#245A8D", "#3D7B7A", "#A86817", "#7E5A8A", "#8B9440"],
        labels={"value": "Input value", "factor": ""},
        title="Key Inputs Used for the Forecast",
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_layout(fig, 360)


def main() -> None:
    st.set_page_config(page_title="Nanjing PM2.5 Tomorrow Forecast", layout="wide")
    style_page()

    st.title("Nanjing PM2.5 Tomorrow Forecast")
    st.markdown(
        '<div class="hero-subtitle">A live short-term PM2.5 prediction tool using recent Open-Meteo air-quality data, weather forecast inputs, and the trained Nanjing PM2.5 model.</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.header("Forecast Controls")
    reference_date = st.sidebar.date_input("Reference date", value=date.today())
    use_live = st.sidebar.checkbox("Use live Open-Meteo data", value=True)
    st.sidebar.caption("Live mode predicts the next day. Turn it off only for a packaged historical demo.")
    if st.sidebar.button("Refresh forecast"):
        cached_prediction.clear()

    try:
        with st.spinner("Building forecast inputs and running models..."):
            result = cached_prediction(reference_date=reference_date, use_live=use_live)
    except ForecastInputError as exc:
        st.markdown(
            f"""
            <div class="error-box">
            <strong>Live forecast could not be produced.</strong><br>
            {str(exc)}<br><br>
            This app now stops instead of replacing tomorrow's forecast with a historical demo date. 
            Check the Streamlit Cloud logs or temporarily turn off live mode to view the packaged validation demo.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    is_demo_result = bool(getattr(result, "is_demo", result.data_mode in {"historical demo", "packaged demo"}))
    is_degraded_result = bool(getattr(result, "is_degraded", "fallback" in result.data_mode.lower()))

    if is_demo_result:
        st.markdown(
            f"""
            <div class="notice-box">
            <strong>Historical demo mode.</strong><br>
            This is not tomorrow's live forecast. It replays the packaged historical validation date: {result.target_date}.
            Turn on live mode to forecast {reference_date + timedelta(days=1)}.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif is_degraded_result:
        st.markdown(
            f"""
            <div class="notice-box">
            <strong>Forecast produced with weather fallback.</strong><br>
            The app produced a live PM2.5 forecast for {result.target_date}, but the weather forecast API was unavailable or rate-limited. 
            Weather features were filled with same-season historical weather averages from the packaged Nanjing dataset. 
            Treat this as a degraded forecast, not the highest-confidence live-weather version.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif result.target_date != reference_date + timedelta(days=1):
        st.markdown(
            f"""
            <div class="notice-box">
            <strong>Forecast date adjusted.</strong><br>
            Requested target: {reference_date + timedelta(days=1)}. Produced target: {result.target_date}.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="forecast-card">
            <div class="forecast-label">Predicted PM2.5 for {result.target_date}</div>
            <div class="forecast-value">{result.primary_prediction:.1f}</div>
            <div class="forecast-unit">micrograms per cubic meter</div>
            <div style="margin-top: 12px;">
                <span class="pill" style="background:{result.category_color};">{result.category}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_small_card(
            "Expected Range",
            f"{result.expected_low:.1f}-{result.expected_high:.1f}",
            f"Range uses conservative one-day-ahead MAE of {result.uncertainty_mae:.2f}.",
        )
    with c2:
        render_small_card(
            "Latest PM2.5 Input",
            f"{result.latest_pm25:.1f}",
            f"Latest usable day: {result.latest_pm25_date}; hourly values: {result.latest_pm25_hours or 'n/a'}.",
        )
    with c3:
        render_small_card(
            "Holiday Context",
            holiday_name(result.target_date),
            "Calendar variables are encoded consistently with the training pipeline.",
        )
    with c4:
        render_small_card(
            "Primary Model",
            result.primary_model_name,
            f"Holdout MAE: {result.primary_mae:.2f}; lower MAE is better.",
        )

    st.markdown(
        f"""
        <div class="status-box">
        <strong>Data mode:</strong> {result.data_mode}<br>
        <strong>Air-quality input:</strong> {result.data_note}<br>
        <strong>Weather input:</strong> {result.weather_note}<br>
        <strong>Generated:</strong> {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_forecast, tab_inputs, tab_method = st.tabs(["Forecast", "Inputs", "Method"])

    with tab_forecast:
        st.markdown(
            '<div class="section-note">The marker shows the next-day prediction. The vertical band uses historical holdout error as an uncertainty guide, not as a formal prediction interval.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(trend_figure(result), use_container_width=True)
        st.plotly_chart(comparison_figure(result), use_container_width=True)

    with tab_inputs:
        st.markdown(
            '<div class="section-note">These are the main numerical inputs used by the primary model. They help explain the forecast context without claiming direct causality.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(driver_figure(result), use_container_width=True)
        input_view = result.feature_row.T.reset_index()
        input_view.columns = ["feature", "value"]
        input_view["value"] = input_view["value"].map(lambda value: f"{float(value):.4f}")
        st.dataframe(input_view, use_container_width=True, hide_index=True)

    with tab_method:
        st.markdown(
            """
            ### What the app predicts

            The target is daily average PM2.5 for Nanjing on the next valid forecast date.

            ### Why two models are shown

            The primary model uses the strongest trained model from the research project, with tomorrow's weather supplied by the Open-Meteo weather forecast API. This avoids using tomorrow's observed weather, which would not be available at prediction time.

            The conservative cross-check model uses the stricter one-day-ahead model trained on lagged PM2.5, previous-day weather, and calendar variables. It is less accurate on the historical holdout set, but it is a useful sanity check.

            ### What the numbers mean

            MAE is mean absolute error, measured in PM2.5 units. A model MAE of 10.82 means the historical test predictions were off by about 10.82 micrograms per cubic meter on average.

            ### Limitations

            The app depends on modeled Open-Meteo air-quality inputs and forecast-weather inputs. It does not directly observe traffic, industrial emissions, construction activity, or regional pollutant transport. Sudden pollution episodes can therefore be missed.
            """
        )


if __name__ == "__main__":
    main()
