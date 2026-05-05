"""
Crop Yield Forecast — Streamlit App
Run: streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS   = ROOT / "outputs"
MODELS    = OUTPUTS / "models"

FEATURE_COLS = [
    "mean_temp", "total_precip", "mean_humidity", "mean_rftra",
    "max_lai", "max_tagp", "max_dvs", "season_days",
    "latitude", "longitude", "elevation", "year", "WAV", "crop_te",
]

# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "yield_full":   joblib.load(MODELS / "baseline_lgbm_fullseason.joblib"),
        "yield_day30":  joblib.load(MODELS / "early_lgbm_day30.joblib"),
        "yield_day60":  joblib.load(MODELS / "early_lgbm_day60.joblib"),
        "yield_day90":  joblib.load(MODELS / "early_lgbm_day90.joblib"),
        "success_day30": joblib.load(MODELS / "sim_success_lgbm_day30.joblib"),
        "success_day60": joblib.load(MODELS / "sim_success_lgbm_day60.joblib"),
        "success_day90": joblib.load(MODELS / "sim_success_lgbm_day90.joblib"),
    }

@st.cache_data
def load_agg():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.loader import load_season_agg
    return load_season_agg()

@st.cache_data
def load_crop_te():
    with open(PROCESSED / "crop_te_map.json") as f:
        d = json.load(f)
    global_mean = d.pop("__global_mean__")
    return d, global_mean

@st.cache_data
def load_climate():
    return pd.read_parquet(OUTPUTS / "climate_change_simulation.parquet")

@st.cache_data
def load_recommendations():
    return pd.read_parquet(OUTPUTS / "crop_recommendations.parquet")

# ── App layout ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Yield Forecast", layout="wide")
st.title("🌾 Crop Yield Forecast — Turkey")

page = st.sidebar.radio(
    "Navigation",
    ["Yield Prediction", "Harvest Success", "Crop Recommendation", "Climate Simulation"],
)

models    = load_models()
crop_te_map, global_mean = load_crop_te()

CROPS    = sorted(crop_te_map.keys())
WAV_MAP  = {"Dry (10 cm)": 10, "Normal (50 cm)": 50, "Wet (100 cm)": 100}
SCENARIO = ["dry", "normal", "wet"]

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Yield Prediction
# ══════════════════════════════════════════════════════════════════════════
if page == "Yield Prediction":
    st.header("Yield Prediction")
    st.caption("Predict end-of-season harvest (kg/ha) from season features.")

    col1, col2 = st.columns(2)
    with col1:
        crop     = st.selectbox("Crop", CROPS)
        lat      = st.slider("Latitude",  36.0, 42.0, 39.0, step=0.5)
        lon      = st.slider("Longitude", 26.0, 44.0, 35.0, step=0.5)
        elev     = st.number_input("Elevation (m)", 0, 3000, 800, step=50)
        year     = st.slider("Year", 2014, 2024, 2020)
        wav_lbl  = st.selectbox("Soil water scenario", list(WAV_MAP.keys()))

    with col2:
        mean_temp    = st.slider("Mean temperature (°C)", -5.0, 30.0, 13.0, 0.1)
        total_precip = st.slider("Total precipitation (mm)", 0, 3000, 600, 10)
        mean_humidity= st.slider("Mean humidity (%)", 20, 100, 63)
        mean_rftra   = st.slider("Mean RFTRA (water stress, 0–1)", 0.0, 1.0, 0.5, 0.01)
        max_lai      = st.slider("Max LAI", 0.0, 16.0, 4.0, 0.1)
        max_tagp     = st.slider("Max TAGP (kg/ha)", 0, 35000, 6000, 100)
        max_dvs      = st.slider("Max DVS", 0.0, 2.0, 1.8, 0.01)
        season_days  = st.number_input("Season length (days)", 100, 400, 366)

    cutoff = st.selectbox("Prediction cutoff", ["Full season", "Day 30", "Day 60", "Day 90"])
    model_key = {"Full season": "yield_full", "Day 30": "yield_day30",
                 "Day 60": "yield_day60", "Day 90": "yield_day90"}[cutoff]

    X = pd.DataFrame([{
        "mean_temp": mean_temp, "total_precip": total_precip,
        "mean_humidity": mean_humidity, "mean_rftra": mean_rftra,
        "max_lai": max_lai, "max_tagp": max_tagp, "max_dvs": max_dvs,
        "season_days": season_days, "latitude": lat, "longitude": lon,
        "elevation": elev, "year": year, "WAV": WAV_MAP[wav_lbl],
        "crop_te": crop_te_map.get(crop, global_mean),
    }])[FEATURE_COLS]

    pred = models[model_key].predict(X)[0]
    pred = max(0, pred)

    st.divider()
    st.metric("Predicted yield", f"{pred:,.0f} kg/ha")

    # Context: how does this compare to the crop's average?
    avg = crop_te_map.get(crop, global_mean)
    delta = pred - avg
    st.caption(f"Crop average (train): {avg:,.0f} kg/ha  |  Δ {delta:+,.0f} kg/ha")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Harvest Success
# ══════════════════════════════════════════════════════════════════════════
elif page == "Harvest Success":
    st.header("Harvest Success Probability")
    st.caption("Predict the probability of a successful harvest using early-season data only.")

    col1, col2 = st.columns(2)
    with col1:
        crop     = st.selectbox("Crop", CROPS)
        lat      = st.slider("Latitude",  36.0, 42.0, 39.0, step=0.5)
        lon      = st.slider("Longitude", 26.0, 44.0, 35.0, step=0.5)
        elev     = st.number_input("Elevation (m)", 0, 3000, 800, step=50)
        year     = st.slider("Year", 2014, 2024, 2020)
        wav_lbl  = st.selectbox("Soil water scenario", list(WAV_MAP.keys()))

    with col2:
        mean_temp    = st.slider("Mean temperature (°C)", -5.0, 30.0, 13.0, 0.1)
        total_precip = st.slider("Total precipitation (mm)", 0, 3000, 600, 10)
        mean_humidity= st.slider("Mean humidity (%)", 20, 100, 63)
        mean_rftra   = st.slider("Mean RFTRA (water stress, 0–1)", 0.0, 1.0, 0.5, 0.01)
        max_lai      = st.slider("Max LAI", 0.0, 16.0, 4.0, 0.1)
        max_tagp     = st.slider("Max TAGP (kg/ha)", 0, 35000, 6000, 100)
        max_dvs      = st.slider("Max DVS", 0.0, 2.0, 1.8, 0.01)
        season_days  = st.number_input("Season length (days)", 100, 400, 366)

    cutoff = st.selectbox("Day cutoff", ["Day 30", "Day 60", "Day 90"])
    model_key = {"Day 30": "success_day30", "Day 60": "success_day60",
                 "Day 90": "success_day90"}[cutoff]

    X = pd.DataFrame([{
        "mean_temp": mean_temp, "total_precip": total_precip,
        "mean_humidity": mean_humidity, "mean_rftra": mean_rftra,
        "max_lai": max_lai, "max_tagp": max_tagp, "max_dvs": max_dvs,
        "season_days": season_days, "latitude": lat, "longitude": lon,
        "elevation": elev, "year": year, "WAV": WAV_MAP[wav_lbl],
        "crop_te": crop_te_map.get(crop, global_mean),
    }])[FEATURE_COLS]

    proba = models[model_key].predict_proba(X)[0][1]

    st.divider()
    color = "green" if proba >= 0.7 else "orange" if proba >= 0.4 else "red"
    st.metric("Success probability", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= 0.7:
        st.success("High likelihood of successful harvest.")
    elif proba >= 0.4:
        st.warning("Moderate risk — conditions are marginal.")
    else:
        st.error("High risk of harvest failure.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Crop Recommendation
# ══════════════════════════════════════════════════════════════════════════
elif page == "Crop Recommendation":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    st.header("Crop Recommendation")
    st.caption("Best crop per location × year × scenario based on full-season model predictions.")

    # ── Load and predict for all scenarios ──────────────────────────────
    @st.cache_data
    def build_all_recs(_model, _crop_te_map, _global_mean):
        agg = load_agg()
        df  = agg[agg["sim_success"] == 1].copy()
        df["crop_te"] = df["crop_name"].map(_crop_te_map).fillna(_global_mean)
        df["predicted_twso"] = _model.predict(df[FEATURE_COLS])
        best = (
            df.loc[df.groupby(["latitude", "longitude", "year", "wav_scenario"])
                     ["predicted_twso"].idxmax()]
            [["latitude", "longitude", "year", "wav_scenario",
              "crop_name", "predicted_twso", "harvest_twso"]]
            .reset_index(drop=True)
        )
        return best, df

    with st.spinner("Computing recommendations for all scenarios..."):
        all_recs, full_df = build_all_recs(
            models["yield_full"], crop_te_map, global_mean
        )

    YEARS     = sorted(all_recs["year"].unique(), reverse=True)
    ALL_CROPS = sorted(all_recs["crop_name"].unique())

    # ── Sidebar filters ──────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    scenario  = st.sidebar.selectbox("Scenario", ["normal", "dry", "wet"])
    year      = st.sidebar.selectbox("Year", YEARS)
    crop_filt = st.sidebar.multiselect("Show only these crops", ALL_CROPS, default=ALL_CROPS)
    top_n     = st.sidebar.slider("Top N crops in ranking", 1, 10, 5)

    filtered = all_recs[
        (all_recs["year"] == year) &
        (all_recs["wav_scenario"] == scenario) &
        (all_recs["crop_name"].isin(crop_filt))
    ].copy()

    # ── Color-coded scatter map ──────────────────────────────────────────
    st.subheader(f"Recommended crop by location — {year} ({scenario})")

    palette    = plt.get_cmap("tab20").colors
    crop_list  = sorted(filtered["crop_name"].unique())
    color_map  = {c: palette[i % len(palette)] for i, c in enumerate(crop_list)}

    fig, ax = plt.subplots(figsize=(13, 6))
    for crop, grp in filtered.groupby("crop_name"):
        ax.scatter(grp["longitude"], grp["latitude"],
                   color=color_map[crop], s=220, marker="s",
                   edgecolors="gray", linewidths=0.3, label=crop)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Recommended Crop — {year} ({scenario} scenario)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Top N crops ranking ──────────────────────────────────────────────
    st.subheader(f"Top {top_n} most recommended crops")
    top = (filtered["crop_name"].value_counts()
           .head(top_n).reset_index())
    top.columns = ["Crop", "# Locations"]
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.divider()

    # ── Location deep-dive ───────────────────────────────────────────────
    st.subheader("Location deep-dive")
    lats    = sorted(all_recs["latitude"].unique())
    lons    = sorted(all_recs["longitude"].unique())
    col1, col2 = st.columns(2)
    sel_lat = col1.selectbox("Latitude",  lats, index=len(lats) // 2)
    sel_lon = col2.selectbox("Longitude", lons, index=len(lons) // 2)

    loc_recs = all_recs[
        (all_recs["latitude"] == sel_lat) &
        (all_recs["longitude"] == sel_lon)
    ]

    # Top 3 for selected year × scenario
    loc_year_scen = full_df[
        (full_df["latitude"] == sel_lat) &
        (full_df["longitude"] == sel_lon) &
        (full_df["year"] == year) &
        (full_df["wav_scenario"] == scenario)
    ].sort_values("predicted_twso", ascending=False).head(3)

    if not loc_year_scen.empty:
        st.markdown(f"**Top 3 crops — lat {sel_lat}, lon {sel_lon}, {year}, {scenario}**")
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], loc_year_scen.iterrows()):
            col.metric(row["crop_name"], f"{row['predicted_twso']:,.0f} kg/ha")
    else:
        st.info("No data for this location.")

    # Year-over-year recommendation history
    st.markdown(f"**Year-over-year recommendation — {scenario} scenario**")
    history = loc_recs[loc_recs["wav_scenario"] == scenario].sort_values("year")
    if not history.empty:
        fig2, ax2 = plt.subplots(figsize=(11, 3))
        for _, row in history.iterrows():
            c = color_map.get(row["crop_name"], "gray")
            ax2.bar(row["year"], row["predicted_twso"], color=c, width=0.6, label=row["crop_name"])
            ax2.text(row["year"], row["predicted_twso"] + 50, row["crop_name"],
                     ha="center", va="bottom", fontsize=7, rotation=45)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Predicted yield (kg/ha)")
        ax2.set_title(f"Best Crop per Year — lat {sel_lat}, lon {sel_lon} ({scenario})")
        ax2.set_xticks(history["year"])
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Consistency
        total_years   = history["year"].nunique()
        top_crop      = history["crop_name"].mode()[0]
        consistency   = (history["crop_name"] == top_crop).sum() / total_years
        st.caption(f"Most consistent recommendation: **{top_crop}** ({consistency:.0%} of years)")
    else:
        st.info("No history for this location and scenario.")

    # Scenario comparison for selected location × year
    st.divider()
    st.subheader(f"Scenario comparison — lat {sel_lat}, lon {sel_lon}, {year}")
    scen_rows = all_recs[
        (all_recs["latitude"] == sel_lat) &
        (all_recs["longitude"] == sel_lon) &
        (all_recs["year"] == year)
    ]
    if not scen_rows.empty:
        cols = st.columns(3)
        for col, scen in zip(cols, ["dry", "normal", "wet"]):
            r = scen_rows[scen_rows["wav_scenario"] == scen]
            if not r.empty:
                col.metric(
                    f"{scen.capitalize()} scenario",
                    r.iloc[0]["crop_name"],
                    f"{r.iloc[0]['predicted_twso']:,.0f} kg/ha"
                )
            else:
                col.info(f"No data for {scen}")
    else:
        st.info("No scenario data for this location and year.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Climate Simulation
# ══════════════════════════════════════════════════════════════════════════
elif page == "Climate Simulation":
    st.header("Climate Change Simulation")
    st.caption("Impact of temperature increase on crop yields (normal scenario, full-season model).")

    climate = load_climate()

    import matplotlib.pyplot as plt

    col1, col2 = st.columns(2)
    with col1:
        crop_filter = st.multiselect("Filter by crop", sorted(climate["crop_name"].unique()),
                                     default=sorted(climate["crop_name"].unique()))
    with col2:
        year_filter = st.multiselect("Filter by year", sorted(climate["year"].unique()),
                                     default=sorted(climate["year"].unique()))

    df = climate[
        climate["crop_name"].isin(crop_filter) &
        climate["year"].isin(year_filter)
    ]

    st.divider()

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean yield change", f"{df['yield_change_pct'].mean():+.2f}%")
    c2.metric("Most impacted crop",
              df.groupby("crop_name")["yield_change_pct"].median().idxmin())
    c3.metric("Least impacted crop",
              df.groupby("crop_name")["yield_change_pct"].median().idxmax())

    # Bar chart by crop
    st.subheader("Yield change by crop (+2°C)")
    crop_impact = (df.groupby("crop_name")["yield_change_pct"]
                   .median().sort_values().reset_index())
    crop_impact.columns = ["crop", "yield_change_pct"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in crop_impact["yield_change_pct"]]
    ax.barh(crop_impact["crop"], crop_impact["yield_change_pct"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Median Yield Change (%)")
    ax.set_title("Yield Change with +2°C Warming")
    st.pyplot(fig)
    plt.close()

    # Spatial view
    st.subheader("Spatial impact map")
    spatial = df.groupby(["latitude", "longitude"])["yield_change_pct"].median().reset_index()
    st.map(spatial.rename(columns={"latitude": "lat", "longitude": "lon"}))
