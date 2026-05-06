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
        "yield_full":    joblib.load(MODELS / "baseline"         / "baseline_lgbm_fullseason.joblib"),
        "yield_day30":   joblib.load(MODELS / "early_prediction" / "early_lgbm_day30.joblib"),
        "yield_day60":   joblib.load(MODELS / "early_prediction" / "early_lgbm_day60.joblib"),
        "yield_day90":   joblib.load(MODELS / "early_prediction" / "early_lgbm_day90.joblib"),
        "success_day30": joblib.load(MODELS / "sim_success"      / "sim_success_lgbm_day30.joblib"),
        "success_day60": joblib.load(MODELS / "sim_success"      / "sim_success_lgbm_day60.joblib"),
        "success_day90": joblib.load(MODELS / "sim_success"      / "sim_success_lgbm_day90.joblib"),
        "iot_day30":     joblib.load(MODELS / "iot"              / "iot_lgbm_day30.joblib"),
        "iot_day60":     joblib.load(MODELS / "iot"              / "iot_lgbm_day60.joblib"),
        "iot_day90":     joblib.load(MODELS / "iot"              / "iot_lgbm_day90.joblib"),
    }

@st.cache_data
def load_moisture_profiles():
    with open(PROCESSED / "crop_daily_moisture_profile.json") as f:
        return json.load(f)

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
    return pd.read_parquet(OUTPUTS / "predictions" / "climate_change_simulation.parquet")

@st.cache_data
def load_recommendations():
    return pd.read_parquet(OUTPUTS / "predictions" / "crop_recommendations.parquet")

# ── App layout ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Yield Forecast", layout="wide")
st.title("🌾 Crop Yield Forecast — Turkey")

page = st.sidebar.radio(
    "Navigation",
    ["Yield Prediction", "Harvest Success", "Crop Recommendation", "Climate Simulation", "IoT Decision Engine"],
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


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — IoT Decision Engine
# ══════════════════════════════════════════════════════════════════════════
elif page == "IoT Decision Engine":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    st.header("IoT Decision Engine")
    st.caption("Enter today's sensor readings → get irrigation decision + yield forecast.")

    # Target encoding per crop (from training data)
    CROP_TE = {
        "barley": 4850.5, "cassava": 1106.6, "chickpea": 61.0, "cotton": 545.1,
        "cowpea": 438.8, "fababean": 4096.8, "groundnut": 224.7, "maize": 2179.9,
        "millet": 672.0, "mungbean": 523.5, "pigeonpea": 1924.9, "potato": 2367.9,
        "rapeseed": 1539.9, "rice": 565.0, "seed_onion": 5638.9, "sorghum": 1161.9,
        "soybean": 1069.3, "sugarbeet": 4436.6, "sunflower": 601.9,
        "sweetpotato": 2112.3, "tobacco": 11.8, "wheat": 4900.8,
    }
    CROP_PRICES_IOT = {
        "wheat": 8.5, "barley": 7.0, "maize": 9.0, "chickpea": 22.0,
        "rapeseed": 18.0, "sunflower": 20.0, "soybean": 19.0, "cotton": 25.0,
        "potato": 6.0, "sugarbeet": 3.5, "rice": 14.0, "tobacco": 45.0,
    }
    WATER_COST = 2.5  # TL/m³

    moisture_profiles = load_moisture_profiles()
    iot_m = {
        30: models["iot_day30"],
        60: models["iot_day60"],
        90: models["iot_day90"],
    }

    IOT_CROPS = sorted(CROP_TE.keys())

    # ── Inputs ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tarla & Konum")
        crop         = st.selectbox("Bitki", IOT_CROPS, index=IOT_CROPS.index("wheat"))
        season_day   = st.slider("Sezon günü (ekim sonrası)", 1, 365, 45)
        field_area_ha= st.number_input("Tarla alanı (hektar)", 0.1, 500.0, 1.0, 0.1)
        root_depth   = st.slider("Kök / sulama derinliği (m)", 0.10, 0.60, 0.30, 0.05)
        lat          = st.slider("Enlem", 36.0, 42.0, 39.0, 0.5)
        lon          = st.slider("Boylam", 26.0, 44.0, 35.0, 0.5)
        elev         = st.number_input("Rakım (m)", 0, 3000, 800, 50)
        year         = st.slider("Yıl", 2014, 2030, 2024)

    with col2:
        st.subheader("IoT Sensör Okumaları")
        current_sm    = st.slider("Mevcut toprak nemi (m³/m³)", 0.05, 0.50, 0.22, 0.01)
        mean_temp     = st.slider("Ort. hava sıcaklığı — sezon (°C)", -5.0, 35.0, 13.0, 0.5)
        total_precip  = st.slider("Toplam yağış — sezon (mm)", 0, 600, 95, 5)
        mean_humidity = st.slider("Ort. hava nemi — sezon (%)", 20, 100, 65)
        mean_soil_temp= st.slider("Ort. toprak sıcaklığı — sezon (°C)", 0.0, 35.0, 10.0, 0.5)
        max_lai       = st.slider("Maks. LAI (Sentinel-2)", 0.0, 6.0, 1.8, 0.1)

    # ── Compute ───────────────────────────────────────────────────────────
    if crop not in moisture_profiles:
        st.error(f"{crop} için nem profili bulunamadı.")
        st.stop()

    p   = moisture_profiles[crop]
    idx = min(season_day, p["max_day"])

    target_sm   = p["optimal_sm"][idx]
    low_sm      = p["low_sm"][idx]
    critical_sm = p["critical_sm"][idx]
    upper_sm    = p["upper_sm"][idx]

    if current_sm < critical_sm:
        alarm = "KRİTİK"
    elif current_sm < low_sm:
        alarm = "SULA"
    elif current_sm > upper_sm:
        alarm = "DURDUR"
    else:
        alarm = "NORMAL"

    irr_target = target_sm if alarm == "KRİTİK" else low_sm
    delta_sm   = max(0.0, irr_target - current_sm)
    field_m2   = field_area_ha * 10_000
    litre      = delta_sm * field_m2 * root_depth * 1000
    water_cost = (litre / 1000) * WATER_COST

    if alarm == "KRİTİK":
        decision = "SULA (zorunlu)"
    elif alarm == "SULA":
        decision = "SULA"
    elif alarm == "DURDUR":
        decision = "SULAMA DURDUR"
    else:
        decision = "BEKLEME"

    # IoT verim tahmini
    chk = 30 if season_day <= 45 else (60 if season_day <= 75 else 90)
    feat = pd.DataFrame([{
        "mean_temp": mean_temp, "total_precip": total_precip,
        "mean_humidity": mean_humidity, "mean_soil_temp": mean_soil_temp,
        "mean_soil_moisture": current_sm, "max_lai": max_lai,
        "season_days": season_day, "latitude": lat, "longitude": lon,
        "elevation": elev, "year": year, "WAV": 50,
        "crop_te": CROP_TE.get(crop, 1000.0),
    }])
    yield_forecast = max(0.0, float(iot_m[chk].predict(feat)[0]))

    yield_optimal  = p["yield_optimal_kg_ha"]
    yield_drought  = p["yield_drought_kg_ha"]
    yield_at_risk  = max(0.0, yield_optimal - yield_drought)
    price          = CROP_PRICES_IOT.get(crop, 10.0)
    season_risk_tl = yield_at_risk * field_area_ha * price

    # ── Output ────────────────────────────────────────────────────────────
    st.divider()

    # Alarm banner
    alarm_colors = {"KRİTİK": "error", "SULA": "warning", "DURDUR": "warning", "NORMAL": "success"}
    getattr(st, alarm_colors[alarm])(f"**{alarm}** — {decision}")

    # Metrics row 1: nem
    st.subheader("Toprak Nemi")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mevcut nem",  f"{current_sm:.3f} m³/m³")
    c2.metric("Hedef nem",   f"{irr_target:.3f} m³/m³", delta=f"{irr_target - current_sm:+.3f}")
    c3.metric("Kritik eşik", f"{critical_sm:.3f} m³/m³")
    c4.metric("Üst sınır",   f"{upper_sm:.3f} m³/m³")

    # Metrics row 2: sulama
    st.subheader("Sulama")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gerekli su",   f"{litre:,.0f} litre")
    c2.metric("Su maliyeti",  f"{water_cost:,.2f} TL")
    c3.metric("Model checkpoint", f"Gün {chk}")

    # Metrics row 3: verim
    st.subheader("Verim Referansı (Tarihi)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimum sezon",  f"{yield_optimal:,.0f} kg/ha", help="Üst %25 verimli sezonlar medyanı")
    c2.metric("Kurak sezon",    f"{yield_drought:,.0f} kg/ha", help="Alt %25 verimli sezonlar medyanı")
    c3.metric("Sezonluk risk",  f"{season_risk_tl:,.0f} TL",  help="Kuraklık olursa tahmini kayıp")
    st.caption("Verim referansları, bu bitki için Türkiye genelinde geçmiş sezonlardan hesaplanmıştır. Sulama kararı nem profiline dayanmaktadır.")

    # Nem profil grafiği
    st.divider()
    st.subheader("Sezon Nem Profili")
    days = list(range(min(season_day + 60, p["max_day"] + 1)))
    opt_line  = [p["optimal_sm"][d] for d in days]
    low_line  = [p["low_sm"][d] for d in days]
    high_line = [p["high_sm"][d] for d in days]
    crit_line = [p["critical_sm"][d] for d in days]
    up_line   = [p["upper_sm"][d] for d in days]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(days, low_line, high_line, alpha=0.15, color="steelblue", label="Optimum aralık")
    ax.plot(days, opt_line,  color="steelblue", lw=2,   label="Optimal hedef")
    ax.plot(days, crit_line, color="red",    lw=1.2, ls="--", label="Kritik alt sınır")
    ax.plot(days, up_line,   color="orange", lw=1.2, ls=":",  label="Üst sınır")
    ax.axvline(season_day, color="black", lw=1.5, ls="-", label=f"Bugün (gün {season_day})")
    ax.axhline(current_sm, color="purple", lw=1.2, ls="-.", label=f"Mevcut nem ({current_sm:.3f})")
    alarm_c = {"KRİTİK": "red", "SULA": "orange", "DURDUR": "orange", "NORMAL": "green"}[alarm]
    ax.scatter([season_day], [current_sm], color=alarm_c, s=120, zorder=5)
    ax.set_xlabel("Gün (ekim sonrası)")
    ax.set_ylabel("Toprak nemi (m³/m³)")
    ax.set_title(f"{crop.capitalize()} — Günlük Nem Profili")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Aktüatör komutu
    st.divider()
    st.subheader("Aktüatör Komutu (MQTT Payload)")
    flow_rate = st.number_input("Pompa debisi (litre/dakika)", 100, 5000, 500, 100)
    duration  = litre / flow_rate if flow_rate > 0 else 0
    cmd = {
        "action":       "OPEN" if decision.startswith("SULA") else ("CLOSE" if alarm == "DURDUR" else "IDLE"),
        "duration_min": round(duration, 1),
        "litre":        round(litre, 0),
        "alarm":        alarm,
        "crop":         crop,
        "season_day":   season_day,
        "field_ha":     field_area_ha,
    }
    st.json(cmd)
