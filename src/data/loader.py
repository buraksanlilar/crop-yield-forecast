"""
Dataset yükleme yardımcıları — 5GB parquet için memory-efficient okuma.
"""
import duckdb
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).parents[2] / "data" / "raw" / "final_hourly_pcse_dataset_multiyear.parquet"

WOFOST_COLS = ["DVS", "LAI", "TAGP", "TWSO", "TWLV", "TWST", "TWRT", "TRA", "RD", "SM", "WWLOW", "RFTRA"]
WEATHER_COLS = ["AIR_TEMP", "AIR_HUMIDITY", "PRECIP", "SOIL_TEMP_0_7", "SOIL_MOISTURE_0_7"]
ID_COLS = ["DATETIME", "latitude", "longitude", "elevation", "crop_name", "variety_name", "year", "wav_scenario", "WAV"]
TARGET_COLS = ["harvest_twso", "sim_success"]


def query(sql: str) -> pd.DataFrame:
    """DuckDB ile doğrudan parquet üzerinde SQL çalıştır (RAM'e yüklemeden)."""
    return duckdb.query(sql).df()


def load_sample(n: int = 500_000, seed: int = 42) -> pd.DataFrame:
    """Hızlı keşif için rastgele örnek yükle."""
    return query(f"""
        SELECT * FROM '{RAW_PATH}'
        USING SAMPLE {n} ROWS (BERNOULLI, {seed})
    """)


def load_crop(crop: str, wav_scenario: str | None = None) -> pd.DataFrame:
    """Tek bitki için tüm veriyi yükle, opsiyonel su senaryosu filtresi."""
    where = f"WHERE crop_name = '{crop}'"
    if wav_scenario:
        where += f" AND wav_scenario = '{wav_scenario}'"
    return query(f"SELECT * FROM '{RAW_PATH}' {where}")


def load_season_agg() -> pd.DataFrame:
    """
    Her (lokasyon × bitki × yıl × WAV) kombinasyonu için sezon özeti.
    Günlük forward-fill'den kurtulmak için günlük max alınır.
    """
    return query(f"""
        SELECT
            latitude, longitude, elevation,
            crop_name, variety_name, year, wav_scenario, WAV,
            MAX(harvest_twso)   AS harvest_twso,
            MAX(sim_success)    AS sim_success,
            AVG(AIR_TEMP)       AS mean_temp,
            SUM(PRECIP)         AS total_precip,
            AVG(AIR_HUMIDITY)   AS mean_humidity,
            AVG(RFTRA)          AS mean_rftra,
            MIN(RFTRA)          AS min_rftra,
            MAX(LAI)            AS max_lai,
            MAX(TAGP)           AS max_tagp,
            MAX(DVS)            AS max_dvs,
            COUNT(DISTINCT DATE_TRUNC('day', DATETIME::TIMESTAMP)) AS season_days
        FROM '{RAW_PATH}'
        GROUP BY
            latitude, longitude, elevation,
            crop_name, variety_name, year, wav_scenario, WAV
    """)


def schema_info() -> pd.DataFrame:
    """Kolon isimleri ve tipleri."""
    return query(f"DESCRIBE SELECT * FROM '{RAW_PATH}' LIMIT 1")
