"""
Etapa 02 - Engenharia de atributos (features).

Lê os dados brutos extraídos na etapa 01, gera features pré-jogo e salva em:
- data/processed/model_dataset.csv
- data/processed/model_dataset.parquet
- data/processed/model_dataset.xlsx
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "understat_matches_raw.parquet"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_FILE}. Rode primeiro o script 01_extract_understat.py"
        )
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["league", "season", "date", "home_team", "away_team"]).reset_index(drop=True)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    played_mask = out["home_goals"].notna() & out["away_goals"].notna()

    out["total_goals"] = out["home_goals"] + out["away_goals"]

    out["target_home_win"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["target_draw"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["target_away_win"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    out.loc[played_mask, "target_home_win"] = (
        out.loc[played_mask, "home_goals"] > out.loc[played_mask, "away_goals"]
    ).astype("Int64")
    out.loc[played_mask, "target_draw"] = (
        out.loc[played_mask, "home_goals"] == out.loc[played_mask, "away_goals"]
    ).astype("Int64")
    out.loc[played_mask, "target_away_win"] = (
        out.loc[played_mask, "home_goals"] < out.loc[played_mask, "away_goals"]
    ).astype("Int64")
    return out


def _team_long_table(df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_idx": df.index,
            "date": df["date"],
            "league": df["league"],
            "season": df["season"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": 1,
            "goals_for": df["home_goals"],
            "goals_against": df["away_goals"],
            "xg_for": df["home_xg"],
            "xg_against": df["away_xg"],
            "ppda": df["home_ppda"],
            "deep": df["home_deep"],
        }
    )

    away = pd.DataFrame(
        {
            "match_idx": df.index,
            "date": df["date"],
            "league": df["league"],
            "season": df["season"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "goals_for": df["away_goals"],
            "goals_against": df["home_goals"],
            "xg_for": df["away_xg"],
            "xg_against": df["home_xg"],
            "ppda": df["away_ppda"],
            "deep": df["away_deep"],
        }
    )

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "league", "date", "match_idx", "is_home"]).reset_index(drop=True)
    return long_df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.copy()
    long_df = _team_long_table(out)

    # Reset por temporada evita vazamento/artefatos entre temporadas (offseason)
    group_cols = ["team", "league", "season"]
    metrics = ["goals_for", "goals_against", "xg_for", "xg_against", "ppda", "deep"]

    for m in metrics:
        long_df[f"{m}_avg_{window}"] = (
            long_df.groupby(group_cols)[m]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .astype(float)
        )

    played_mask = long_df["goals_for"].notna() & long_df["goals_against"].notna()
    long_df["form_points"] = pd.Series(pd.NA, index=long_df.index, dtype="Float64")
    long_df.loc[played_mask, "form_points"] = 0.0
    long_df.loc[played_mask & (long_df["goals_for"] > long_df["goals_against"]), "form_points"] = 3.0
    long_df.loc[played_mask & (long_df["goals_for"] == long_df["goals_against"]), "form_points"] = 1.0
    long_df[f"form_points_avg_{window}"] = (
        long_df.groupby(group_cols)["form_points"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .astype(float)
    )

    # Features de contexto: descanso e congestão de jogos
    long_df = long_df.sort_values(["team", "league", "date", "match_idx", "is_home"]).reset_index(drop=True)
    long_df["rest_days"] = long_df.groupby(group_cols)["date"].diff().dt.days
    long_df["rest_days"] = long_df["rest_days"].clip(lower=0)

    def _games_last_n_days(group: pd.DataFrame, n_days: int = 14) -> pd.Series:
        dates = group["date"].to_numpy()
        counts = np.zeros(len(group), dtype=float)
        for i in range(len(group)):
            if i == 0:
                counts[i] = 0.0
                continue
            delta_days = (dates[i] - dates[:i]) / np.timedelta64(1, "D")
            counts[i] = float(np.sum((delta_days >= 0) & (delta_days <= n_days)))
        return pd.Series(counts, index=group.index)

    long_df["games_last_14d"] = (
        long_df.groupby(group_cols, group_keys=False)
        .apply(_games_last_n_days)
        .astype(float)
    )

    home_feats = long_df[long_df["is_home"] == 1].copy()
    away_feats = long_df[long_df["is_home"] == 0].copy()

    feat_cols = [c for c in long_df.columns if c.endswith(f"avg_{window}") or c in ["rest_days", "games_last_14d"]]
    home_rename = {c: f"home_{c}" for c in feat_cols}
    away_rename = {c: f"away_{c}" for c in feat_cols}

    home_feats = home_feats[["match_idx", *feat_cols]].rename(columns=home_rename)
    away_feats = away_feats[["match_idx", *feat_cols]].rename(columns=away_rename)

    out = out.reset_index(names="match_idx")
    out = out.merge(home_feats, on="match_idx", how="left")
    out = out.merge(away_feats, on="match_idx", how="left")

    base_home = ["goals_for", "goals_against", "xg_for", "xg_against", "form_points"]
    for m in base_home:
        h_col = f"home_{m}_avg_{window}"
        a_col = f"away_{m}_avg_{window}"
        if h_col in out.columns and a_col in out.columns:
            out[f"diff_{m}_avg_{window}"] = out[h_col] - out[a_col]

    if "home_rest_days" in out.columns and "away_rest_days" in out.columns:
        out["diff_rest_days"] = out["home_rest_days"] - out["away_rest_days"]
    if "home_games_last_14d" in out.columns and "away_games_last_14d" in out.columns:
        out["diff_games_last_14d"] = out["home_games_last_14d"] - out["away_games_last_14d"]

    out = out.drop(columns=["match_idx"])
    out = out.sort_values(["league", "season", "date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def save_outputs(df: pd.DataFrame) -> None:
    csv_path = PROCESSED_DIR / "model_dataset.csv"
    parquet_path = PROCESSED_DIR / "model_dataset.parquet"
    xlsx_path = PROCESSED_DIR / "model_dataset.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    df.to_excel(xlsx_path, index=False)

    quality_report = {
        "n_rows": int(len(df)),
        "n_duplicates_match_key": int(df.duplicated(subset=["date", "league", "home_team", "away_team"]).sum()),
        "missing_rate_top_10": (df.isna().mean().sort_values(ascending=False).head(10).round(4)).to_dict(),
    }
    with open(PROCESSED_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)

    print("=== Features concluídas ===")
    print(f"Dataset modelagem: {len(df):,} linhas")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Excel: {xlsx_path}")
    print(f"Qualidade: {PROCESSED_DIR / 'data_quality_report.json'}")


if __name__ == "__main__":
    data = load_data()
    data = add_targets(data)
    data = add_rolling_features(data, window=5)
    save_outputs(data)
