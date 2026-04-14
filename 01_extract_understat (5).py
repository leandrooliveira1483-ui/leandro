"""
Etapa 01 - Extração dos dados do Understat via soccerdata.

Objetivo:
- Baixar partidas de ligas/temporadas selecionadas.
- Salvar dados brutos em CSV, Parquet e Excel.
- Gerar um arquivo consolidado para as próximas etapas.

Uso (Google Colab ou local):
    python 01_extract_understat.py

Antes de rodar no Colab:
    !python -m pip install -q -r colab_pipeline/requirements_colab.txt
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Iterable, List

import pandas as pd
import soccerdata as sd


# ==========================
# CONFIGURAÇÕES
# ==========================
LEAGUES: List[str] = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "BRA-Serie A",
]
SEASONS: List[str] = [
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
]

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte MultiIndex columns em nomes simples."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(c) for c in col if str(c) != ""]).strip("_") for col in out.columns]
    return out


def _find_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_understat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas esperadas pelas etapas seguintes."""
    out = df.copy()

    date_col = _find_first_existing(out, ["date", "game_date", "datetime"])
    league_col = _find_first_existing(out, ["league", "competition"])
    season_col = _find_first_existing(out, ["season"])
    home_col = _find_first_existing(out, ["home_team", "team_home", "home"])
    away_col = _find_first_existing(out, ["away_team", "team_away", "away"])

    goals_home_col = _find_first_existing(out, ["home_goals", "goals_home", "home_score"])
    goals_away_col = _find_first_existing(out, ["away_goals", "goals_away", "away_score"])

    xg_home_col = _find_first_existing(out, ["home_xg", "xg_home"])
    xg_away_col = _find_first_existing(out, ["away_xg", "xg_away"])

    ppda_home_col = _find_first_existing(out, ["home_ppda", "ppda_home"])
    ppda_away_col = _find_first_existing(out, ["away_ppda", "ppda_away"])

    deep_home_col = _find_first_existing(out, ["home_deep", "deep_home"])
    deep_away_col = _find_first_existing(out, ["away_deep", "deep_away"])

    rename_map = {}
    if date_col:
        rename_map[date_col] = "date"
    if league_col:
        rename_map[league_col] = "league"
    if season_col:
        rename_map[season_col] = "season"
    if home_col:
        rename_map[home_col] = "home_team"
    if away_col:
        rename_map[away_col] = "away_team"
    if goals_home_col:
        rename_map[goals_home_col] = "home_goals"
    if goals_away_col:
        rename_map[goals_away_col] = "away_goals"
    if xg_home_col:
        rename_map[xg_home_col] = "home_xg"
    if xg_away_col:
        rename_map[xg_away_col] = "away_xg"
    if ppda_home_col:
        rename_map[ppda_home_col] = "home_ppda"
    if ppda_away_col:
        rename_map[ppda_away_col] = "away_ppda"
    if deep_home_col:
        rename_map[deep_home_col] = "home_deep"
    if deep_away_col:
        rename_map[deep_away_col] = "away_deep"

    out = out.rename(columns=rename_map)

    expected_min_cols = ["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [c for c in expected_min_cols if c not in out.columns]
    if missing:
        raise ValueError(
            f"Não foi possível padronizar colunas mínimas do Understat. Faltando: {missing}. "
            f"Colunas disponíveis: {list(out.columns)}"
        )

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")

    for c in ["home_xg", "away_xg", "home_ppda", "away_ppda", "home_deep", "away_deep"]:
        if c not in out.columns:
            out[c] = pd.NA
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values(["league", "season", "date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def _fetch_single_league_season(league: str, season: str) -> pd.DataFrame:
    reader = sd.Understat(leagues=[league], seasons=[season])

    if hasattr(reader, "read_schedule"):
        df = reader.read_schedule(include_matches_without_data=True)
    elif hasattr(reader, "read_league_results"):
        df = reader.read_league_results()
    else:
        methods = [m for m in dir(reader) if m.startswith("read_")]
        raise AttributeError(
            "A classe Understat não possui read_schedule/read_league_results. "
            f"Métodos disponíveis: {methods}"
        )

    df = _flatten_columns(df)
    df = df.reset_index(drop=False)
    df = _flatten_columns(df)
    return df


def fetch_understat(leagues: list[str], seasons: list[str], retries: int = 3, retry_wait_sec: int = 4) -> pd.DataFrame:
    """
    Baixa jogos do Understat com retries por liga/temporada.
    Em caso de timeout/TLS em uma combinação, continua com as demais.
    """
    chunks: list[pd.DataFrame] = []
    failures: list[str] = []

    for league in leagues:
        for season in seasons:
            ok = False
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    print(f"[INFO] Baixando {league} - {season} (tentativa {attempt}/{retries})")
                    part = _fetch_single_league_season(league, season)
                    chunks.append(part)
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001 - queremos robustez operacional no Colab
                    last_err = exc
                    if attempt < retries:
                        time.sleep(retry_wait_sec)
            if not ok:
                msg = f"{league} - {season} -> {type(last_err).__name__}: {last_err}"
                print(f"[WARN] Falha ao baixar {msg}")
                failures.append(msg)

    if not chunks:
        raise RuntimeError(
            "Nenhuma liga/temporada foi baixada com sucesso. "
            "Verifique conexão com Understat ou tente novamente em alguns minutos."
        )

    league_results = pd.concat(chunks, ignore_index=True)
    out = _normalize_understat_columns(league_results)

    if failures:
        fail_path = RAW_DIR / "understat_download_failures.txt"
        fail_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"[WARN] Houve falhas parciais. Detalhes em: {fail_path}")

    return out


def save_outputs(df: pd.DataFrame) -> None:
    csv_path = RAW_DIR / "understat_matches_raw.csv"
    parquet_path = RAW_DIR / "understat_matches_raw.parquet"
    xlsx_path = RAW_DIR / "understat_matches_raw.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    df.to_excel(xlsx_path, index=False)

    summary = (
        df.groupby(["league", "season"], dropna=False)
        .size()
        .reset_index(name="n_matches")
        .sort_values(["league", "season"])
    )
    summary.to_excel(RAW_DIR / "understat_summary_by_league_season.xlsx", index=False)

    schema_report = {
        "n_rows": int(len(df)),
        "n_duplicates_match_key": int(df.duplicated(subset=["date", "league", "home_team", "away_team"]).sum()),
        "missing_rate_top_10": (df.isna().mean().sort_values(ascending=False).head(10).round(4)).to_dict(),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
    }
    with open(RAW_DIR / "understat_schema_report.json", "w", encoding="utf-8") as f:
        json.dump(schema_report, f, indent=2, ensure_ascii=False)

    print("=== Extração concluída ===")
    print(f"Partidas: {len(df):,}")
    print(f"CSV:     {csv_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Excel:   {xlsx_path}")
    print(f"Schema:  {RAW_DIR / 'understat_schema_report.json'}")


if __name__ == "__main__":
    matches = fetch_understat(LEAGUES, SEASONS)
    save_outputs(matches)
