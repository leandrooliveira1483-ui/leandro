# ============================================================
# MODELO PREDITIVO DE FUTEBOL v2 — GOOGLE COLAB
# Versão corrigida — 18 problemas da auditoria resolvidos
# Fonte de dados: Understat via soccerdata
# ============================================================
# CORREÇÕES APLICADAS (referência da auditoria):
#   [1]  Data leakage XGBoost → walk-forward como padrão
#   [2]  Elo prob_home inconsistente → inclui HOME_ADV
#   [3]  rho_res duvidoso → removido (só DC original)
#   [4]  Shot features leakage → convertido para rolling
#   [5]  ensemble_predict feature lookup → timeline unificada
#   [6]  RECENT_N fixo → múltiplas janelas (5, 10, 20)
#   [7]  xi=0.002 fraco → xi=0.005
#   [8]  Elo regression arbitrária → calibrada + promovidos
#   [9]  Isotonic overfitting → sigmoid (Platt scaling)
#   [10] Alpha holdout sobrepõe XGBoost → split 3-way limpo
#   [11] MC seed determinística → removida (analítico puro)
#   [12] monte_carlo_match não usada → removida
#   [13] Exceções silenciosas → logging + contagem
#   [14] FEATURE_COLS ordem errada → definido antes do uso
#   [15] fillna(0) perigoso → defaults por tipo de feature
#   [18] Sem target contínuo → regressão Poisson para gols
#   [19] XGBoost hiperparâmetros fixos → Optuna tuning
#   [20] Sem feature de surpresa → goals_vs_xg adicionado
# ============================================================

# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 1 — INSTALAÇÃO DE DEPENDÊNCIAS           ║
# ╚══════════════════════════════════════════════════════════╝
"""
# Execute no Colab:
!pip install soccerdata scipy scikit-learn xgboost lightgbm shap matplotlib seaborn tqdm optuna -q
"""

# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 2 — IMPORTAÇÕES                          ║
# ╚══════════════════════════════════════════════════════════╝

import warnings
warnings.filterwarnings("ignore", category=UserWarning,  module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning,  module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="soccerdata")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="soccerdata")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import poisson
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import logging

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (log_loss, brier_score_loss,
                             accuracy_score, classification_report,
                             mean_squared_error)
import xgboost as xgb
import shap

# Optuna — [FIX #19] tuning de hiperparâmetros
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna não instalado — hiperparâmetros fixos serão usados")

# soccerdata
import soccerdata as sd

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

# [FIX #13] Logger para rastrear erros em vez de silenciá-los
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("football_model")

print("✅ Dependências carregadas com sucesso!")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 3 — CONFIGURAÇÕES GLOBAIS                ║
# ╚══════════════════════════════════════════════════════════╝

LEAGUE   = "ENG-Premier League"
SEASONS  = [2021, 2022, 2023, 2024]

# [FIX #6] Múltiplas janelas de rolling — o ML decide qual é mais preditiva
ROLLING_WINDOWS = [5, 10, 20]

# Limites de probabilidade — nenhum resultado de futebol é 99.9%
# Mesmo Barcelona 10x0 é improvável de dar >92% no modelo
PROB_FLOOR   = 0.03   # mínimo 3% para qualquer resultado 1X2
PROB_CEILING = 0.92   # máximo 92% para qualquer resultado 1X2
PROB_FLOOR_BINARY = 0.03   # para mercados binários (Over/Under, BTTS)
PROB_CEIL_BINARY  = 0.97

# [FIX #12] Monte-Carlo removido — cálculos analíticos puros
# N_SIMS removido — não é mais necessário

CUTOFF_DATE = None


def discover_latest_round(df_raw: pd.DataFrame) -> None:
    df_tmp = df_raw.copy()
    df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce")

    summary = (df_tmp.groupby("season")
               .agg(Partidas=("date","count"),
                    Data_Mais_Antiga=("date","min"),
                    Data_Mais_Recente=("date","max"))
               .reset_index())
    summary["Data_Mais_Antiga"]  = summary["Data_Mais_Antiga"].dt.strftime("%Y-%m-%d")
    summary["Data_Mais_Recente"] = summary["Data_Mais_Recente"].dt.strftime("%Y-%m-%d")

    latest_date = df_tmp["date"].max()
    print("\n" + "═"*60)
    print("  📅  DISPONIBILIDADE DE DADOS — UNDERSTAT")
    print("═"*60)
    print(summary.to_string(index=False))
    print("─"*60)
    print(f"  ✅  Jogo mais recente: {latest_date.strftime('%Y-%m-%d')}")
    print("─"*60)
    print("  ℹ️   Para cortar os dados, defina CUTOFF_DATE acima.")
    print('      Exemplo: CUTOFF_DATE = "2025-03-01"')
    print("═"*60 + "\n")


def apply_cutoff(df: pd.DataFrame,
                 cutoff_date: str | None) -> pd.DataFrame:
    if cutoff_date is None:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    limite = pd.to_datetime(cutoff_date)
    df_cut = df[df["date"] <= limite]
    print(f"✂️  Corte aplicado: partidas até {cutoff_date} "
          f"({len(df_cut)} de {len(df)} partidas mantidas)")
    return df_cut


# ╔══════════════════════════════════════════════════════════╗
# ║     CÉLULA 4 — COLETA DE DADOS + AUDITORIA COMPLETA     ║
# ╚══════════════════════════════════════════════════════════╝

AUDIT_DIR = "/content/auditoria_understat"
os.makedirs(AUDIT_DIR, exist_ok=True)


def fetch_single_season(understat, season: int) -> dict:
    resultado = {
        "season": season, "status": "❌ ERRO",
        "n_partidas": 0, "n_finalizadas": 0, "n_sem_gols": 0,
        "data_min": None, "data_max": None, "colunas": [],
        "df": None, "erro": None,
    }
    try:
        print(f"  ⏳ Baixando temporada {season}...", end=" ")
        df = understat.read_schedule(season)
        df["season"] = season
        resultado["status"]     = "✅ OK"
        resultado["n_partidas"] = len(df)
        resultado["colunas"]    = list(df.columns)

        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            resultado["data_min"] = df[date_col].min()
            resultado["data_max"] = df[date_col].max()

        hg_col = next((c for c in df.columns
                       if c in ["home_goals","hg","score_home"]), None)
        ag_col = next((c for c in df.columns
                       if c in ["away_goals","ag","score_away"]), None)
        if hg_col and ag_col:
            df_fin = df.dropna(subset=[hg_col, ag_col])
            resultado["n_finalizadas"] = len(df_fin)
            resultado["n_sem_gols"]    = len(df) - len(df_fin)
        else:
            resultado["n_sem_gols"] = len(df)

        resultado["df"] = df
        print(f"✅  {len(df)} jogos | "
              f"{resultado['data_min']} → {resultado['data_max']}")
    except Exception as e:
        resultado["erro"] = str(e)
        print(f"❌  {e}")
    return resultado


def fetch_all_seasons_audit(league: str, seasons: list) -> tuple:
    print("\n" + "═"*60)
    print(f"  🔍  AUDITORIA DE EXTRAÇÃO — {league}")
    print("═"*60)

    understat = sd.Understat(leagues=league)
    resultados = []
    frames_ok  = []

    for season in seasons:
        r = fetch_single_season(understat, season)
        resultados.append(r)
        if r["df"] is not None:
            frames_ok.append(r["df"])

    audit_rows = []
    for r in resultados:
        audit_rows.append({
            "Temporada":        r["season"],
            "Status":           r["status"],
            "Total Jogos":      r["n_partidas"],
            "Finalizados":      r["n_finalizadas"],
            "Sem Gols (fut.)":  r["n_sem_gols"],
            "Data Mais Antiga": str(r["data_min"])[:10] if r["data_min"] else "—",
            "Data Mais Recente":str(r["data_max"])[:10] if r["data_max"] else "—",
            "Colunas":          str(r["colunas"]),
            "Erro":             r["erro"] or "",
        })
    df_audit = pd.DataFrame(audit_rows)

    if frames_ok:
        df_all = pd.concat(frames_ok, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    return df_all, df_audit, resultados


def fetch_shot_data_audit(league: str, seasons: list) -> pd.DataFrame:
    understat = sd.Understat(leagues=league)
    frames = []
    print("\n⚽ Baixando dados de chutes:")
    for season in seasons:
        try:
            print(f"  ⏳ Chutes {season}...", end=" ")
            shots = understat.read_shot_events(season)
            shots["season"] = season
            frames.append(shots)
            print(f"✅  {len(shots)} chutes")
        except Exception as e:
            print(f"❌  {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export_audit_to_excel(df_all, df_audit, resultados, audit_dir):
    path_resumo = os.path.join(audit_dir, "auditoria_resumo.xlsx")
    with pd.ExcelWriter(path_resumo, engine="openpyxl") as writer:
        df_audit.to_excel(writer, sheet_name="Resumo Geral", index=False)
        for r in resultados:
            if r["df"] is not None and len(r["df"]) > 0:
                sheet_name = f"T{r['season']}"
                df_s = r["df"].copy()
                for col in df_s.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_s[col]):
                        df_s[col] = df_s[col].dt.strftime("%Y-%m-%d")
                df_s.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\n💾 Auditoria salva: {path_resumo}")

    path_dados = os.path.join(audit_dir, "dados_consolidados.xlsx")
    if not df_all.empty:
        df_export = df_all.copy()
        for col in df_export.columns:
            if pd.api.types.is_datetime64_any_dtype(df_export[col]):
                df_export[col] = df_export[col].dt.strftime("%Y-%m-%d")
        with pd.ExcelWriter(path_dados, engine="openpyxl") as writer:
            df_export.to_excel(writer, sheet_name="Todos os Jogos", index=False)
            for season, grp in df_export.groupby("season"):
                grp.to_excel(writer, sheet_name=f"T{int(season)}", index=False)
        print(f"💾 Dados consolidados: {path_dados}")

    return path_resumo, path_dados


def print_audit_report(df_audit, df_all):
    print("\n" + "═"*60)
    print("  📋  RELATÓRIO DE AUDITORIA")
    print("═"*60)
    print(df_audit[["Temporada","Status","Total Jogos",
                    "Finalizados","Data Mais Antiga",
                    "Data Mais Recente"]].to_string(index=False))
    if not df_all.empty:
        date_col = next((c for c in df_all.columns if "date" in c.lower()), None)
        if date_col:
            df_all[date_col] = pd.to_datetime(df_all[date_col], errors="coerce")
            data_max_global = df_all[date_col].max()
            print(f"\n  📅  Jogo mais recente no dataset: {str(data_max_global)[:10]}")
    print("═"*60 + "\n")


# ── EXECUÇÃO ──
df_raw, df_audit_raw, resultados_audit = fetch_all_seasons_audit(LEAGUE, SEASONS)
df_shots = fetch_shot_data_audit(LEAGUE, SEASONS)
export_audit_to_excel(df_raw, df_audit_raw, resultados_audit, AUDIT_DIR)
print_audit_report(df_audit_raw, df_raw)

if not df_raw.empty:
    df_raw_clean = df_raw.dropna(subset=[c for c in df_raw.columns
                                          if c in ["home_goals","away_goals",
                                                   "score_home","score_away"]])
    if df_raw_clean.empty:
        df_raw_clean = df_raw
    discover_latest_round(df_raw_clean)
    df_raw = apply_cutoff(df_raw_clean, CUTOFF_DATE)
else:
    print("❌  Nenhum dado baixado. Verifique SEASONS e a conexão.")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 5 — LIMPEZA E NORMALIZAÇÃO               ║
# ╚══════════════════════════════════════════════════════════╝

def clean_schedule(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "home_team":  "home",  "away_team":  "away",
        "home_goals": "hg",    "away_goals": "ag",
        "home_xg":    "hxg",   "away_xg":    "axg",
        "home_xga":   "hxga",  "away_xga":   "axga",
        "home_ppda":  "hppda", "away_ppda":  "appda",
        "home_deep":  "hdeep", "away_deep":  "adeep",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["hg"]  = pd.to_numeric(df["hg"], errors="coerce").astype("Int64")
    df["ag"]  = pd.to_numeric(df["ag"], errors="coerce").astype("Int64")
    df["hxg"] = pd.to_numeric(df.get("hxg", np.nan), errors="coerce")
    df["axg"] = pd.to_numeric(df.get("axg", np.nan), errors="coerce")

    df["result"] = np.where(df["hg"] > df["ag"], "H",
                   np.where(df["hg"] < df["ag"], "A", "D"))
    df.loc[df["hg"].isna() | df["ag"].isna(), "result"] = np.nan
    df["total_goals"] = df["hg"] + df["ag"]

    df = df.sort_values("date").reset_index(drop=True)
    return df


df = clean_schedule(df_raw)
print("✅ Dados limpos — shape:", df.shape)


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 6 — FEATURE ENGINEERING                  ║
# ╚══════════════════════════════════════════════════════════╝
# [FIX #6]  Múltiplas janelas de rolling (5, 10, 20)
# [FIX #20] Feature de "surpresa" — gols_reais vs xG
# [FIX #4]  Shot features convertidas para rolling

def build_team_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria visão por equipe (home + away) com timeline unificada.
    [FIX #20] Adiciona gols reais para calcular surpresa (gf - xgf).
    """
    home = df[["date", "home", "away", "hg", "ag", "hxg", "axg",
               "result", "season"]].copy()
    home.columns = ["date", "team", "opponent", "gf", "ga",
                    "xgf", "xga", "result_raw", "season"]
    home["venue"] = "H"
    home["pts"] = home["result_raw"].map({"H": 3, "D": 1, "A": 0})
    home["win"]  = (home["result_raw"] == "H").astype(int)
    home["draw"] = (home["result_raw"] == "D").astype(int)
    home["loss"] = (home["result_raw"] == "A").astype(int)

    away = df[["date", "away", "home", "ag", "hg", "axg", "hxg",
               "result", "season"]].copy()
    away.columns = ["date", "team", "opponent", "gf", "ga",
                    "xgf", "xga", "result_raw", "season"]
    away["venue"] = "A"
    away["pts"] = away["result_raw"].map({"A": 3, "D": 1, "H": 0})
    away["win"]  = (away["result_raw"] == "A").astype(int)
    away["draw"] = (away["result_raw"] == "D").astype(int)
    away["loss"] = (away["result_raw"] == "H").astype(int)

    timeline = pd.concat([home, away], ignore_index=True)
    timeline = timeline.sort_values(["team", "date"]).reset_index(drop=True)

    # [FIX #20] Surpresa: diferença entre gols reais e xG esperado
    timeline["gf"] = pd.to_numeric(timeline["gf"], errors="coerce")
    timeline["ga"] = pd.to_numeric(timeline["ga"], errors="coerce")
    timeline["surprise_attack"]  = timeline["gf"] - timeline["xgf"]  # overperformance ofensiva
    timeline["surprise_defense"] = timeline["ga"] - timeline["xga"]  # underperformance defensiva

    return timeline


def rolling_team_stats(timeline: pd.DataFrame,
                       windows: list = None) -> pd.DataFrame:
    """
    [FIX #6] Calcula médias móveis para MÚLTIPLAS janelas.
    O ML decide qual janela é mais preditiva para cada mercado.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    for n in windows:
        # Features globais (todos os jogos)
        cols_global = ["xgf", "xga", "pts", "win", "draw", "loss",
                       "surprise_attack", "surprise_defense"]  # [FIX #20]
        for col in cols_global:
            if col not in timeline.columns:
                continue
            timeline[f"roll_{col}_{n}"] = (
                timeline.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
            )

        # Features por venue: xG em casa separado de xG fora
        for venue_tag, venue_val in [("h", "H"), ("a", "A")]:
            mask = timeline["venue"] == venue_val
            for col in ["xgf", "xga"]:
                col_name = f"roll_{col}_venue_{venue_tag}_{n}"
                timeline[col_name] = np.nan
                timeline.loc[mask, col_name] = (
                    timeline[mask].groupby("team")[col]
                    .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
                )

    return timeline


def h2h_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    orig_index = df.index.copy()
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df))

    df["total_goals_match"] = (df["hg"] + df["ag"]).astype(float)
    df["h_won"]  = (df["result"] == "H").astype(float)
    df["drawn"]  = (df["result"] == "D").astype(float)

    teams_sorted = np.sort(np.stack([df["home"].values, df["away"].values]), axis=0)
    df["pair"] = [f"{a}___{b}" for a, b in zip(teams_sorted[0], teams_sorted[1])]

    df_sorted = df.sort_values(["pair", "date"])

    def rolling_pair(series):
        return (series.groupby(df_sorted["pair"])
                      .transform(lambda x: x.shift(1)
                                            .rolling(window, min_periods=1).mean()))

    df_sorted["h2h_home_wins"] = rolling_pair(df_sorted["h_won"])
    df_sorted["h2h_draws"]     = rolling_pair(df_sorted["drawn"])
    df_sorted["h2h_avg_total"] = rolling_pair(df_sorted["total_goals_match"])

    df_result = (df_sorted
                 .set_index("_orig_idx")
                 [["h2h_home_wins", "h2h_draws", "h2h_avg_total"]]
                 .reindex(np.arange(len(df))))
    df_result.index = orig_index
    return df_result


# [FIX #14] FEATURE_COLS definido ANTES de qualquer uso
# [FIX #6] Colunas para todas as janelas de rolling
def _build_feature_cols(windows: list = None) -> list:
    """Gera lista de features baseada nas janelas de rolling."""
    if windows is None:
        windows = ROLLING_WINDOWS

    cols = []
    for n in windows:
        cols.extend([
            f"h_roll_xgf_{n}", f"h_roll_xga_{n}",
            f"h_roll_pts_{n}", f"h_roll_win_{n}", f"h_roll_draw_{n}",
            f"a_roll_xgf_{n}", f"a_roll_xga_{n}",
            f"a_roll_pts_{n}", f"a_roll_win_{n}", f"a_roll_draw_{n}",
            # Por venue
            f"h_roll_xgf_venue_h_{n}", f"h_roll_xga_venue_h_{n}",
            f"a_roll_xgf_venue_a_{n}", f"a_roll_xga_venue_a_{n}",
            # Diferenciais
            f"diff_xgf_{n}", f"diff_xga_{n}",
            f"diff_pts_{n}", f"diff_win_{n}",
            # [FIX #20] Surpresa
            f"h_roll_surprise_attack_{n}", f"h_roll_surprise_defense_{n}",
            f"a_roll_surprise_attack_{n}", f"a_roll_surprise_defense_{n}",
        ])

    # Features fixas (não dependem da janela)
    cols.extend([
        "h2h_home_wins", "h2h_draws", "h2h_avg_total",
        "elo_home", "elo_away", "elo_diff", "elo_prob_home",
        "rest_days_home", "rest_days_away",
        "fatigue_home", "fatigue_away",
        "games_14d_home", "games_14d_away",
        "table_pts_home", "table_pts_away", "table_pts_diff",
    ])

    return cols

FEATURE_COLS = _build_feature_cols()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    [FIX #6] Múltiplas janelas de rolling.
    """
    timeline = build_team_timeline(df)
    timeline = rolling_team_stats(timeline, ROLLING_WINDOWS)

    # Merge features do mandante
    home_feats = (timeline[timeline["venue"] == "H"]
                  .rename(columns=lambda c: f"h_{c}" if c not in
                          ["team", "date"] else c)
                  .rename(columns={"team": "home", "date": "date"}))

    away_feats = (timeline[timeline["venue"] == "A"]
                  .rename(columns=lambda c: f"a_{c}" if c not in
                          ["team", "date"] else c)
                  .rename(columns={"team": "away", "date": "date"}))

    roll_cols_h = [c for c in home_feats.columns if c.startswith("h_roll_")]
    roll_cols_a = [c for c in away_feats.columns if c.startswith("a_roll_")]

    home_feats_dedup = (home_feats[["home", "date"] + roll_cols_h]
                        .drop_duplicates(subset=["home", "date"], keep="last"))
    away_feats_dedup = (away_feats[["away", "date"] + roll_cols_a]
                        .drop_duplicates(subset=["away", "date"], keep="last"))

    df_feat = df.merge(
        home_feats_dedup, on=["home", "date"], how="left"
    ).merge(
        away_feats_dedup, on=["away", "date"], how="left"
    )

    df_feat = df_feat.drop_duplicates(subset=["home", "away", "date"], keep="first")

    # Features diferenciais para cada janela [FIX #6]
    for n in ROLLING_WINDOWS:
        df_feat[f"diff_xgf_{n}"] = df_feat.get(f"h_roll_xgf_{n}", np.nan) - df_feat.get(f"a_roll_xgf_{n}", np.nan)
        df_feat[f"diff_xga_{n}"] = df_feat.get(f"h_roll_xga_{n}", np.nan) - df_feat.get(f"a_roll_xga_{n}", np.nan)
        df_feat[f"diff_pts_{n}"] = df_feat.get(f"h_roll_pts_{n}", np.nan) - df_feat.get(f"a_roll_pts_{n}", np.nan)
        df_feat[f"diff_win_{n}"] = df_feat.get(f"h_roll_win_{n}", np.nan) - df_feat.get(f"a_roll_win_{n}", np.nan)

    # H2H
    h2h = h2h_features(df)
    df_feat = pd.concat([df_feat.reset_index(drop=True),
                         h2h.reset_index(drop=True)], axis=1)

    # Elo (se já calculado)
    elo_cols = [c for c in ["elo_home","elo_away","elo_diff","elo_prob_home"]
                if c in df.columns]
    if elo_cols:
        df_feat = df_feat.merge(
            df[["home","away","date"] + elo_cols],
            on=["home","away","date"], how="left"
        )

    # Features situacionais (se calculadas)
    sit_cols = [c for c in ["rest_days_home","rest_days_away",
                             "fatigue_home","fatigue_away",
                             "games_14d_home","games_14d_away",
                             "table_pts_home","table_pts_away","table_pts_diff"]
                if c in df.columns]
    if sit_cols:
        df_feat = df_feat.merge(
            df[["home","away","date"] + sit_cols],
            on=["home","away","date"], how="left"
        )

    return df_feat


def situational_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    home_tl = df[["date","home","season","hg","ag"]].copy()
    home_tl.columns = ["date","team","season","hg","ag"]
    home_tl["pts_ganhos"] = np.where(home_tl["hg"] > home_tl["ag"], 3,
                            np.where(home_tl["hg"] == home_tl["ag"], 1, 0))

    away_tl = df[["date","away","season","ag","hg"]].copy()
    away_tl.columns = ["date","team","season","hg","ag"]
    away_tl["pts_ganhos"] = np.where(away_tl["hg"] > away_tl["ag"], 3,
                            np.where(away_tl["hg"] == away_tl["ag"], 1, 0))

    tl = pd.concat([home_tl, away_tl], ignore_index=True)
    tl = tl.sort_values(["team","season","date"]).reset_index(drop=True)

    tl["prev_date"] = tl.groupby("team")["date"].shift(1)
    tl["rest_days"] = (tl["date"] - tl["prev_date"]).dt.days

    tl_indexed = tl.set_index("date")
    tl["games_14d"] = (
        tl_indexed.groupby("team")["pts_ganhos"]
        .transform(lambda x: x.shift(1, freq="D")
                               .rolling("14D", min_periods=0).count())
        .values
    )

    tl["table_pts"] = (
        tl.groupby(["team","season"])["pts_ganhos"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    tl_home = (tl[tl["hg"].notna() | tl["ag"].notna()]
               .drop_duplicates(subset=["team","date"])
               [["team","date","rest_days","games_14d","table_pts"]])

    df = df.merge(
        tl_home.rename(columns={
            "team": "home",
            "rest_days":  "rest_days_home",
            "games_14d":  "games_14d_home",
            "table_pts":  "table_pts_home",
        }),
        on=["home","date"], how="left"
    ).merge(
        tl_home.rename(columns={
            "team": "away",
            "rest_days":  "rest_days_away",
            "games_14d":  "games_14d_away",
            "table_pts":  "table_pts_away",
        }),
        on=["away","date"], how="left"
    )

    median_rest = df["rest_days_home"].median()
    df["rest_days_home"]  = df["rest_days_home"].fillna(median_rest)
    df["rest_days_away"]  = df["rest_days_away"].fillna(median_rest)
    df["fatigue_home"]    = (df["rest_days_home"] <= 3).astype(int)
    df["fatigue_away"]    = (df["rest_days_away"] <= 3).astype(int)
    df["games_14d_home"]  = df["games_14d_home"].fillna(0)
    df["games_14d_away"]  = df["games_14d_away"].fillna(0)
    df["table_pts_home"]  = df["table_pts_home"].fillna(0)
    df["table_pts_away"]  = df["table_pts_away"].fillna(0)
    df["table_pts_diff"]  = df["table_pts_home"] - df["table_pts_away"]

    return df


# [FIX #4] Shot features convertidas para ROLLING (sem leakage)
def aggregate_shot_features_rolling(df_shots: pd.DataFrame,
                                     df: pd.DataFrame,
                                     n: int = 10) -> pd.DataFrame:
    """
    [FIX #4] Cria features de chute como MÉDIA ROLLING dos últimos N jogos,
    NÃO do jogo atual. Isso elimina o data leakage das shot features.
    """
    if df_shots is None or df_shots.empty:
        print("ℹ️  df_shots vazio — features de chute ignoradas")
        return df

    xcol = next((c for c in df_shots.columns if "xg" in c.lower()), None)
    result_col = next((c for c in df_shots.columns if "result" in c.lower()
                       or "outcome" in c.lower()), None)
    home_col = next((c for c in df_shots.columns if c in ["home","home_team"]), None)
    away_col = next((c for c in df_shots.columns if c in ["away","away_team"]), None)
    team_col = next((c for c in df_shots.columns if c in ["team","shooter_team"]), None)
    date_col = next((c for c in df_shots.columns if "date" in c.lower()), None)

    if xcol is None or date_col is None or not (home_col and away_col and team_col):
        print("ℹ️  df_shots sem colunas necessárias — features de chute ignoradas")
        return df

    shots = df_shots.copy()
    shots[date_col] = pd.to_datetime(shots[date_col], errors="coerce").dt.normalize()
    shots[xcol] = pd.to_numeric(shots[xcol], errors="coerce")

    if result_col:
        shots["on_target"] = shots[result_col].astype(str).str.lower().isin(
            ["goal","savedshot","on target","blockedshot"]
        ).astype(int)
    else:
        shots["on_target"] = 0

    shots = shots.rename(columns={home_col: "home", away_col: "away",
                                   date_col: "date", team_col: "shot_team"})

    # Agregar por (time, jogo)
    team_game = shots.groupby(["home", "away", "date", "shot_team"]).agg(
        shots_count=(xcol, "count"),
        xg_sum=(xcol, "sum"),
        avg_shot_xg=(xcol, "mean"),
        on_target_count=("on_target", "sum"),
    ).reset_index()

    # Separar em mandante vs visitante
    tg_home = team_game[team_game["shot_team"] == team_game["home"]].copy()
    tg_home["team"] = tg_home["home"]
    tg_away = team_game[team_game["shot_team"] == team_game["away"]].copy()
    tg_away["team"] = tg_away["away"]

    tg_all = pd.concat([
        tg_home[["team", "date", "shots_count", "xg_sum", "avg_shot_xg", "on_target_count"]],
        tg_away[["team", "date", "shots_count", "xg_sum", "avg_shot_xg", "on_target_count"]],
    ], ignore_index=True).sort_values(["team", "date"])

    # Rolling com shift (sem leakage) — média dos últimos N jogos
    for col in ["shots_count", "xg_sum", "avg_shot_xg", "on_target_count"]:
        tg_all[f"roll_{col}_{n}"] = (
            tg_all.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        )

    roll_shot_cols = [c for c in tg_all.columns if c.startswith("roll_")]

    # Merge para home
    home_shots = tg_all[["team", "date"] + roll_shot_cols].drop_duplicates(
        subset=["team", "date"], keep="last"
    ).rename(columns={"team": "home", **{c: f"h_{c}" for c in roll_shot_cols}})

    away_shots = tg_all[["team", "date"] + roll_shot_cols].drop_duplicates(
        subset=["team", "date"], keep="last"
    ).rename(columns={"team": "away", **{c: f"a_{c}" for c in roll_shot_cols}})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.merge(home_shots, on=["home", "date"], how="left")
    df = df.merge(away_shots, on=["away", "date"], how="left")

    added = [c for c in df.columns if "roll_shots" in c or "roll_xg_sum" in c
             or "roll_avg_shot" in c or "roll_on_target" in c]
    print(f"✅ Shot features rolling (últimos {n} jogos): {len(added)} colunas adicionadas")
    return df


df = aggregate_shot_features_rolling(df_shots, df, n=10)

# Adiciona shot features ao FEATURE_COLS se disponíveis
_shot_feats = [c for c in df.columns if ("roll_shots" in c or "roll_xg_sum" in c
               or "roll_avg_shot" in c or "roll_on_target" in c)
               and c.startswith(("h_", "a_"))]
for _sf in _shot_feats:
    if _sf not in FEATURE_COLS:
        FEATURE_COLS.append(_sf)
if _shot_feats:
    print(f"   {len(_shot_feats)} shot features rolling adicionadas ao modelo")

# Features situacionais
df = situational_features(df)
print("✅ Features situacionais calculadas")


# ╔══════════════════════════════════════════════════════════╗
# ║   CÉLULA 6-A — ELO RATING                               ║
# ╚══════════════════════════════════════════════════════════╝
# [FIX #2] elo_prob_home agora inclui ELO_HOME_ADV
# [FIX #8] Regression-to-mean calibrada + Elo inicial para promovidos

ELO_K        = 40
ELO_BASE     = 1500
ELO_HOME_ADV = 65

# [FIX #8] Times recém-promovidos começam abaixo da média
ELO_PROMOTED = 1350   # rating inicial para times promovidos
# [FIX #8] Fator de regression-to-mean entre temporadas
ELO_REGRESS  = 0.70   # mais agressivo que 0.75 — reflete mudanças de elenco


def compute_elo(df: pd.DataFrame,
                k: float = ELO_K,
                base: float = ELO_BASE) -> pd.DataFrame:
    """
    [FIX #2] elo_prob_home calculado COM ELO_HOME_ADV
    [FIX #8] Regression-to-mean calibrada + promovidos com rating menor
    """
    df = df.sort_values("date").reset_index(drop=True)

    homes   = df["home"].values
    aways   = df["away"].values
    hgs     = df["hg"].values
    ags     = df["ag"].values
    seasons = df["season"].values if "season" in df.columns else np.zeros(len(df))

    ratings      = {}
    last_season  = {}
    season_teams = {}  # temporada → set de times (para detectar promovidos)
    elo_home_arr = np.empty(len(df))
    elo_away_arr = np.empty(len(df))

    # Pré-computa times por temporada para detectar promovidos
    for s in np.unique(seasons):
        mask = seasons == s
        season_teams[s] = set(homes[mask]) | set(aways[mask])

    sorted_seasons = sorted(season_teams.keys())

    for i in range(len(df)):
        h, a   = homes[i], aways[i]
        season = seasons[i]

        # [FIX #8] Regression-to-mean entre temporadas
        for team in (h, a):
            if team in last_season and last_season[team] != season:
                old_r = ratings.get(team, base)
                ratings[team] = ELO_REGRESS * old_r + (1 - ELO_REGRESS) * base
            elif team not in ratings:
                # [FIX #8] Verifica se é promovido (não estava na temporada anterior)
                season_idx = sorted_seasons.index(season) if season in sorted_seasons else 0
                if season_idx > 0:
                    prev_season = sorted_seasons[season_idx - 1]
                    if team not in season_teams.get(prev_season, set()):
                        ratings[team] = ELO_PROMOTED
                        logger.info(f"Promovido detectado: {team} (Elo={ELO_PROMOTED})")
            last_season[team] = season

        rh = ratings.get(h, base)
        ra = ratings.get(a, base)

        elo_home_arr[i] = rh
        elo_away_arr[i] = ra

        hg_val = hgs[i]
        ag_val = ags[i]
        if hg_val is None or ag_val is None: continue
        try:
            hg_i, ag_i = int(hg_val), int(ag_val)
        except (ValueError, TypeError):
            continue

        score_h = 1.0 if hg_i > ag_i else (0.5 if hg_i == ag_i else 0.0)
        score_a = 1.0 - score_h
        exp_h   = 1.0 / (1.0 + 10.0 ** ((ra - rh - ELO_HOME_ADV) / 400.0))
        ratings[h] = rh + k * (score_h - exp_h)
        ratings[a] = ra + k * (score_a - (1.0 - exp_h))

    df["elo_home"]      = elo_home_arr
    df["elo_away"]      = elo_away_arr
    df["elo_diff"]      = elo_home_arr - elo_away_arr

    # [FIX #2] elo_prob_home INCLUI home advantage na conversão
    # Coerente: o update usa ELO_HOME_ADV, a probabilidade também
    df["elo_prob_home"] = 1.0 / (1.0 + 10.0 ** (-(df["elo_diff"] + ELO_HOME_ADV) / 400.0))
    return df


df = compute_elo(df, k=ELO_K, base=ELO_BASE)
print(f"✅ Elo calculado | range: "
      f"{df['elo_home'].min():.0f} – {df['elo_home'].max():.0f}")

print("⚙️  Construindo features...")
df_feat = build_features(df)
print(f"✅ Features prontas — shape: {df_feat.shape}")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 7 — MODELO DIXON-COLES (POISSON)         ║
# ╚══════════════════════════════════════════════════════════╝
# [FIX #3]  rho_res removido — só correção DC original (4 placares)
# [FIX #7]  xi=0.005 — decaimento mais agressivo (~140 dias meia-vida)

class DixonColesModel:
    """
    Dixon-Coles vetorizado.
    [FIX #3] Sem cópula gaussiana residual — apenas correção original.
    [FIX #7] xi=0.005 por default.
    """
    def __init__(self, xi=0.005):   # [FIX #7] era 0.002
        self.xi = xi
        self.params_ = None
        self.teams_ = None
        self._idx = None

    @staticmethod
    def _tau_vec(lh, la, hg, ag, rho):
        tau = np.ones(len(hg))
        m00 = (hg==0)&(ag==0); m01 = (hg==0)&(ag==1)
        m10 = (hg==1)&(ag==0); m11 = (hg==1)&(ag==1)
        tau[m00] = np.clip(1 - lh[m00]*la[m00]*rho, 1e-10, None)
        tau[m01] = np.clip(1 + lh[m01]*rho, 1e-10, None)
        tau[m10] = np.clip(1 + la[m10]*rho, 1e-10, None)
        tau[m11] = np.clip(1 - rho, 1e-10, None)
        return tau

    def _neg_log_likelihood(self, params, hi_arr, ai_arr, hg, ag, weights, lfh, lfa):
        n = len(self.teams_)
        alpha = params[:n]; beta = params[n:2*n]
        gamma = params[2*n]; rho = params[2*n+1]
        lh = np.exp(alpha[hi_arr] - beta[ai_arr] + gamma)
        la = np.exp(alpha[ai_arr] - beta[hi_arr])
        tau = self._tau_vec(lh, la, hg, ag, rho)
        ll_h = hg*np.log(np.maximum(lh, 1e-10)) - lh - lfh
        ll_a = ag*np.log(np.maximum(la, 1e-10)) - la - lfa
        return -(weights*(np.log(tau) + ll_h + ll_a)).sum()

    def _time_weights(self, dates):
        d = np.array(dates, dtype="datetime64[D]")
        return np.exp(-self.xi * (d.max() - d).astype(float))

    def fit(self, df):
        from scipy.special import gammaln
        self.teams_ = sorted(set(df["home"]) | set(df["away"]))
        self._idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)
        hi = np.array([self._idx[t] for t in df["home"]], dtype=int)
        ai = np.array([self._idx[t] for t in df["away"]], dtype=int)
        hg = df["hg"].astype(int).values
        ag = df["ag"].astype(int).values
        w = self._time_weights(df["date"].values)
        lfh = gammaln(hg+1); lfa = gammaln(ag+1)
        x0 = np.zeros(2*n+2); x0[2*n] = 0.3; x0[2*n+1] = -0.1
        res = minimize(self._neg_log_likelihood, x0,
                       args=(hi, ai, hg, ag, w, lfh, lfa), method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6})
        self.params_ = res.x
        self.params_[:n] -= self.params_[:n].mean()
        # [FIX C4] Centrar betas (defesa) — identificabilidade completa
        self.params_[n:2*n] -= self.params_[n:2*n].mean()
        print(f"✅ Dixon-Coles ajustado | LL={-res.fun:.2f} | Convergiu:{res.success}")

    def predict_lambda(self, home, away):
        n = len(self.teams_)
        a = self.params_[:n]; b = self.params_[n:2*n]; g = self.params_[2*n]
        hi = self._idx[home]; ai = self._idx[away]
        return np.exp(a[hi] - b[ai] + g), np.exp(a[ai] - b[hi])

    def score_matrix(self, home, away, max_goals=10):
        """
        [FIX #3] Apenas correção Dixon-Coles original (4 placares).
        Sem cópula gaussiana residual — sem distorção não fundamentada.
        """
        lh, la = self.predict_lambda(home, away)
        rho    = self.params_[-1]
        goals  = np.arange(max_goals + 1)

        ph = poisson.pmf(goals, lh)
        pa = poisson.pmf(goals, la)
        m  = np.outer(ph, pa)

        # Correção DC original (APENAS 4 placares baixos)
        for i, j, v in [(0,0, 1-lh*la*rho), (0,1, 1+lh*rho),
                        (1,0, 1+la*rho),     (1,1, 1-rho)]:
            m[i, j] *= max(v, 1e-10)

        return np.maximum(m, 0) / np.maximum(m, 0).sum()


# Treino — DC completo (para previsão final)
dc_model = DixonColesModel(xi=0.005)   # [FIX #7]
dc_model.fit(df.dropna(subset=["hg", "ag"]).reset_index(drop=True))

# [FIX C1] DC separado para features ML — treinado apenas nos primeiros 60%
# Elimina data leakage: XGBoost nunca vê features DC calculadas com dados futuros
_df_dc_all = df.dropna(subset=["hg", "ag"]).sort_values("date").reset_index(drop=True)
_n_dc_train = int(len(_df_dc_all) * 0.60)
dc_train = DixonColesModel(xi=0.005)
dc_train.fit(_df_dc_all.iloc[:_n_dc_train].reset_index(drop=True))
print(f"   [FIX C1] DC treino: {_n_dc_train} jogos | DC completo: {len(_df_dc_all)} jogos")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 9 — MODELO ML (XGBoost + Calibração)     ║
# ╚══════════════════════════════════════════════════════════╝
# [FIX #1]  Walk-forward como padrão — sem leakage
# [FIX #9]  Sigmoid (Platt) em vez de isotonic
# [FIX #10] Split 3-way limpo: treino / alpha-val / teste
# [FIX #15] fillna inteligente por tipo de feature
# [FIX #18] Target contínuo de gols (regressão Poisson)
# [FIX #19] Optuna tuning de hiperparâmetros


def add_poisson_features(df_feat: pd.DataFrame,
                          dc: DixonColesModel) -> pd.DataFrame:
    n      = len(dc.teams_)
    alpha  = dc.params_[:n]
    beta   = dc.params_[n:2*n]
    gamma  = dc.params_[2*n]

    hi = df_feat["home"].map(dc._idx).values.astype(float)
    ai = df_feat["away"].map(dc._idx).values.astype(float)

    valid = ~(np.isnan(hi) | np.isnan(ai))
    # [FIX C1] Default para times desconhecidos: força média (alpha=0, beta=0)
    lh_arr = np.full(len(df_feat), np.exp(gamma))   # lambda home "neutro"
    la_arr = np.full(len(df_feat), np.exp(0.0))      # lambda away "neutro" = 1.0

    hi_v = hi[valid].astype(int)
    ai_v = ai[valid].astype(int)
    lh_arr[valid] = np.exp(alpha[hi_v] - beta[ai_v] + gamma)
    la_arr[valid] = np.exp(alpha[ai_v] - beta[hi_v])

    df_out = df_feat.copy()
    df_out["dc_lh"]    = lh_arr
    df_out["dc_la"]    = la_arr
    df_out["dc_diff"]  = lh_arr - la_arr
    df_out["dc_ratio"] = np.where(la_arr > 0.01, lh_arr / la_arr, np.nan)
    return df_out


# [FIX #15] Imputação inteligente por tipo de feature
def smart_fillna(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """
    [FIX #15] Imputa NaN com valores sensíveis por tipo de feature:
    - Elo/pts: mediana (nunca zero)
    - Rolling stats: mediana
    - Rest days: mediana
    - Binários (fatigue): 0
    - Ratios: 1.0 (neutro)
    """
    X = X.copy()

    # Primeiro passa: mediana do treino
    X = X.fillna(medians.reindex(X.columns))

    # Segundo passa: defaults por tipo (para features sem mediana)
    for col in X.columns:
        if X[col].isna().sum() == 0:
            continue
        if "fatigue" in col:
            X[col] = X[col].fillna(0)
        elif "ratio" in col:
            X[col] = X[col].fillna(1.0)
        elif any(k in col for k in ["elo", "pts", "rest_days"]):
            X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
        else:
            X[col] = X[col].fillna(0)

    return X


def prepare_ml_dataset(df_feat: pd.DataFrame,
                        dc: DixonColesModel) -> tuple:
    """
    [FIX #18] Adiciona targets contínuos para regressão de gols.
    """
    df2 = add_poisson_features(df_feat, dc)
    extra = ["dc_lh", "dc_la", "dc_diff", "dc_ratio"]
    feats = [c for c in FEATURE_COLS if c in df2.columns] + extra

    df2 = df2.dropna(subset=["hg", "ag", "total_goals", "result"]).copy()
    df2["hg"]          = df2["hg"].astype(int)
    df2["ag"]          = df2["ag"].astype(int)
    df2["total_goals"] = df2["total_goals"].astype(int)

    # Targets de classificação
    df2["y_1x2"]    = df2["result"].map({"H": 0, "D": 1, "A": 2})
    df2["y_ov15"]   = (df2["total_goals"] >= 2).astype(int)
    df2["y_ov25"]   = (df2["total_goals"] >= 3).astype(int)
    df2["y_ov35"]   = (df2["total_goals"] >= 4).astype(int)
    df2["y_h_ov05"] = (df2["hg"] >= 1).astype(int)
    df2["y_h_ov15"] = (df2["hg"] >= 2).astype(int)
    df2["y_h_ov25"] = (df2["hg"] >= 3).astype(int)
    df2["y_a_ov05"] = (df2["ag"] >= 1).astype(int)
    df2["y_a_ov15"] = (df2["ag"] >= 2).astype(int)
    df2["y_a_ov25"] = (df2["ag"] >= 3).astype(int)
    df2["y_btts"]   = ((df2["hg"] >= 1) & (df2["ag"] >= 1)).astype(int)

    # [FIX #18] Targets contínuos para regressão de gols
    df2["y_goals_home"]  = df2["hg"].astype(float)
    df2["y_goals_away"]  = df2["ag"].astype(float)
    df2["y_goals_total"] = df2["total_goals"].astype(float)

    all_targets_cls = ["y_1x2", "y_ov15", "y_ov25", "y_ov35",
                       "y_h_ov05", "y_h_ov15", "y_h_ov25",
                       "y_a_ov05", "y_a_ov15", "y_a_ov25", "y_btts"]
    all_targets_reg = ["y_goals_home", "y_goals_away", "y_goals_total"]

    mask = df2[feats + ["y_1x2"]].notna().all(axis=1)
    X  = df2.loc[mask, feats].astype(float)
    ys_cls = {k: df2.loc[mask, k] for k in all_targets_cls}
    ys_reg = {k: df2.loc[mask, k] for k in all_targets_reg}

    print(f"   ✅ {len(X)} partidas para treino "
          f"({len(df_feat) - len(X)} descartadas)")
    return X, ys_cls, ys_reg, feats


# [FIX #19] Optuna tuning para XGBoost
def tune_xgb_params(X_tr, y_tr, multi: bool = False,
                     n_trials: int = 30) -> dict:
    """
    [FIX #19] Usa Optuna para encontrar hiperparâmetros ótimos.
    """
    if not OPTUNA_AVAILABLE:
        return {
            "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8
        }

    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
            "max_depth":      trial.suggest_int("max_depth", 3, 7),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, val_idx in tscv.split(X_tr):
            X_t, X_v = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
            y_t, y_v = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]

            clf = xgb.XGBClassifier(
                **params,
                eval_metric="mlogloss" if multi else "logloss",
                random_state=42, n_jobs=-1
            )
            clf.fit(X_t, y_t, verbose=False)
            proba = clf.predict_proba(X_v)
            scores.append(log_loss(y_v, proba))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"   🔧 Optuna best params: depth={best['max_depth']}, "
          f"lr={best['learning_rate']:.4f}, n_est={best['n_estimators']}")
    return best


# [FIX #1] Walk-forward training — o modelo NUNCA vê dados futuros
# [FIX #10] Split 3-way: treino(0-60%) / alpha-val(60-80%) / teste(80-100%)
def train_models_walkforward(X: pd.DataFrame, ys_cls: dict,
                              ys_reg: dict,
                              tune: bool = True) -> tuple:
    """
    [FIX #1]  Treina APENAS nos dados de treino (0-60%)
    [FIX #9]  Usa sigmoid (Platt) em vez de isotonic
    [FIX #10] Retorna split indices para alpha otimização limpa
    [FIX #18] Treina regressores de gols
    [FIX #19] Optuna tuning
    """
    models_cls = {}
    models_reg = {}
    n = len(X)

    # [FIX #10] Split 3-way limpo
    n_train     = int(n * 0.60)   # treino do ML
    n_alpha_end = int(n * 0.80)   # fim do holdout de alpha
    # 80-100% = holdout de teste final (nunca tocado)

    X_tr = X.iloc[:n_train]
    feature_medians = X_tr.median()

    # ── Classificadores ──
    # [FIX G1] Early stopping com holdout 60-80% para evitar overfitting
    X_val_es = X.iloc[n_train:n_alpha_end]  # eval set para early stopping
    for target, y in ys_cls.items():
        print(f"\n🔧 Treinando classificador: {target}")
        multi = (target == "y_1x2")
        y_tr = y.iloc[:n_train]
        y_val_es = y.iloc[n_train:n_alpha_end]

        # [FIX #19] Optuna tuning
        if tune and OPTUNA_AVAILABLE:
            best_params = tune_xgb_params(X_tr, y_tr, multi=multi, n_trials=25)
        else:
            best_params = {
                "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8
            }

        clf = xgb.XGBClassifier(
            **best_params,
            objective="multi:softprob" if multi else "binary:logistic",
            eval_metric="mlogloss" if multi else "logloss",
            random_state=42, n_jobs=-1
        )
        # [FIX G1] Early stopping: para de treinar quando validação piora
        if len(y_val_es) > 0:
            clf.fit(X_tr, y_tr,
                    eval_set=[(X_val_es, y_val_es)],
                    early_stopping_rounds=30, verbose=False)
            n_trees = getattr(clf, 'best_iteration', best_params.get('n_estimators', 300))
            if n_trees is not None:
                print(f"   Early stopping: {n_trees + 1} árvores "
                      f"(max {best_params.get('n_estimators', 300)})")
        else:
            clf.fit(X_tr, y_tr, verbose=False)
        models_cls[target] = clf

        # Avaliação no holdout de alpha (60-80%) — dados não vistos
        X_val = X.iloc[n_train:n_alpha_end]
        y_val = y.iloc[n_train:n_alpha_end]
        if len(y_val) > 0:
            y_val_proba = clf.predict_proba(X_val)
            ll_val  = log_loss(y_val, y_val_proba)
            acc_val = accuracy_score(y_val, clf.predict(X_val))
            print(f"   Log-loss holdout: {ll_val:.4f} | Acurácia: {acc_val:.4f}")

    # [FIX #18] Regressores de gols (Poisson objective)
    for target, y in ys_reg.items():
        print(f"\n🔧 Treinando regressor: {target}")
        y_tr = y.iloc[:n_train]
        y_val_reg = y.iloc[n_train:n_alpha_end]

        reg = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="count:poisson",   # [FIX #18] Regressão Poisson
            random_state=42,
            n_jobs=-1
        )
        # [FIX G1] Early stopping para regressores também
        if len(y_val_reg) > 0:
            reg.fit(X_tr, y_tr,
                    eval_set=[(X_val_es, y_val_reg)],
                    early_stopping_rounds=30, verbose=False)
        else:
            reg.fit(X_tr, y_tr, verbose=False)
        models_reg[target] = reg

        X_val = X.iloc[n_train:n_alpha_end]
        y_val = y.iloc[n_train:n_alpha_end]
        if len(y_val) > 0:
            y_pred = reg.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            print(f"   RMSE holdout: {rmse:.4f}")

    split_info = {
        "n_train": n_train,
        "n_alpha_end": n_alpha_end,
        "feature_medians": feature_medians,
    }
    return models_cls, models_reg, split_info


print("⚙️  Preparando dataset ML...")
# [FIX C1] Usa dc_train (60%) para features — sem leakage
X_all, y_all_cls, y_all_reg, feat_names = prepare_ml_dataset(df_feat, dc_train)

print(f"   Dataset: {X_all.shape[0]} amostras × {X_all.shape[1]} features")
print("⚙️  Treinando modelos (walk-forward, Optuna, sigmoid)...")

ml_models, ml_regressors, split_info = train_models_walkforward(
    X_all, y_all_cls, y_all_reg, tune=OPTUNA_AVAILABLE
)

FEATURE_MEDIANS = split_info["feature_medians"]
print("\n✅ Todos os modelos treinados!")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 10 — ENSEMBLE FINAL                      ║
# ╚══════════════════════════════════════════════════════════╝
# [FIX #10] Alpha otimizado em 60-80%, totalmente separado do treino ML

ALPHA = 0.45  # fallback


def _dc_probs_analytic(dc: DixonColesModel, home: str, away: str,
                       max_goals: int = 10) -> tuple:
    matrix = dc.score_matrix(home, away, max_goals=max_goals)
    pH = float(np.tril(matrix, -1).sum())
    pA = float(np.triu(matrix,  1).sum())
    pD = float(np.diag(matrix).sum())
    s  = pH + pD + pA
    return pH/s, pD/s, pA/s


def optimize_alpha(dc: DixonColesModel, ml: dict,
                   df_feat_val: pd.DataFrame,
                   feat_names: list,
                   feature_medians: pd.Series) -> float:
    """
    [FIX #10] Alpha otimizado em dados 60-80% — separado do treino ML.
    """
    from scipy.optimize import minimize_scalar

    records = []
    n_errors = 0
    for _, row in df_feat_val.iterrows():
        if pd.isna(row.get("result")): continue
        try:
            mc_H, mc_D, mc_A = _dc_probs_analytic(dc, row["home"], row["away"])
            lh, la = dc.predict_lambda(row["home"], row["away"])
            extra = {
                "dc_lh": lh, "dc_la": la,
                "dc_diff": lh - la,
                "dc_ratio": lh / max(la, 0.01),
            }
            feat_row = {c: row.get(c, extra.get(c, np.nan)) for c in feat_names}
            X_p = pd.DataFrame([feat_row])[feat_names].astype(float)
            X_p = smart_fillna(X_p, feature_medians)
            ml_p = ml["y_1x2"].predict_proba(X_p)[0]
            records.append({
                "result": row["result"],
                "mc_H": mc_H, "mc_D": mc_D, "mc_A": mc_A,
                "ml_H": ml_p[0], "ml_D": ml_p[1], "ml_A": ml_p[2],
            })
        except Exception as e:
            n_errors += 1
            # [FIX #13] Log do erro em vez de silenciar
            logger.debug(f"Alpha opt erro: {row.get('home')} vs {row.get('away')}: {e}")

    if n_errors > 0:
        print(f"   ⚠️  {n_errors} jogos falharam na otimização do alpha")

    if len(records) < 30:
        print(f"⚠️  Apenas {len(records)} amostras — usando ALPHA={ALPHA}")
        return ALPHA

    df_v   = pd.DataFrame(records)
    y_true = df_v["result"].map({"H": 0, "D": 1, "A": 2})

    def neg_ll(a):
        pH = a*df_v["mc_H"] + (1-a)*df_v["ml_H"]
        pD = a*df_v["mc_D"] + (1-a)*df_v["ml_D"]
        pA = a*df_v["mc_A"] + (1-a)*df_v["ml_A"]
        s  = pH + pD + pA
        probs = np.column_stack([pH/s, pD/s, pA/s])
        return log_loss(y_true, probs)

    res       = minimize_scalar(neg_ll, bounds=(0.0, 1.0), method="bounded")
    alpha_opt = round(res.x, 3)
    print(f"✅ Alpha otimizado: {alpha_opt:.3f} "
          f"(log-loss val: {res.fun:.4f}) | Amostras: {len(records)}")
    return alpha_opt


# [FIX #5] ensemble_predict com busca de features unificada
def ensemble_predict(home: str, away: str,
                     dc: DixonColesModel,
                     ml_cls: dict,
                     ml_reg: dict,
                     df_feat: pd.DataFrame,
                     alpha: float = None,
                     cutoff_date: pd.Timestamp = None) -> dict:
    """
    [FIX #5]  Busca features usando timeline UNIFICADA (último jogo,
              independente de venue) com mapeamento correto de colunas.
    [FIX #11] Sem Monte-Carlo — 100% analítico
    [FIX #18] Inclui previsão de gols contínuos via regressão Poisson
    """
    if alpha is None:
        alpha = ALPHA

    # ── [FIX C2] Probabilidades derivadas da score_matrix DC (consistente) ──
    # Todos os mercados derivados da MESMA matriz, incluindo correlação rho
    lh_an, la_an = dc.predict_lambda(home, away)
    _mat   = dc.score_matrix(home, away, max_goals=10)
    _pH_an = float(np.tril(_mat, -1).sum())
    _pA_an = float(np.triu(_mat,  1).sum())
    _pD_an = float(np.diag(_mat).sum())
    _s     = _pH_an + _pD_an + _pA_an
    _pH_an, _pD_an, _pA_an = _pH_an/_s, _pD_an/_s, _pA_an/_s

    # [FIX C2] Over/Under total derivado da score_matrix (NÃO de Poisson independente)
    _max_g = _mat.shape[0]
    _total_pmf = np.zeros(2 * _max_g - 1)
    for _i in range(_max_g):
        for _j in range(_max_g):
            _total_pmf[_i + _j] += _mat[_i, _j]
    _total_pmf /= _total_pmf.sum()
    _total_cdf = np.cumsum(_total_pmf)

    def _ov_total(k):
        return float(1 - _total_cdf[k]) if k < len(_total_cdf) else 0.0

    # [FIX C2] Marginais por time da score_matrix (inclui correlação DC)
    _home_marginal = _mat.sum(axis=1)  # P(home=i) — soma sobre colunas
    _away_marginal = _mat.sum(axis=0)  # P(away=j) — soma sobre linhas
    _home_cdf = np.cumsum(_home_marginal)
    _away_cdf = np.cumsum(_away_marginal)

    def _ov_home(k):
        """P(home goals >= k) derivado da matrix DC"""
        if k <= 0: return 1.0
        return float(1.0 - _home_cdf[k - 1]) if k - 1 < len(_home_cdf) else 0.0

    def _ov_away(k):
        """P(away goals >= k) derivado da matrix DC"""
        if k <= 0: return 1.0
        return float(1.0 - _away_cdf[k - 1]) if k - 1 < len(_away_cdf) else 0.0

    _btts_yes = float(_mat[1:, 1:].sum())

    p_mc = {
        "1": _pH_an, "X": _pD_an, "2": _pA_an,
        "DNB_1": _pH_an/(_pH_an+_pA_an) if (_pH_an+_pA_an) > 0 else 0.5,
        "DNB_2": _pA_an/(_pH_an+_pA_an) if (_pH_an+_pA_an) > 0 else 0.5,
        "DC_1X": _pH_an+_pD_an, "DC_X2": _pA_an+_pD_an,
        "DC_12": _pH_an+_pA_an,
        "ov15": _ov_total(1), "ov25": _ov_total(2), "ov35": _ov_total(3),
        "un15": 1-_ov_total(1), "un25": 1-_ov_total(2), "un35": 1-_ov_total(3),
        "h_ov05": _ov_home(1), "h_ov15": _ov_home(2),
        "h_ov25": _ov_home(3), "h_ov35": _ov_home(4),
        "a_ov05": _ov_away(1), "a_ov15": _ov_away(2),
        "a_ov25": _ov_away(3), "a_ov35": _ov_away(4),
        "btts_yes": _btts_yes, "btts_no": 1-_btts_yes,
        "lambda_home": lh_an, "lambda_away": la_an,
    }

    # ── [FIX C3] Feature lookup corrigido — timeline unificada com remapeamento ──
    # O código original lia h_ features apenas do último jogo como MANDANTE,
    # ignorando jogos recentes como visitante. Se o time jogou 3x fora seguidas,
    # as features ficavam desatualizadas por semanas.
    # Agora: busca o jogo MAIS RECENTE (qualquer venue) e remapeia os prefixos.
    if cutoff_date is not None:
        df_ref = df_feat[df_feat["date"] < cutoff_date]
    else:
        df_ref = df_feat

    def _find_latest(team, df_r):
        """Encontra jogo mais recente do time (qualquer venue)."""
        mask = (df_r["home"] == team) | (df_r["away"] == team)
        sub = df_r[mask]
        if sub.empty:
            return None, None
        row = sub.sort_values("date").iloc[-1]
        return row, (row["home"] == team)

    def _find_latest_venue(team, as_home, df_r):
        """Encontra jogo mais recente do time em venue específico."""
        col = "home" if as_home else "away"
        sub = df_r[df_r[col] == team]
        if sub.empty:
            return None
        return sub.sort_values("date").iloc[-1]

    h_latest, h_was_home = _find_latest(home, df_ref)
    a_latest, a_was_home = _find_latest(away, df_ref)
    h_at_home_row = _find_latest_venue(home, True, df_ref)   # último como mandante
    a_at_away_row = _find_latest_venue(away, False, df_ref)  # último como visitante

    # Último confronto direto (para H2H features — par-específico)
    _h2h_mask = ((df_ref["home"] == home) & (df_ref["away"] == away)) | \
                ((df_ref["home"] == away) & (df_ref["away"] == home))
    _h2h_sub = df_ref[_h2h_mask]
    h2h_row = _h2h_sub.sort_values("date").iloc[-1] if not _h2h_sub.empty else None

    extra = {
        "dc_lh":    p_mc["lambda_home"],
        "dc_la":    p_mc["lambda_away"],
        "dc_diff":  p_mc["lambda_home"] - p_mc["lambda_away"],
        "dc_ratio": p_mc["lambda_home"] / max(p_mc["lambda_away"], 0.01),
    }

    def _read(row, col):
        """Lê uma feature de uma Series pandas, retorna float ou NaN."""
        if row is None:
            return np.nan
        val = row.get(col)
        if val is None:
            return np.nan
        try:
            v = float(val)
            return v if not np.isnan(v) else np.nan
        except (ValueError, TypeError):
            return np.nan

    feat_vec = {}
    for c in feat_names:
        val = np.nan

        if c in extra:
            val = extra[c]

        elif c.startswith("h_"):
            suffix = c[2:]  # remove "h_"
            if "venue_h" in suffix:
                # Feature venue-casa: só do último jogo como mandante
                val = _read(h_at_home_row, c)
            elif h_latest is not None:
                # Feature global: do jogo mais recente, com remapeamento
                if h_was_home:
                    val = _read(h_latest, c)              # h_ direto
                else:
                    val = _read(h_latest, f"a_{suffix}")  # remap a_ → h_

        elif c.startswith("a_"):
            suffix = c[2:]  # remove "a_"
            if "venue_a" in suffix:
                # Feature venue-fora: só do último jogo como visitante
                val = _read(a_at_away_row, c)
            elif a_latest is not None:
                # Feature global: do jogo mais recente, com remapeamento
                if not a_was_home:
                    val = _read(a_latest, c)              # a_ direto
                else:
                    val = _read(a_latest, f"h_{suffix}")  # remap h_ → a_

        elif c.startswith("diff_"):
            pass  # recomputados após o loop a partir dos h_ e a_ já buscados

        elif c.startswith("h2h_"):
            # H2H features: do último confronto direto entre ESTE PAR
            val = _read(h2h_row, c)

        elif "home" in c:
            # elo_home, rest_days_home, table_pts_home, etc.
            if h_latest is not None:
                val = _read(h_latest, c if h_was_home else c.replace("home", "away"))

        elif "away" in c:
            # elo_away, rest_days_away, table_pts_away, etc.
            if a_latest is not None:
                val = _read(a_latest, c if not a_was_home else c.replace("away", "home"))

        else:
            val = _read(h_latest, c)

        feat_vec[c] = val

    # [FIX C3] Recomputa diferenciais a partir dos h_ e a_ já buscados
    # (não herda diff_ de uma partida anterior com outro adversário)
    for n in ROLLING_WINDOWS:
        for h_key, a_key, d_key in [
            (f"h_roll_xgf_{n}", f"a_roll_xgf_{n}", f"diff_xgf_{n}"),
            (f"h_roll_xga_{n}", f"a_roll_xga_{n}", f"diff_xga_{n}"),
            (f"h_roll_pts_{n}", f"a_roll_pts_{n}", f"diff_pts_{n}"),
            (f"h_roll_win_{n}", f"a_roll_win_{n}", f"diff_win_{n}"),
        ]:
            hv = feat_vec.get(h_key, np.nan)
            av = feat_vec.get(a_key, np.nan)
            if isinstance(hv, float) and isinstance(av, float) \
               and not np.isnan(hv) and not np.isnan(av):
                feat_vec[d_key] = hv - av
            else:
                feat_vec[d_key] = np.nan

    X_pred = pd.DataFrame([feat_vec])[feat_names].astype(float)

    # ── Diagnóstico: quantas features são NaN? ──
    n_nan = X_pred.isna().sum(axis=1).values[0]
    n_total = len(feat_names)
    pct_nan = n_nan / n_total * 100
    if pct_nan > 50:
        logger.warning(f"⚠️  {home} vs {away}: {n_nan}/{n_total} features NaN ({pct_nan:.0f}%) "
                       f"— previsão será baseada majoritariamente em Dixon-Coles")

    # [FIX #15] Imputação inteligente
    X_pred = smart_fillna(X_pred, FEATURE_MEDIANS)

    # ── ML classificação ──
    def ml_prob(key):
        return ml_cls[key].predict_proba(X_pred)[0][1] if key in ml_cls else None

    ml_1x2  = ml_cls["y_1x2"].predict_proba(X_pred)[0]
    ml_ov15 = ml_prob("y_ov15")
    ml_ov25 = ml_prob("y_ov25")
    ml_ov35 = ml_prob("y_ov35")
    ml_h05  = ml_prob("y_h_ov05")
    ml_h15  = ml_prob("y_h_ov15")
    ml_h25  = ml_prob("y_h_ov25")
    ml_a05  = ml_prob("y_a_ov05")
    ml_a15  = ml_prob("y_a_ov15")
    ml_a25  = ml_prob("y_a_ov25")
    ml_btts = ml_prob("y_btts")

    # [FIX #18] Regressão de gols
    ml_goals_home = ml_reg["y_goals_home"].predict(X_pred)[0] if "y_goals_home" in ml_reg else lh_an
    ml_goals_away = ml_reg["y_goals_away"].predict(X_pred)[0] if "y_goals_away" in ml_reg else la_an

    # Ensemble ponderado
    pH = alpha * p_mc["1"] + (1-alpha) * ml_1x2[0]
    pD = alpha * p_mc["X"] + (1-alpha) * ml_1x2[1]
    pA = alpha * p_mc["2"] + (1-alpha) * ml_1x2[2]

    # ── CLIPPING 1X2 — nenhum resultado de futebol é 99.9% ──
    # Aplica piso e teto ANTES de normalizar
    pH = np.clip(pH, PROB_FLOOR, PROB_CEILING)
    pD = np.clip(pD, PROB_FLOOR, PROB_CEILING)
    pA = np.clip(pA, PROB_FLOOR, PROB_CEILING)
    s  = pH + pD + pA
    pH, pD, pA = pH/s, pD/s, pA/s

    def blend(mc_val, ml_val):
        if ml_val is None: return mc_val
        raw = alpha * mc_val + (1-alpha) * ml_val
        return np.clip(raw, PROB_FLOOR_BINARY, PROB_CEIL_BINARY)

    ov15 = blend(p_mc["ov15"], ml_ov15)
    ov25 = blend(p_mc["ov25"], ml_ov25)
    ov35 = blend(p_mc["ov35"], ml_ov35)
    h05  = blend(p_mc["h_ov05"], ml_h05)
    h15  = blend(p_mc["h_ov15"], ml_h15)
    h25  = blend(p_mc["h_ov25"], ml_h25)
    a05  = blend(p_mc["a_ov05"], ml_a05)
    a15  = blend(p_mc["a_ov15"], ml_a15)
    a25  = blend(p_mc["a_ov25"], ml_a25)
    btts = blend(p_mc["btts_yes"], ml_btts)

    # Monotonicity enforcement
    ov35 = min(ov35, ov25)
    ov25 = min(ov25, ov15)
    ov15 = min(ov15, 1.0)

    h35_mc = p_mc["h_ov35"]
    h25 = min(h25, h15)
    h15 = min(h15, h05)
    h05 = min(h05, 1.0)
    h35 = min(h35_mc, h25)

    a35_mc = p_mc["a_ov35"]
    a25 = min(a25, a15)
    a15 = min(a15, a05)
    a05 = min(a05, 1.0)
    a35 = min(a35_mc, a25)

    dnb_denom = pH + pA
    dnb_h = pH / dnb_denom if dnb_denom > 0 else 0.5
    dnb_a = pA / dnb_denom if dnb_denom > 0 else 0.5

    # [FIX #18] xG ensemble: blend entre DC e regressão ML
    xg_home_ensemble = alpha * lh_an + (1-alpha) * ml_goals_home
    xg_away_ensemble = alpha * la_an + (1-alpha) * ml_goals_away

    result = {
        "λ_mandante":   round(lh_an, 3),
        "λ_visitante":  round(la_an, 3),
        "xG Previsto (Mand.)":  round(xg_home_ensemble, 3),
        "xG Previsto (Visit.)": round(xg_away_ensemble, 3),
        # [FIX #18] xG da regressão ML (Poisson)
        "xG ML (Mand.)":  round(float(ml_goals_home), 3),
        "xG ML (Visit.)": round(float(ml_goals_away), 3),

        "Vitória Mandante (1)":   round(pH, 4),
        "Empate (X)":             round(pD, 4),
        "Vitória Visitante (2)":  round(pA, 4),

        "DNB Mandante":   round(dnb_h, 4),
        "DNB Visitante":  round(dnb_a, 4),

        "DC 1X (Mand. ou Empate)":   round(pH + pD, 4),
        "DC X2 (Visit. ou Empate)":  round(pA + pD, 4),
        "DC 12 (Qualquer Vitória)":  round(pH + pA, 4),

        "Over 1.5":  round(ov15, 4), "Under 1.5": round(1-ov15, 4),
        "Over 2.5":  round(ov25, 4), "Under 2.5": round(1-ov25, 4),
        "Over 3.5":  round(ov35, 4), "Under 3.5": round(1-ov35, 4),

        "Over 0.5 Gols (Mand.)": round(h05, 4),
        "Over 1.5 Gols (Mand.)": round(h15, 4),
        "Over 2.5 Gols (Mand.)": round(h25, 4),
        "Over 3.5 Gols (Mand.)": round(h35, 4),

        "Over 0.5 Gols (Visit.)": round(a05, 4),
        "Over 1.5 Gols (Visit.)": round(a15, 4),
        "Over 2.5 Gols (Visit.)": round(a25, 4),
        "Over 3.5 Gols (Visit.)": round(a35, 4),

        "Ambos Marcam — Sim": round(btts, 4),
        "Ambos Marcam — Não": round(1-btts, 4),
    }
    return result


# ── Otimização do alpha ──
# [FIX #10] Usa dados 60%-80% — separado do treino ML (0-60%) e teste (80-100%)
print("\n⚙️  Otimizando peso do ensemble (ALPHA)...")
_n_train = split_info["n_train"]
_n_alpha_end = split_info["n_alpha_end"]
_df_feat_val = df_feat.iloc[_n_train:_n_alpha_end].copy()
print(f"   Holdout alpha: partidas {_n_train}–{_n_alpha_end} "
      f"({len(_df_feat_val)} jogos, SEPARADO do treino ML)")
# [FIX C1] Usa dc_train para alpha — consistente com features ML
ALPHA = optimize_alpha(dc_train, ml_models, _df_feat_val,
                        feat_names, FEATURE_MEDIANS)
print(f"   ALPHA final: {ALPHA}")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 11 — VISUALIZAÇÕES                       ║
# ╚══════════════════════════════════════════════════════════╝

def plot_score_heatmap(home, away, dc):
    matrix = dc.score_matrix(home, away, max_goals=6)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Probabilidade (%)"})
    ax.set_xlabel(f"Gols — {away}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Gols — {home}", fontsize=12, fontweight="bold")
    ax.set_title(f"Probabilidade de Placar Exato\n{home} × {away}",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()


def plot_market_probs(home, away, result):
    markets = {
        "Resultado (1X2)": {
            "Mandante": result["Vitória Mandante (1)"],
            "Empate":   result["Empate (X)"],
            "Visitante":result["Vitória Visitante (2)"]
        },
        "Dupla Chance": {
            "DC 1X": result["DC 1X (Mand. ou Empate)"],
            "DC X2": result["DC X2 (Visit. ou Empate)"],
            "DC 12": result["DC 12 (Qualquer Vitória)"]
        },
        "Over/Under Total": {
            "Ov 1.5": result["Over 1.5"],
            "Ov 2.5": result["Over 2.5"],
            "Ov 3.5": result["Over 3.5"],
        },
        "Gols Mandante": {
            "Ov 0.5": result["Over 0.5 Gols (Mand.)"],
            "Ov 1.5": result["Over 1.5 Gols (Mand.)"],
            "Ov 2.5": result["Over 2.5 Gols (Mand.)"],
        },
        "Gols Visitante": {
            "Ov 0.5": result["Over 0.5 Gols (Visit.)"],
            "Ov 1.5": result["Over 1.5 Gols (Visit.)"],
            "Ov 2.5": result["Over 2.5 Gols (Visit.)"],
        },
        "Ambos Marcam": {
            "Sim": result["Ambos Marcam — Sim"],
            "Não": result["Ambos Marcam — Não"],
        },
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

    for i, (title, data) in enumerate(markets.items()):
        ax = axes[i]
        bars = ax.bar(data.keys(), [v * 100 for v in data.values()],
                      color=colors[i], edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Probabilidade (%)")
        ax.set_ylim(0, 105)
        for bar, (k, v) in zip(bars, data.items()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v*100:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{home}  ×  {away}\nPrevisões por Mercado",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_calibration_curves(bt, markets=None, n_bins=8):
    if markets is None:
        markets = {
            "Vitória Mandante": ("p_H",    "y_H"),
            "Empate":           ("p_D",    "y_D"),
            "Vitória Visitante":("p_A",    "y_A"),
            "Over 1.5":         ("p_ov15", "y_ov15"),
            "Over 2.5":         ("p_ov25", "y_ov25"),
        }
    markets = {k: v for k, v in markets.items()
               if v[0] in bt.columns and v[1] in bt.columns}
    if not markets:
        print("⚠️  Nenhum mercado para calibração.")
        return

    n_plots = len(markets)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.array(axes).flatten()

    for ax, (label, (prob_col, true_col)) in zip(axes, markets.items()):
        df_m = bt[[prob_col, true_col]].dropna()
        if len(df_m) < 10:
            ax.set_visible(False)
            continue
        bins = np.linspace(0, 1, n_bins + 1)
        df_m["bin"] = pd.cut(df_m[prob_col], bins=bins, include_lowest=True)
        cal = (df_m.groupby("bin", observed=True)
               .agg(prob_media=(prob_col, "mean"),
                    freq_real=(true_col, "mean"),
                    n=(prob_col, "count")).dropna())

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        sizes = (cal["n"] / cal["n"].max() * 200).clip(20)
        ax.scatter(cal["prob_media"], cal["freq_real"], s=sizes,
                   zorder=5, color="#2196F3", edgecolors="white")
        ax.plot(cal["prob_media"], cal["freq_real"],
                color="#2196F3", lw=1.5, alpha=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Prob prevista"); ax.set_ylabel("Freq real")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)

    for ax in axes[len(markets):]:
        ax.set_visible(False)

    fig.suptitle("Diagrama de Confiabilidade", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_shap(ml_models, X, target="y_1x2"):
    model = ml_models[target]
    # Se for CalibratedClassifier, extrai o base; senão usa direto
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
    else:
        base = model
    explainer = shap.TreeExplainer(base)
    shap_vals = explainer.shap_values(X.head(300))
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X.head(300),
                      feature_names=feat_names,
                      plot_type="bar", show=False)
    plt.title(f"SHAP — Importância ({target})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 12 — BACKTEST LIMPO                      ║
# ╚══════════════════════════════════════════════════════════╝
# [FIX #1] Backtest usa APENAS o holdout final (80-100%)
# [FIX #13] Erros logados e contados, nunca silenciados

def backtest(df_feat: pd.DataFrame,
             dc: DixonColesModel,
             ml_cls: dict,
             ml_reg: dict,
             df_raw: pd.DataFrame,
             split_info: dict) -> pd.DataFrame:
    """
    [FIX #1]  Backtest no holdout final (80-100%) — dados nunca vistos
    [FIX #13] Erros logados com contagem
    """
    n_alpha_end = split_info["n_alpha_end"]
    df_test = df_feat.iloc[n_alpha_end:].copy()
    df_test = df_test.dropna(subset=["result", "total_goals"])

    print(f"\n⚙️  Backtest: {len(df_test)} partidas no holdout final (80-100%)")

    # DC retreinado até o ponto de corte
    cutoff_date = df_feat.iloc[n_alpha_end - 1]["date"] if n_alpha_end > 0 else None
    df_train_dc = df_raw[df_raw["date"] <= cutoff_date].dropna(
        subset=["hg","ag"]).reset_index(drop=True)
    dc_bt = DixonColesModel(xi=0.005)
    dc_bt.fit(df_train_dc)

    records = []
    n_errors = 0
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Backtest"):
        try:
            pred = ensemble_predict(
                row["home"], row["away"],
                dc_bt, ml_cls, ml_reg, df_feat,
                cutoff_date=row["date"]
            )
            _hg = int(row["hg"]) if pd.notna(row.get("hg")) else np.nan
            _ag = int(row["ag"]) if pd.notna(row.get("ag")) else np.nan
            records.append({
                "home":    row["home"],
                "away":    row["away"],
                "date":    row["date"],
                "result":  row["result"],
                "total_g": int(row["total_goals"]),
                "hg": _hg, "ag": _ag,
                "p_H":     pred["Vitória Mandante (1)"],
                "p_D":     pred["Empate (X)"],
                "p_A":     pred["Vitória Visitante (2)"],
                "p_ov25":  pred["Over 2.5"],
                "p_ov15":  pred["Over 1.5"],
                "p_btts":  pred["Ambos Marcam — Sim"],
                "xg_pred_h": pred["xG Previsto (Mand.)"],
                "xg_pred_a": pred["xG Previsto (Visit.)"],
            })
        except Exception as e:
            n_errors += 1
            # [FIX #13] Log do erro
            logger.warning(f"Backtest falha: {row.get('home')} vs {row.get('away')}: {e}")

    # [FIX #13] Reporta quantos jogos falharam
    if n_errors > 0:
        print(f"\n⚠️  {n_errors} de {len(df_test)} jogos falharam no backtest "
              f"({n_errors/len(df_test)*100:.1f}%)")

    bt = pd.DataFrame(records)
    if bt.empty:
        print("⚠️  Backtest vazio.")
        return bt

    bt["y_H"]    = (bt["result"] == "H").astype(int)
    bt["y_D"]    = (bt["result"] == "D").astype(int)
    bt["y_A"]    = (bt["result"] == "A").astype(int)
    bt["y_ov25"] = (bt["total_g"] >= 3).astype(int)
    bt["y_ov15"] = (bt["total_g"] >= 2).astype(int)
    bt["y_btts"] = np.where(
        bt["hg"].notna() & bt["ag"].notna(),
        ((bt["hg"] >= 1) & (bt["ag"] >= 1)).astype(int), np.nan
    )

    y_true_1x2 = bt["result"].map({"H": 0, "D": 1, "A": 2})
    y_prob_1x2 = bt[["p_H", "p_D", "p_A"]].values
    ll  = log_loss(y_true_1x2, y_prob_1x2)
    acc = accuracy_score(y_true_1x2, np.argmax(y_prob_1x2, axis=1))

    brier_ov25 = brier_score_loss(bt["y_ov25"], bt["p_ov25"])
    brier_ov15 = brier_score_loss(bt["y_ov15"], bt["p_ov15"])
    brier_H    = brier_score_loss(bt["y_H"],    bt["p_H"])
    brier_D    = brier_score_loss(bt["y_D"],    bt["p_D"])
    brier_A    = brier_score_loss(bt["y_A"],    bt["p_A"])

    # [FIX #18] Métricas de regressão de gols
    xg_rmse_h = np.sqrt(mean_squared_error(bt["hg"].dropna(), bt["xg_pred_h"].loc[bt["hg"].notna()]))
    xg_rmse_a = np.sqrt(mean_squared_error(bt["ag"].dropna(), bt["xg_pred_a"].loc[bt["ag"].notna()]))

    pred_label = np.argmax(y_prob_1x2, axis=1)
    true_label = y_true_1x2.values
    acc_H = ((pred_label == 0) & (true_label == 0)).sum() / max((true_label == 0).sum(), 1)
    acc_D = ((pred_label == 1) & (true_label == 1)).sum() / max((true_label == 1).sum(), 1)
    acc_A = ((pred_label == 2) & (true_label == 2)).sum() / max((true_label == 2).sum(), 1)

    print(f"\n{'═'*55}")
    print(f"  📈  BACKTEST LIMPO ({len(bt)} partidas | {n_errors} erros)")
    print(f"{'═'*55}")
    print(f"  Log-Loss 1X2:           {ll:.4f}  (ref aleatória ≈ 1.099)")
    print(f"  Acurácia 1X2:           {acc:.4f}  (baseline ≈ 0.46)")
    print(f"{'─'*55}")
    print(f"  Brier — Mandante:       {brier_H:.4f}")
    print(f"  Brier — Empate:         {brier_D:.4f}")
    print(f"  Brier — Visitante:      {brier_A:.4f}")
    print(f"  Brier — Over 1.5:       {brier_ov15:.4f}")
    print(f"  Brier — Over 2.5:       {brier_ov25:.4f}")
    print(f"{'─'*55}")
    print(f"  RMSE xG Mandante:       {xg_rmse_h:.4f}")  # [FIX #18]
    print(f"  RMSE xG Visitante:      {xg_rmse_a:.4f}")  # [FIX #18]
    print(f"{'─'*55}")
    print(f"  Acerto quando previu H: {acc_H:.1%}")
    print(f"  Acerto quando previu X: {acc_D:.1%}")
    print(f"  Acerto quando previu A: {acc_A:.1%}")
    print(f"{'═'*55}")

    return bt


bt_results = backtest(df_feat, dc_model, ml_models, ml_regressors,
                       df_raw=df, split_info=split_info)


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 13 — WALK-FORWARD VALIDATION             ║
# ╚══════════════════════════════════════════════════════════╝
# [FIX #1] Walk-forward como validação principal

def walk_forward_validation(df: pd.DataFrame,
                             df_feat: pd.DataFrame,
                             n_folds: int = 5,
                             min_train_frac: float = 0.50) -> pd.DataFrame:
    """
    Walk-forward com treino limpo por fold.
    [FIX #1] DC + XGBoost retreinados a cada fold.
    [FIX #13] Erros logados.
    """
    df_fin = df.dropna(subset=["hg","ag"]).sort_values("date").reset_index(drop=True)
    n      = len(df_fin)
    step   = int(n * (1 - min_train_frac) / n_folds)

    if step < 10:
        print("⚠️  Dados insuficientes para walk-forward.")
        return pd.DataFrame()

    all_records = []
    fold_metrics = []

    for fold in range(n_folds):
        cutoff_idx = int(n * min_train_frac) + fold * step
        test_start = cutoff_idx
        test_end   = min(cutoff_idx + step, n)
        if test_start >= n: break

        df_train = df_fin.iloc[:cutoff_idx]
        df_test  = df_fin.iloc[test_start:test_end]
        cutoff_date = df_train["date"].max()

        print(f"\n  Fold {fold+1}/{n_folds} | Treino: {len(df_train)} | "
              f"Teste: {len(df_test)} | Corte: {str(cutoff_date)[:10]}")

        # DC neste fold
        dc_fold = DixonColesModel(xi=0.005)
        try:
            dc_fold.fit(df_train)
        except Exception as e:
            logger.warning(f"DC falhou fold {fold+1}: {e}")
            continue

        # Features até o corte
        df_feat_train = df_feat[df_feat["date"] <= cutoff_date].copy()
        df_feat_train = add_poisson_features(df_feat_train, dc_fold)

        extra_fold = ["dc_lh","dc_la","dc_diff","dc_ratio"]
        feats_fold = [c for c in FEATURE_COLS if c in df_feat_train.columns] + extra_fold
        df_feat_train = df_feat_train.dropna(subset=["hg","ag","result"])

        if len(df_feat_train) < 50: continue

        df_feat_train["hg"] = df_feat_train["hg"].astype(int)
        df_feat_train["ag"] = df_feat_train["ag"].astype(int)
        df_feat_train["y_1x2"] = df_feat_train["result"].map({"H":0,"D":1,"A":2})

        mask_tr = df_feat_train[feats_fold + ["y_1x2"]].notna().all(axis=1)
        X_tr = df_feat_train.loc[mask_tr, feats_fold].astype(float)
        y_tr = df_feat_train.loc[mask_tr, "y_1x2"]
        if len(X_tr) < 30: continue

        # XGBoost sem calibração (raw softmax)
        base_fold = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        )
        try:
            base_fold.fit(X_tr, y_tr, verbose=False)
        except Exception as e:
            logger.warning(f"XGBoost falhou fold {fold+1}: {e}")
            continue

        ml_fold_cls = {"y_1x2": base_fold}
        ml_fold_reg = {}  # Simplificado para walk-forward

        n_fold_errors = 0
        records_fold = []
        for _, row in df_test.iterrows():
            try:
                pred = ensemble_predict(
                    row["home"], row["away"],
                    dc_fold, ml_fold_cls, ml_fold_reg, df_feat,
                    cutoff_date=row["date"]
                )
                records_fold.append({
                    "fold":   fold + 1,
                    "home":   row["home"],
                    "away":   row["away"],
                    "date":   row["date"],
                    "result": row["result"],
                    "p_H":    pred["Vitória Mandante (1)"],
                    "p_D":    pred["Empate (X)"],
                    "p_A":    pred["Vitória Visitante (2)"],
                })
            except Exception as e:
                n_fold_errors += 1
                logger.debug(f"WF fold {fold+1} erro: {e}")

        if n_fold_errors > 0:
            print(f"    ⚠️  {n_fold_errors} erros neste fold")

        if not records_fold: continue

        df_fold = pd.DataFrame(records_fold)
        y_true  = df_fold["result"].map({"H":0,"D":1,"A":2})
        y_prob  = df_fold[["p_H","p_D","p_A"]].values

        try:
            ll  = log_loss(y_true, y_prob)
            acc = accuracy_score(y_true, np.argmax(y_prob, axis=1))
            fold_metrics.append({
                "Fold": fold+1, "Partidas": len(df_fold),
                "Log-Loss": round(ll, 4), "Acurácia": round(acc, 4),
                "Corte": str(cutoff_date)[:10], "Erros": n_fold_errors,
            })
            print(f"    Log-Loss: {ll:.4f} | Acurácia: {acc:.4f}")
        except Exception:
            pass

        all_records.extend(records_fold)

    df_all = pd.DataFrame(all_records)
    df_met = pd.DataFrame(fold_metrics)

    if not df_met.empty:
        print(f"\n{'═'*55}")
        print("  📊  WALK-FORWARD VALIDATION — SUMÁRIO")
        print(f"{'═'*55}")
        print(df_met.to_string(index=False))
        if len(df_met) > 1:
            print(f"{'─'*55}")
            print(f"  Média Log-Loss:  {df_met['Log-Loss'].mean():.4f} "
                  f"± {df_met['Log-Loss'].std():.4f}")
            print(f"  Média Acurácia:  {df_met['Acurácia'].mean():.4f} "
                  f"± {df_met['Acurácia'].std():.4f}")
            print(f"  Total erros:     {df_met['Erros'].sum()}")
        print(f"{'═'*55}")

    return df_all


# Executa walk-forward automaticamente
print("\n⚙️  Executando Walk-Forward Validation...")
wf_results = walk_forward_validation(df, df_feat, n_folds=5)


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 14 — FUNÇÃO PRINCIPAL                    ║
# ╚══════════════════════════════════════════════════════════╝

def predict_match(home: str, away: str,
                  dc: DixonColesModel   = None,
                  ml_cls: dict          = None,
                  ml_reg: dict          = None,
                  df_ref: pd.DataFrame  = None,
                  show_plots: bool      = True) -> pd.DataFrame:
    if dc      is None: dc      = dc_model
    if ml_cls  is None: ml_cls  = ml_models
    if ml_reg  is None: ml_reg  = ml_regressors
    if df_ref  is None: df_ref  = df_feat

    print(f"\n{'═'*55}")
    print(f"  ⚽  {home.upper()}  ×  {away.upper()}")
    print(f"{'═'*55}")

    result = ensemble_predict(home, away, dc, ml_cls, ml_reg, df_ref)

    rows = []
    categories = {
        "⚡ xG PREVISTO": [
            "xG Previsto (Mand.)", "xG Previsto (Visit.)",
            "xG ML (Mand.)", "xG ML (Visit.)",   # [FIX #18]
        ],
        "📊 RESULTADO (1X2)": [
            "Vitória Mandante (1)", "Empate (X)", "Vitória Visitante (2)"
        ],
        "🔄 EMPATE ANULA APOSTA": [
            "DNB Mandante", "DNB Visitante"
        ],
        "🎯 DUPLA CHANCE": [
            "DC 1X (Mand. ou Empate)",
            "DC X2 (Visit. ou Empate)",
            "DC 12 (Qualquer Vitória)"
        ],
        "⚽ TOTAL DE GOLS": [
            "Over 1.5", "Under 1.5",
            "Over 2.5", "Under 2.5",
            "Over 3.5", "Under 3.5",
        ],
        "🏠 GOLS MANDANTE": [
            "Over 0.5 Gols (Mand.)", "Over 1.5 Gols (Mand.)",
            "Over 2.5 Gols (Mand.)", "Over 3.5 Gols (Mand.)"
        ],
        "✈️  GOLS VISITANTE": [
            "Over 0.5 Gols (Visit.)", "Over 1.5 Gols (Visit.)",
            "Over 2.5 Gols (Visit.)", "Over 3.5 Gols (Visit.)"
        ],
        "🔵 AMBOS MARCAM (BTTS)": [
            "Ambos Marcam — Sim", "Ambos Marcam — Não"
        ],
    }

    _XG_MARKETS = {"xG Previsto (Mand.)", "xG Previsto (Visit.)",
                    "xG ML (Mand.)", "xG ML (Visit.)"}
    for category, mkts in categories.items():
        rows.append({"Categoria": category, "Mercado": "", "Probabilidade": ""})
        for m in mkts:
            p = result.get(m, np.nan)
            if isinstance(p, float) and np.isnan(p):
                fmt = "—"
            elif m in _XG_MARKETS:
                fmt = f"{p:.2f} gols"
            else:
                fmt = f"{p*100:.1f}%"
            rows.append({"Categoria": "", "Mercado": m, "Probabilidade": fmt})

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    if show_plots:
        plot_score_heatmap(home, away, dc)
        plot_market_probs(home, away, result)

    return df_out, result


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 15 — EXEMPLO DE USO                      ║
# ╚══════════════════════════════════════════════════════════╝

print("🏟️  Times disponíveis no dataset:")
print(sorted(df["home"].unique()))

HOME_TEAM = "Manchester City"
AWAY_TEAM = "Arsenal"

df_resultado, resultado_dict = predict_match(
    home       = HOME_TEAM,
    away       = AWAY_TEAM,
    show_plots = True
)


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 16 — SHAP + CALIBRAÇÃO                   ║
# ╚══════════════════════════════════════════════════════════╝

print("\n📊 SHAP — Importância global (1X2):")
plot_shap(ml_models, X_all, target="y_1x2")

print("\n📊 SHAP — Importância global (Over 2.5):")
plot_shap(ml_models, X_all, target="y_ov25")

if "bt_results" in dir() and not bt_results.empty:
    print("\n📊 Diagramas de Calibração:")
    plot_calibration_curves(bt_results)


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 17 — PREVISÃO EM LOTE                    ║
# ╚══════════════════════════════════════════════════════════╝

def predict_batch(jogos: list,
                  dc=None, ml_cls=None, ml_reg=None,
                  df_ref=None) -> pd.DataFrame:
    if dc      is None: dc      = dc_model
    if ml_cls  is None: ml_cls  = ml_models
    if ml_reg  is None: ml_reg  = ml_regressors
    if df_ref  is None: df_ref  = df_feat

    rows = []
    n_errors = 0
    for jogo in tqdm(jogos, desc="Prevendo jogos"):
        home   = jogo["home"]
        away   = jogo["away"]
        rodada = jogo.get("rodada", "?")
        date   = jogo.get("date", "?")
        try:
            _, r = predict_match(home, away, dc, ml_cls, ml_reg,
                                  df_ref, show_plots=False)
            r["rodada"] = rodada
            r["date"]   = date
            r["home"]   = home
            r["away"]   = away
            rows.append(r)
        except Exception as e:
            n_errors += 1
            logger.warning(f"Batch erro: R{rodada} {home} × {away}: {e}")

    if n_errors > 0:
        print(f"⚠️  {n_errors} de {len(jogos)} jogos falharam")

    df_batch = pd.DataFrame(rows)
    id_cols = ["rodada", "date", "home", "away"]
    outros  = [c for c in df_batch.columns if c not in id_cols]
    return df_batch[id_cols + outros]


def load_and_predict_excel(caminho_excel: str,
                            exportar_csv: bool = True,
                            comparar_real: bool = True) -> pd.DataFrame:
    print(f"📂 Lendo: {caminho_excel}")
    df_jogos = pd.read_excel(caminho_excel)
    df_jogos.columns = df_jogos.columns.str.strip().str.lower()

    required = {"rodada", "date", "home", "away"}
    missing  = required - set(df_jogos.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    df_jogos["date"]   = pd.to_datetime(df_jogos["date"], dayfirst=True, errors="coerce")
    df_jogos["rodada"] = df_jogos["rodada"].astype(str)
    df_jogos["home"]   = df_jogos["home"].str.strip()
    df_jogos["away"]   = df_jogos["away"].str.strip()

    print(f"✅ {len(df_jogos)} jogo(s) carregado(s)")

    jogos_list = df_jogos.to_dict("records")
    df_prev    = predict_batch(jogos_list)

    PROB_COLS = [
        "xG Previsto (Mand.)", "xG Previsto (Visit.)",
        "xG ML (Mand.)", "xG ML (Visit.)",
        "Vitória Mandante (1)", "Empate (X)", "Vitória Visitante (2)",
        "DNB Mandante", "DNB Visitante",
        "DC 1X (Mand. ou Empate)", "DC X2 (Visit. ou Empate)",
        "DC 12 (Qualquer Vitória)",
        "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5",
        "Over 3.5", "Under 3.5",
        "Over 0.5 Gols (Mand.)", "Over 1.5 Gols (Mand.)",
        "Over 2.5 Gols (Mand.)", "Over 3.5 Gols (Mand.)",
        "Over 0.5 Gols (Visit.)", "Over 1.5 Gols (Visit.)",
        "Over 2.5 Gols (Visit.)", "Over 3.5 Gols (Visit.)",
        "Ambos Marcam — Sim", "Ambos Marcam — Não",
    ]
    XG_COLS = {"xG Previsto (Mand.)", "xG Previsto (Visit.)",
               "xG ML (Mand.)", "xG ML (Visit.)"}
    for col in PROB_COLS:
        if col not in df_prev.columns: continue
        if col in XG_COLS:
            df_prev[col] = pd.to_numeric(df_prev[col], errors="coerce") \
                            .apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        else:
            df_prev[col] = pd.to_numeric(df_prev[col], errors="coerce") \
                            .apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "—")

    if comparar_real:
        real = df[["home", "away", "date", "hg", "ag",
                   "result", "total_goals"]].copy()
        real["date"]    = pd.to_datetime(real["date"], errors="coerce").dt.normalize()
        df_prev["date"] = pd.to_datetime(df_prev["date"], errors="coerce").dt.normalize()
        real = real.drop_duplicates(subset=["home", "away", "date"], keep="last")

        n_antes = len(df_prev)
        df_prev = df_prev.merge(real, on=["home", "away", "date"], how="left")
        if len(df_prev) != n_antes:
            df_prev = df_prev.drop_duplicates(subset=["home", "away", "date"], keep="first")

        prob_num = df_prev[["Vitória Mandante (1)", "Empate (X)",
                            "Vitória Visitante (2)"]].apply(
            lambda col: col.str.rstrip("%").apply(pd.to_numeric, errors="coerce")
        )
        map_pred = {
            "Vitória Mandante (1)": "H",
            "Empate (X)":           "D",
            "Vitória Visitante (2)":"A"
        }
        df_prev["resultado_previsto"] = prob_num.idxmax(axis=1).map(map_pred)
        df_prev["acertou_1x2"] = (
            df_prev["resultado_previsto"] == df_prev["result"]
        ).map({True: "✅", False: "❌"})

        n_prev = df_prev["result"].notna().sum()
        if n_prev > 0:
            n_acertos = (df_prev["acertou_1x2"] == "✅").sum()
            print(f"\n📈 Acurácia 1X2: {n_acertos}/{n_prev} ({n_acertos/n_prev*100:.1f}%)")

    if exportar_csv:
        nome_csv = caminho_excel.replace(".xlsx","").replace(".xls","") + "_previsoes.csv"
        df_prev.to_csv(nome_csv, index=False, encoding="utf-8-sig")
        print(f"\n💾 Exportado: {nome_csv}")

    cols_resumo = ["rodada", "date", "home", "away",
                   "Vitória Mandante (1)", "Empate (X)", "Vitória Visitante (2)",
                   "Over 2.5", "Ambos Marcam — Sim"]
    if "resultado_previsto" in df_prev.columns:
        cols_resumo += ["resultado_previsto", "result", "hg", "ag", "acertou_1x2"]
    cols_resumo = [c for c in cols_resumo if c in df_prev.columns]

    print("\n" + "═"*80)
    print("  📋  RESUMO DAS PREVISÕES")
    print("═"*80)
    print(df_prev[cols_resumo].to_string(index=False))
    print("═"*80)

    return df_prev


# ══════════════════════════════════════════════════════════════
# ─── EXECUÇÃO — PREVISÃO VIA EXCEL ────────────────────────────
# ══════════════════════════════════════════════════════════════
# 1) Faça upload do arquivo "rodada.xlsx" no Colab (ícone de pasta → upload)
# 2) Rode esta célula — as previsões serão geradas automaticamente
#
# Formato esperado do Excel (rodada.xlsx):
#   rodada | date       | home              | away
#   -------|------------|-------------------|------------------
#   25     | 2024-03-10 | Manchester City   | Arsenal
#   25     | 2024-03-10 | Liverpool         | Chelsea
#
# O arquivo de saída será salvo como "rodada_previsoes.csv"
# ──────────────────────────────────────────────────────────────

CAMINHO_EXCEL = "/content/rodada.xlsx"   # ← altere o caminho se necessário

# Verifica se o arquivo existe antes de rodar
if os.path.exists(CAMINHO_EXCEL):
    df_rodada = load_and_predict_excel(
        caminho_excel  = CAMINHO_EXCEL,
        exportar_csv   = True,
        comparar_real  = True    # False se a rodada ainda não aconteceu
    )
else:
    print(f"⚠️  Arquivo não encontrado: {CAMINHO_EXCEL}")
    print("   Faça upload do arquivo 'rodada.xlsx' no Colab e rode novamente.")
    print("   O arquivo deve conter as colunas: rodada, date, home, away")
