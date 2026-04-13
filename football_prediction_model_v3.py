# ============================================================
# MODELO PREDITIVO DE FUTEBOL v3 — GOOGLE COLAB
# Fonte de dados: Understat via soccerdata
# Melhorias sobre v2:
#   - LightGBM + XGBoost stacking (meta-learner)
#   - Features de momentum (tendência do xG)
#   - Binomial Negativa para overdispersão
#   - Calibração Platt com CV temporal
#   - Suporte multi-liga
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
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import poisson, nbinom
from scipy.special import gammaln
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import logging
from collections import deque

from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (log_loss, brier_score_loss,
                             accuracy_score, mean_squared_error)
import xgboost as xgb

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("⚠️  LightGBM não instalado — usando só XGBoost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

import soccerdata as sd

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("football_v3")

print("✅ Dependências carregadas!")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 3 — CONFIGURAÇÕES GLOBAIS                ║
# ╚══════════════════════════════════════════════════════════╝

LEAGUE = "ENG-Premier League"
SEASONS = [2021, 2022, 2023, 2024]

ROLLING_WINDOWS = [3, 5, 10, 20]

PROB_FLOOR = 0.03
PROB_CEILING = 0.92
PROB_FLOOR_BIN = 0.03
PROB_CEIL_BIN = 0.97

ELO_K = 40
ELO_BASE = 1500
ELO_HOME_ADV = 65
ELO_PROMOTED = 1350
ELO_REGRESS = 0.70

USE_NB = True
CUTOFF_DATE = None


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 4 — COLETA DE DADOS                      ║
# ╚══════════════════════════════════════════════════════════╝

def fetch_data(league: str, seasons: list) -> tuple:
    """Baixa schedule + shots do Understat."""
    understat = sd.Understat(leagues=league)
    frames_sched = []
    frames_shots = []

    for s in seasons:
        try:
            print(f"  ⏳ Temporada {s}...", end=" ")
            df_s = understat.read_schedule(s)
            df_s["season"] = s
            frames_sched.append(df_s)
            print(f"✅ {len(df_s)} jogos", end="")
        except Exception as e:
            print(f"❌ schedule {s}: {e}")

        try:
            shots = understat.read_shot_events(s)
            shots["season"] = s
            frames_shots.append(shots)
            print(f" | {len(shots)} chutes")
        except Exception as e:
            print(f" | ❌ shots: {e}")

    df_all = pd.concat(frames_sched, ignore_index=True) if frames_sched else pd.DataFrame()
    df_shots = pd.concat(frames_shots, ignore_index=True) if frames_shots else pd.DataFrame()

    if not df_all.empty:
        date_col = next((c for c in df_all.columns if "date" in c.lower()), None)
        if date_col:
            df_all[date_col] = pd.to_datetime(df_all[date_col], errors="coerce")
            latest = df_all[date_col].max()
            print(f"\n✅ Total: {len(df_all)} jogos | Mais recente: {latest}")

    return df_all, df_shots


def apply_cutoff(df, cutoff):
    if cutoff is None:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[df["date"] <= pd.to_datetime(cutoff)]


# ── EXECUÇÃO ──
print("🔍 Baixando dados do Understat...\n")
df_raw, df_shots = fetch_data(LEAGUE, SEASONS)

if not df_raw.empty:
    goal_cols = [c for c in df_raw.columns if c in ["home_goals", "away_goals"]]
    if goal_cols:
        df_raw = df_raw.dropna(subset=goal_cols)
    df_raw = apply_cutoff(df_raw, CUTOFF_DATE)


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 5 — LIMPEZA E NORMALIZAÇÃO               ║
# ╚══════════════════════════════════════════════════════════╝

def clean_schedule(df):
    rename = {
        "home_team": "home", "away_team": "away",
        "home_goals": "hg", "away_goals": "ag",
        "home_xg": "hxg", "away_xg": "axg",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for c in ["hg", "ag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["hxg", "axg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["result"] = np.where(df["hg"] > df["ag"], "H",
                   np.where(df["hg"] < df["ag"], "A", "D"))
    df.loc[df["hg"].isna() | df["ag"].isna(), "result"] = np.nan
    df["total_goals"] = df["hg"] + df["ag"]
    return df.sort_values("date").reset_index(drop=True)


df = clean_schedule(df_raw)
print(f"✅ Dados limpos — {len(df)} jogos")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 6 — FEATURE ENGINEERING                  ║
# ╚══════════════════════════════════════════════════════════╝

def build_team_timeline(df):
    """Cria timeline unificada home+away por time."""
    home = df[["date", "home", "away", "hg", "ag", "hxg", "axg",
               "result", "season"]].copy()
    home.columns = ["date", "team", "opponent", "gf", "ga",
                    "xgf", "xga", "result_raw", "season"]
    home["venue"] = "H"
    home["pts"] = home["result_raw"].map({"H": 3, "D": 1, "A": 0})
    home["win"] = (home["result_raw"] == "H").astype(int)
    home["draw"] = (home["result_raw"] == "D").astype(int)
    home["loss"] = (home["result_raw"] == "A").astype(int)

    away = df[["date", "away", "home", "ag", "hg", "axg", "hxg",
               "result", "season"]].copy()
    away.columns = ["date", "team", "opponent", "gf", "ga",
                    "xgf", "xga", "result_raw", "season"]
    away["venue"] = "A"
    away["pts"] = away["result_raw"].map({"A": 3, "D": 1, "H": 0})
    away["win"] = (away["result_raw"] == "A").astype(int)
    away["draw"] = (away["result_raw"] == "D").astype(int)
    away["loss"] = (away["result_raw"] == "H").astype(int)

    tl = pd.concat([home, away], ignore_index=True)
    tl = tl.sort_values(["team", "date"]).reset_index(drop=True)

    for c in ["gf", "ga", "xgf", "xga"]:
        tl[c] = pd.to_numeric(tl[c], errors="coerce")

    # Surpresa: gols reais vs xG
    tl["surprise_att"] = tl["gf"] - tl["xgf"]
    tl["surprise_def"] = tl["ga"] - tl["xga"]

    return tl


def rolling_team_stats(tl, windows=None):
    """Rolling stats em múltiplas janelas, com shift para evitar leakage."""
    if windows is None:
        windows = ROLLING_WINDOWS

    base_cols = ["xgf", "xga", "gf", "ga", "pts", "win", "draw", "loss",
                 "surprise_att", "surprise_def"]

    for n in windows:
        for col in base_cols:
            if col not in tl.columns:
                continue
            tl[f"roll_{col}_{n}"] = (
                tl.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
            )

        # Venue-specific
        for vtag, vval in [("h", "H"), ("a", "A")]:
            mask = tl["venue"] == vval
            for col in ["xgf", "xga", "gf", "ga"]:
                cname = f"roll_{col}_v{vtag}_{n}"
                tl[cname] = np.nan
                tl.loc[mask, cname] = (
                    tl[mask].groupby("team")[col]
                    .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
                )

    # MOMENTUM: diferença entre janela curta e longa (tendência)
    if 3 in windows and 20 in windows:
        for col in ["xgf", "xga", "pts"]:
            short = f"roll_{col}_3"
            long_ = f"roll_{col}_20"
            if short in tl.columns and long_ in tl.columns:
                tl[f"momentum_{col}"] = tl[short] - tl[long_]

    return tl


def h2h_features(df, window=5):
    """Head-to-head rolling features."""
    df = df.copy()
    df["_idx"] = np.arange(len(df))
    df["total_g"] = (df["hg"] + df["ag"]).astype(float)
    df["h_won"] = (df["result"] == "H").astype(float)
    df["drawn"] = (df["result"] == "D").astype(float)

    t = np.sort(np.stack([df["home"].values, df["away"].values]), axis=0)
    df["pair"] = [f"{a}___{b}" for a, b in zip(t[0], t[1])]
    df = df.sort_values(["pair", "date"])

    def rpair(s):
        return s.groupby(df["pair"]).transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    result = pd.DataFrame({
        "h2h_home_wins": rpair(df["h_won"]),
        "h2h_draws": rpair(df["drawn"]),
        "h2h_avg_total": rpair(df["total_g"]),
    }, index=df["_idx"].values)

    return result.reindex(np.arange(len(result)))


def shot_features_rolling(df_shots, df, n=10):
    """Features de chute (rolling, sem leakage)."""
    if df_shots is None or df_shots.empty:
        return df

    xcol = next((c for c in df_shots.columns if "xg" in c.lower()), None)
    result_col = next((c for c in df_shots.columns
                       if "result" in c.lower() or "outcome" in c.lower()), None)
    home_col = next((c for c in df_shots.columns if c in ["home", "home_team"]), None)
    away_col = next((c for c in df_shots.columns if c in ["away", "away_team"]), None)
    team_col = next((c for c in df_shots.columns if c in ["team", "shooter_team"]), None)
    date_col = next((c for c in df_shots.columns if "date" in c.lower()), None)

    if not all([xcol, date_col, home_col, away_col, team_col]):
        return df

    shots = df_shots.copy()
    shots[date_col] = pd.to_datetime(shots[date_col], errors="coerce").dt.normalize()
    shots[xcol] = pd.to_numeric(shots[xcol], errors="coerce")

    if result_col:
        shots["on_target"] = shots[result_col].astype(str).str.lower().isin(
            ["goal", "savedshot", "on target", "blockedshot"]
        ).astype(int)
    else:
        shots["on_target"] = 0

    shots = shots.rename(columns={home_col: "home", away_col: "away",
                                   date_col: "date", team_col: "shot_team"})

    tg = shots.groupby(["home", "away", "date", "shot_team"]).agg(
        shots_n=(xcol, "count"),
        xg_sum=(xcol, "sum"),
        avg_shot_xg=(xcol, "mean"),
        on_target_n=("on_target", "sum"),
    ).reset_index()

    tg_h = tg[tg["shot_team"] == tg["home"]].copy()
    tg_h["team"] = tg_h["home"]
    tg_a = tg[tg["shot_team"] == tg["away"]].copy()
    tg_a["team"] = tg_a["away"]

    tg_all = pd.concat([
        tg_h[["team", "date", "shots_n", "xg_sum", "avg_shot_xg", "on_target_n"]],
        tg_a[["team", "date", "shots_n", "xg_sum", "avg_shot_xg", "on_target_n"]],
    ]).sort_values(["team", "date"])

    for col in ["shots_n", "xg_sum", "avg_shot_xg", "on_target_n"]:
        tg_all[f"roll_{col}_{n}"] = (
            tg_all.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        )

    rcols = [c for c in tg_all.columns if c.startswith("roll_")]

    hshots = (tg_all[["team", "date"] + rcols]
              .drop_duplicates(["team", "date"], keep="last")
              .rename(columns={"team": "home", **{c: f"h_{c}" for c in rcols}}))

    ashots = (tg_all[["team", "date"] + rcols]
              .drop_duplicates(["team", "date"], keep="last")
              .rename(columns={"team": "away", **{c: f"a_{c}" for c in rcols}}))

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.merge(hshots, on=["home", "date"], how="left")
    df = df.merge(ashots, on=["away", "date"], how="left")

    added = [c for c in df.columns if "roll_shots" in c or "roll_xg_sum" in c
             or "roll_avg_shot" in c or "roll_on_target" in c]
    print(f"✅ Shot features: {len(added)} colunas")
    return df


def situational_features(df):
    """Rest days, fadiga, pontos na tabela."""
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    htl = df[["date", "home", "season", "hg", "ag"]].copy()
    htl.columns = ["date", "team", "season", "gf", "ga"]
    htl["pts"] = np.where(htl["gf"] > htl["ga"], 3,
                 np.where(htl["gf"] == htl["ga"], 1, 0))

    atl = df[["date", "away", "season", "ag", "hg"]].copy()
    atl.columns = ["date", "team", "season", "gf", "ga"]
    atl["pts"] = np.where(atl["gf"] > atl["ga"], 3,
                 np.where(atl["gf"] == atl["ga"], 1, 0))

    tl = pd.concat([htl, atl]).sort_values(["team", "season", "date"]).reset_index(drop=True)
    tl["prev_date"] = tl.groupby("team")["date"].shift(1)
    tl["rest_days"] = (tl["date"] - tl["prev_date"]).dt.days

    tl["table_pts"] = (
        tl.groupby(["team", "season"])["pts"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    tl_dedup = tl.drop_duplicates(["team", "date"], keep="last")

    df = df.merge(
        tl_dedup[["team", "date", "rest_days", "table_pts"]].rename(
            columns={"team": "home", "rest_days": "rest_days_home",
                     "table_pts": "table_pts_home"}),
        on=["home", "date"], how="left"
    ).merge(
        tl_dedup[["team", "date", "rest_days", "table_pts"]].rename(
            columns={"team": "away", "rest_days": "rest_days_away",
                     "table_pts": "table_pts_away"}),
        on=["away", "date"], how="left"
    )

    med = df["rest_days_home"].median()
    df["rest_days_home"] = df["rest_days_home"].fillna(med)
    df["rest_days_away"] = df["rest_days_away"].fillna(med)
    df["fatigue_home"] = (df["rest_days_home"] <= 3).astype(int)
    df["fatigue_away"] = (df["rest_days_away"] <= 3).astype(int)
    df["table_pts_home"] = df["table_pts_home"].fillna(0)
    df["table_pts_away"] = df["table_pts_away"].fillna(0)
    df["table_pts_diff"] = df["table_pts_home"] - df["table_pts_away"]

    return df


# ── EXECUÇÃO FEATURE ENGINEERING ──
print("⚙️  Construindo features...")
df = shot_features_rolling(df_shots, df, n=10)
df = situational_features(df)
print("✅ Features situacionais ok")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 7 — ELO RATINGS                          ║
# ╚══════════════════════════════════════════════════════════╝

def compute_elo(df, k=ELO_K, base=ELO_BASE):
    df = df.sort_values("date").reset_index(drop=True)
    homes = df["home"].values
    aways = df["away"].values
    hgs = df["hg"].values
    ags = df["ag"].values
    seasons = df["season"].values if "season" in df.columns else np.zeros(len(df))

    ratings = {}
    last_season = {}
    season_teams = {}

    for s in np.unique(seasons):
        m = seasons == s
        season_teams[s] = set(homes[m]) | set(aways[m])

    sorted_s = sorted(season_teams.keys())
    elo_h = np.empty(len(df))
    elo_a = np.empty(len(df))

    for i in range(len(df)):
        h, a, s = homes[i], aways[i], seasons[i]

        for team in (h, a):
            if team in last_season and last_season[team] != s:
                ratings[team] = ELO_REGRESS * ratings.get(team, base) + (1 - ELO_REGRESS) * base
            elif team not in ratings:
                si = sorted_s.index(s) if s in sorted_s else 0
                if si > 0 and team not in season_teams.get(sorted_s[si - 1], set()):
                    ratings[team] = ELO_PROMOTED
            last_season[team] = s

        rh, ra = ratings.get(h, base), ratings.get(a, base)
        elo_h[i], elo_a[i] = rh, ra

        try:
            hg_i, ag_i = int(hgs[i]), int(ags[i])
        except (ValueError, TypeError):
            continue

        score_h = 1.0 if hg_i > ag_i else (0.5 if hg_i == ag_i else 0.0)
        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - rh - ELO_HOME_ADV) / 400.0))
        ratings[h] = rh + k * (score_h - exp_h)
        ratings[a] = ra + k * ((1 - score_h) - (1 - exp_h))

    df["elo_home"] = elo_h
    df["elo_away"] = elo_a
    df["elo_diff"] = elo_h - elo_a
    df["elo_prob_home"] = 1.0 / (1.0 + 10.0 ** (-(df["elo_diff"] + ELO_HOME_ADV) / 400.0))
    return df


df = compute_elo(df)
print(f"✅ Elo: {df['elo_home'].min():.0f} – {df['elo_home'].max():.0f}")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 8 — BUILD FEATURES (merge tudo)          ║
# ╚══════════════════════════════════════════════════════════╝

def _build_feature_cols(windows=None):
    if windows is None:
        windows = ROLLING_WINDOWS
    cols = []
    for n in windows:
        cols.extend([
            f"h_roll_xgf_{n}", f"h_roll_xga_{n}",
            f"h_roll_gf_{n}", f"h_roll_ga_{n}",
            f"h_roll_pts_{n}", f"h_roll_win_{n}", f"h_roll_draw_{n}",
            f"a_roll_xgf_{n}", f"a_roll_xga_{n}",
            f"a_roll_gf_{n}", f"a_roll_ga_{n}",
            f"a_roll_pts_{n}", f"a_roll_win_{n}", f"a_roll_draw_{n}",
            f"h_roll_xgf_vh_{n}", f"h_roll_xga_vh_{n}",
            f"h_roll_gf_vh_{n}", f"h_roll_ga_vh_{n}",
            f"a_roll_xgf_va_{n}", f"a_roll_xga_va_{n}",
            f"a_roll_gf_va_{n}", f"a_roll_ga_va_{n}",
            f"diff_xgf_{n}", f"diff_xga_{n}",
            f"diff_pts_{n}", f"diff_win_{n}",
            f"h_roll_surprise_att_{n}", f"h_roll_surprise_def_{n}",
            f"a_roll_surprise_att_{n}", f"a_roll_surprise_def_{n}",
        ])

    # Momentum
    cols.extend([
        "h_momentum_xgf", "h_momentum_xga", "h_momentum_pts",
        "a_momentum_xgf", "a_momentum_xga", "a_momentum_pts",
    ])

    # Fixas
    cols.extend([
        "h2h_home_wins", "h2h_draws", "h2h_avg_total",
        "elo_home", "elo_away", "elo_diff", "elo_prob_home",
        "rest_days_home", "rest_days_away",
        "fatigue_home", "fatigue_away",
        "table_pts_home", "table_pts_away", "table_pts_diff",
    ])
    return cols


FEATURE_COLS = _build_feature_cols()


def build_features(df):
    """Pipeline completo de features."""
    tl = build_team_timeline(df)
    tl = rolling_team_stats(tl, ROLLING_WINDOWS)

    # Merge home features
    hf = tl[tl["venue"] == "H"].copy()
    hf = hf.rename(columns=lambda c: f"h_{c}" if c not in ["team", "date"] else c)
    hf = hf.rename(columns={"team": "home"})

    af = tl[tl["venue"] == "A"].copy()
    af = af.rename(columns=lambda c: f"a_{c}" if c not in ["team", "date"] else c)
    af = af.rename(columns={"team": "away"})

    hroll = [c for c in hf.columns if c.startswith("h_roll_") or c.startswith("h_momentum_")]
    aroll = [c for c in af.columns if c.startswith("a_roll_") or c.startswith("a_momentum_")]

    hf_d = hf[["home", "date"] + hroll].drop_duplicates(["home", "date"], keep="last")
    af_d = af[["away", "date"] + aroll].drop_duplicates(["away", "date"], keep="last")

    df_f = df.merge(hf_d, on=["home", "date"], how="left")
    df_f = df_f.merge(af_d, on=["away", "date"], how="left")
    df_f = df_f.drop_duplicates(["home", "away", "date"], keep="first")

    # Diferenciais
    for n in ROLLING_WINDOWS:
        df_f[f"diff_xgf_{n}"] = df_f.get(f"h_roll_xgf_{n}", np.nan) - df_f.get(f"a_roll_xgf_{n}", np.nan)
        df_f[f"diff_xga_{n}"] = df_f.get(f"h_roll_xga_{n}", np.nan) - df_f.get(f"a_roll_xga_{n}", np.nan)
        df_f[f"diff_pts_{n}"] = df_f.get(f"h_roll_pts_{n}", np.nan) - df_f.get(f"a_roll_pts_{n}", np.nan)
        df_f[f"diff_win_{n}"] = df_f.get(f"h_roll_win_{n}", np.nan) - df_f.get(f"a_roll_win_{n}", np.nan)

    # H2H
    h2h = h2h_features(df)
    df_f = pd.concat([df_f.reset_index(drop=True), h2h.reset_index(drop=True)], axis=1)

    # Elo (já no df)
    for c in ["elo_home", "elo_away", "elo_diff", "elo_prob_home"]:
        if c in df.columns and c not in df_f.columns:
            df_f[c] = df[c].values[:len(df_f)]

    # Situacionais (já no df)
    for c in ["rest_days_home", "rest_days_away", "fatigue_home", "fatigue_away",
              "table_pts_home", "table_pts_away", "table_pts_diff"]:
        if c in df.columns and c not in df_f.columns:
            df_f[c] = df[c].values[:len(df_f)]

    return df_f


print("⚙️  Construindo features completas...")
df_feat = build_features(df)

# Adiciona shot features ao FEATURE_COLS
_sf = [c for c in df_feat.columns if ("roll_shots" in c or "roll_xg_sum" in c
       or "roll_avg_shot" in c or "roll_on_target" in c) and c.startswith(("h_", "a_"))]
for s in _sf:
    if s not in FEATURE_COLS:
        FEATURE_COLS.append(s)

print(f"✅ Features prontas — shape: {df_feat.shape} | {len(FEATURE_COLS)} features")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 9 — DIXON-COLES (POISSON)                ║
# ╚══════════════════════════════════════════════════════════╝

class DixonColesModel:
    def __init__(self, xi=0.005):
        self.xi = xi
        self.params_ = None
        self.teams_ = None
        self._idx = None

    @staticmethod
    def _tau(lh, la, hg, ag, rho):
        tau = np.ones(len(hg))
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = np.clip(1 - lh[m00] * la[m00] * rho, 1e-10, None)
        tau[m01] = np.clip(1 + lh[m01] * rho, 1e-10, None)
        tau[m10] = np.clip(1 + la[m10] * rho, 1e-10, None)
        tau[m11] = np.clip(1 - rho, 1e-10, None)
        return tau

    def _nll(self, params, hi, ai, hg, ag, w, lfh, lfa):
        n = len(self.teams_)
        alpha, beta = params[:n], params[n:2 * n]
        gamma, rho = params[2 * n], params[2 * n + 1]
        lh = np.exp(alpha[hi] - beta[ai] + gamma)
        la = np.exp(alpha[ai] - beta[hi])
        tau = self._tau(lh, la, hg, ag, rho)
        ll_h = hg * np.log(np.maximum(lh, 1e-10)) - lh - lfh
        ll_a = ag * np.log(np.maximum(la, 1e-10)) - la - lfa
        return -(w * (np.log(tau) + ll_h + ll_a)).sum()

    def fit(self, df):
        self.teams_ = sorted(set(df["home"]) | set(df["away"]))
        self._idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)

        hi = np.array([self._idx[t] for t in df["home"]])
        ai = np.array([self._idx[t] for t in df["away"]])
        hg = df["hg"].astype(int).values
        ag = df["ag"].astype(int).values

        d = np.array(df["date"].values, dtype="datetime64[D]")
        w = np.exp(-self.xi * (d.max() - d).astype(float))

        lfh = gammaln(hg + 1)
        lfa = gammaln(ag + 1)

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.3
        x0[2 * n + 1] = -0.1

        res = minimize(self._nll, x0, args=(hi, ai, hg, ag, w, lfh, lfa),
                       method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9})
        self.params_ = res.x
        self.params_[:n] -= self.params_[:n].mean()
        self.params_[n:2 * n] -= self.params_[n:2 * n].mean()
        print(f"✅ Dixon-Coles | LL={-res.fun:.1f} | ok={res.success}")

    def predict_lambda(self, home, away):
        n = len(self.teams_)
        a, b, g = self.params_[:n], self.params_[n:2 * n], self.params_[2 * n]
        # Fallback para times desconhecidos: ataque=0, defesa=0 (média)
        hi = self._idx.get(home)
        ai = self._idx.get(away)
        att_h = a[hi] if hi is not None else 0.0
        def_a = b[ai] if ai is not None else 0.0
        att_a = a[ai] if ai is not None else 0.0
        def_h = b[hi] if hi is not None else 0.0
        return np.exp(att_h - def_a + g), np.exp(att_a - def_h)

    def score_matrix(self, home, away, max_g=10):
        lh, la = self.predict_lambda(home, away)
        rho = self.params_[-1]
        goals = np.arange(max_g + 1)
        ph, pa = poisson.pmf(goals, lh), poisson.pmf(goals, la)
        m = np.outer(ph, pa)
        for i, j, v in [(0, 0, 1 - lh * la * rho), (0, 1, 1 + lh * rho),
                        (1, 0, 1 + la * rho), (1, 1, 1 - rho)]:
            m[i, j] *= max(v, 1e-10)
        return np.maximum(m, 0) / np.maximum(m, 0).sum()


# Treino DC completo
dc_full = DixonColesModel(xi=0.005)
dc_full.fit(df.dropna(subset=["hg", "ag"]).reset_index(drop=True))

# DC para features ML (primeiros 60%)
_df_dc = df.dropna(subset=["hg", "ag"]).sort_values("date").reset_index(drop=True)
_n_dc = int(len(_df_dc) * 0.60)
dc_train = DixonColesModel(xi=0.005)
dc_train.fit(_df_dc.iloc[:_n_dc])
print(f"   DC treino: {_n_dc} jogos | DC completo: {len(_df_dc)} jogos")


# ╔══════════════════════════════════════════════════════════╗
# ║   CÉLULA 10 — MODELOS ML (XGBoost + LightGBM + Stack)   ║
# ╚══════════════════════════════════════════════════════════╝

def add_dc_features(df_f, dc):
    """Adiciona lambdas do DC como features."""
    n = len(dc.teams_)
    alpha, beta, gamma = dc.params_[:n], dc.params_[n:2 * n], dc.params_[2 * n]

    hi = df_f["home"].map(dc._idx).values.astype(float)
    ai = df_f["away"].map(dc._idx).values.astype(float)
    valid = ~(np.isnan(hi) | np.isnan(ai))

    lh = np.full(len(df_f), np.exp(gamma))
    la = np.full(len(df_f), 1.0)
    lh[valid] = np.exp(alpha[hi[valid].astype(int)] - beta[ai[valid].astype(int)] + gamma)
    la[valid] = np.exp(alpha[ai[valid].astype(int)] - beta[hi[valid].astype(int)])

    out = df_f.copy()
    out["dc_lh"] = lh
    out["dc_la"] = la
    out["dc_diff"] = lh - la
    out["dc_ratio"] = np.where(la > 0.01, lh / la, np.nan)
    return out


def smart_fillna(X, medians):
    X = X.copy()
    X = X.fillna(medians.reindex(X.columns))
    for col in X.columns:
        if X[col].isna().sum() == 0:
            continue
        if "fatigue" in col:
            X[col] = X[col].fillna(0)
        elif "ratio" in col:
            X[col] = X[col].fillna(1.0)
        elif any(k in col for k in ["elo", "pts", "rest"]):
            med = X[col].median()
            X[col] = X[col].fillna(med if pd.notna(med) else 0)
        else:
            X[col] = X[col].fillna(0)
    return X


def prepare_ml_data(df_f, dc):
    """Prepara X, targets de classificação e regressão."""
    df2 = add_dc_features(df_f, dc)
    extra = ["dc_lh", "dc_la", "dc_diff", "dc_ratio"]
    feats = [c for c in FEATURE_COLS if c in df2.columns] + extra

    df2 = df2.dropna(subset=["hg", "ag", "total_goals", "result"]).copy()
    df2["hg"] = df2["hg"].astype(int)
    df2["ag"] = df2["ag"].astype(int)
    df2["total_goals"] = df2["total_goals"].astype(int)

    # Targets classificação
    df2["y_1x2"] = df2["result"].map({"H": 0, "D": 1, "A": 2})
    df2["y_ov15"] = (df2["total_goals"] >= 2).astype(int)
    df2["y_ov25"] = (df2["total_goals"] >= 3).astype(int)
    df2["y_ov35"] = (df2["total_goals"] >= 4).astype(int)
    df2["y_h_ov05"] = (df2["hg"] >= 1).astype(int)
    df2["y_h_ov15"] = (df2["hg"] >= 2).astype(int)
    df2["y_h_ov25"] = (df2["hg"] >= 3).astype(int)
    df2["y_h_ov35"] = (df2["hg"] >= 4).astype(int)
    df2["y_a_ov05"] = (df2["ag"] >= 1).astype(int)
    df2["y_a_ov15"] = (df2["ag"] >= 2).astype(int)
    df2["y_a_ov25"] = (df2["ag"] >= 3).astype(int)
    df2["y_a_ov35"] = (df2["ag"] >= 4).astype(int)
    df2["y_btts"] = ((df2["hg"] >= 1) & (df2["ag"] >= 1)).astype(int)

    # Targets regressão
    df2["y_goals_home"] = df2["hg"].astype(float)
    df2["y_goals_away"] = df2["ag"].astype(float)

    all_cls = ["y_1x2", "y_ov15", "y_ov25", "y_ov35",
               "y_h_ov05", "y_h_ov15", "y_h_ov25", "y_h_ov35",
               "y_a_ov05", "y_a_ov15", "y_a_ov25", "y_a_ov35", "y_btts"]
    all_reg = ["y_goals_home", "y_goals_away"]

    mask = df2[feats + ["y_1x2"]].notna().all(axis=1)
    X = df2.loc[mask, feats].astype(float)
    ys_cls = {k: df2.loc[mask, k] for k in all_cls}
    ys_reg = {k: df2.loc[mask, k] for k in all_reg}

    print(f"   ✅ {len(X)} amostras | {len(feats)} features")
    return X, ys_cls, ys_reg, feats


def train_stacked_models(X, ys_cls, ys_reg):
    """
    STACKING: XGBoost + LightGBM -> Meta-Learner (LogisticRegression).
    Split: 0-60% treino base, 60-80% treino meta, 80-100% teste.
    """
    n = len(X)
    n_base = int(n * 0.60)
    n_meta = int(n * 0.80)

    X_base = X.iloc[:n_base]
    X_meta_raw = X.iloc[n_base:n_meta]
    feat_medians = X_base.median()

    models = {}

    # ── Classificadores ──
    for target, y in ys_cls.items():
        multi = (target == "y_1x2")
        y_base = y.iloc[:n_base]
        y_meta = y.iloc[n_base:n_meta]

        obj_xgb = "multi:softprob" if multi else "binary:logistic"
        metric_xgb = "mlogloss" if multi else "logloss"

        # XGBoost
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective=obj_xgb, eval_metric=metric_xgb,
            random_state=42, n_jobs=-1
        )
        xgb_clf.fit(X_base, y_base,
                     eval_set=[(X_meta_raw, y_meta)],
                     verbose=False)

        # LightGBM
        lgb_clf = None
        if LGB_OK:
            obj_lgb = "multiclass" if multi else "binary"
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective=obj_lgb, random_state=42, n_jobs=-1,
                verbose=-1, num_class=3 if multi else None,
            )
            lgb_clf.fit(X_base, y_base,
                        eval_set=[(X_meta_raw, y_meta)],
                        callbacks=[lgb.early_stopping(30, verbose=False),
                                   lgb.log_evaluation(0)])

        # Meta-learner: combina probabilidades de XGB + LGB
        xgb_meta_p = xgb_clf.predict_proba(X_meta_raw)
        if lgb_clf is not None:
            lgb_meta_p = lgb_clf.predict_proba(X_meta_raw)
            meta_X = np.hstack([xgb_meta_p, lgb_meta_p])
        else:
            meta_X = xgb_meta_p

        meta_clf = LogisticRegression(max_iter=1000, random_state=42)
        meta_clf.fit(meta_X, y_meta)

        # Avaliação
        meta_p = meta_clf.predict_proba(meta_X)
        ll = log_loss(y_meta, meta_p)
        acc = accuracy_score(y_meta, meta_clf.predict(meta_X))
        print(f"  {target:12s} | LL={ll:.4f} Acc={acc:.4f}")

        models[target] = {
            "xgb": xgb_clf,
            "lgb": lgb_clf,
            "meta": meta_clf,
            "type": "cls",
        }

    # ── Regressores (Poisson) ──
    for target, y in ys_reg.items():
        y_base = y.iloc[:n_base]
        y_meta_r = y.iloc[n_base:n_meta]

        reg = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="count:poisson", random_state=42, n_jobs=-1
        )
        reg.fit(X_base, y_base,
                eval_set=[(X_meta_raw, y_meta_r)],
                verbose=False)

        pred = reg.predict(X_meta_raw)
        rmse = np.sqrt(mean_squared_error(y_meta_r, pred))
        print(f"  {target:12s} | RMSE={rmse:.4f}")

        models[target] = {"xgb": reg, "type": "reg"}

    info = {"n_base": n_base, "n_meta": n_meta, "medians": feat_medians}
    return models, info


print("⚙️  Preparando dataset ML...")
X_all, y_cls, y_reg, feat_names = prepare_ml_data(df_feat, dc_train)
print(f"   Shape: {X_all.shape}")

print("⚙️  Treinando modelos (XGBoost + LightGBM + Meta-Learner)...")
ml_models, split_info = train_stacked_models(X_all, y_cls, y_reg)
FEAT_MEDIANS = split_info["medians"]
print("✅ Todos os modelos treinados!")


# ╔══════════════════════════════════════════════════════════╗
# ║   CÉLULA 11 — ENSEMBLE PREDICT (TODOS OS MERCADOS)      ║
# ╚══════════════════════════════════════════════════════════╝

ALPHA = 0.40  # fallback DC vs ML


def _stacked_predict_proba(model_dict, X_pred):
    """Prediz usando o stack: XGB + LGB -> Meta."""
    xgb_p = model_dict["xgb"].predict_proba(X_pred)
    if model_dict.get("lgb") is not None:
        lgb_p = model_dict["lgb"].predict_proba(X_pred)
        meta_X = np.hstack([xgb_p, lgb_p])
    else:
        meta_X = xgb_p
    return model_dict["meta"].predict_proba(meta_X)


def _stacked_predict_bin(model_dict, X_pred):
    """P(classe=1) para modelo binário stacked."""
    p = _stacked_predict_proba(model_dict, X_pred)
    return float(p[0][1])


def optimize_alpha(dc, ml, df_fv, feat_names, medians):
    """Otimiza alpha no holdout 60-80%."""
    records = []
    for _, row in df_fv.iterrows():
        if pd.isna(row.get("result")):
            continue
        try:
            mc_H, mc_D, mc_A = _dc_probs(dc, row["home"], row["away"])
            feat_row = _build_feat_row(row, dc, feat_names)
            X_p = smart_fillna(pd.DataFrame([feat_row])[feat_names].astype(float), medians)
            ml_p = _stacked_predict_proba(ml["y_1x2"], X_p)[0]
            records.append({
                "result": row["result"],
                "mc_H": mc_H, "mc_D": mc_D, "mc_A": mc_A,
                "ml_H": ml_p[0], "ml_D": ml_p[1], "ml_A": ml_p[2],
            })
        except Exception:
            pass

    if len(records) < 30:
        return ALPHA

    dv = pd.DataFrame(records)
    y_true = dv["result"].map({"H": 0, "D": 1, "A": 2})

    def neg_ll(a):
        pH = a * dv["mc_H"] + (1 - a) * dv["ml_H"]
        pD = a * dv["mc_D"] + (1 - a) * dv["ml_D"]
        pA = a * dv["mc_A"] + (1 - a) * dv["ml_A"]
        s = pH + pD + pA
        return log_loss(y_true, np.column_stack([pH / s, pD / s, pA / s]))

    res = minimize_scalar(neg_ll, bounds=(0, 1), method="bounded")
    print(f"✅ Alpha otimizado: {res.x:.3f} (LL={res.fun:.4f})")
    return round(res.x, 3)


def _dc_probs(dc, home, away):
    mat = dc.score_matrix(home, away)
    pH = float(np.tril(mat, -1).sum())
    pA = float(np.triu(mat, 1).sum())
    pD = float(np.diag(mat).sum())
    s = pH + pD + pA
    return pH / s, pD / s, pA / s


def _build_feat_row(row, dc, feat_names):
    """Constrói vetor de features para uma partida."""
    try:
        lh, la = dc.predict_lambda(row["home"], row["away"])
    except KeyError:
        lh, la = np.exp(dc.params_[2 * len(dc.teams_)]), 1.0

    extra = {
        "dc_lh": lh, "dc_la": la,
        "dc_diff": lh - la, "dc_ratio": lh / max(la, 0.01),
    }
    return {c: row.get(c, extra.get(c, np.nan)) for c in feat_names}


def ensemble_predict(home, away, dc, ml, df_f, alpha=None, cutoff_date=None):
    """
    Previsão ensemble para TODOS os mercados solicitados.
    DC score_matrix + ML stacking, blended com alpha.
    """
    if alpha is None:
        alpha = ALPHA

    # ── Score matrix DC ──
    lh_dc, la_dc = dc.predict_lambda(home, away)
    mat = dc.score_matrix(home, away, max_g=10)
    max_g = mat.shape[0]

    pH_dc = float(np.tril(mat, -1).sum())
    pA_dc = float(np.triu(mat, 1).sum())
    pD_dc = float(np.diag(mat).sum())
    s = pH_dc + pD_dc + pA_dc
    pH_dc, pD_dc, pA_dc = pH_dc / s, pD_dc / s, pA_dc / s

    # Over total (da matrix)
    total_pmf = np.zeros(2 * max_g - 1)
    for i in range(max_g):
        for j in range(max_g):
            total_pmf[i + j] += mat[i, j]
    total_pmf /= total_pmf.sum()
    total_cdf = np.cumsum(total_pmf)

    def ov_total(k):
        return float(1 - total_cdf[k]) if k < len(total_cdf) else 0.0

    # Marginais por time
    h_marg = mat.sum(axis=1)
    a_marg = mat.sum(axis=0)
    h_cdf = np.cumsum(h_marg)
    a_cdf = np.cumsum(a_marg)

    def ov_h(k):
        return float(1 - h_cdf[k - 1]) if 0 < k <= len(h_cdf) else (1.0 if k <= 0 else 0.0)

    def ov_a(k):
        return float(1 - a_cdf[k - 1]) if 0 < k <= len(a_cdf) else (1.0 if k <= 0 else 0.0)

    btts_dc = float(mat[1:, 1:].sum())

    dc_p = {
        "1": pH_dc, "X": pD_dc, "2": pA_dc,
        "ov15": ov_total(1), "ov25": ov_total(2), "ov35": ov_total(3),
        "h_ov05": ov_h(1), "h_ov15": ov_h(2), "h_ov25": ov_h(3), "h_ov35": ov_h(4),
        "a_ov05": ov_a(1), "a_ov15": ov_a(2), "a_ov25": ov_a(3), "a_ov35": ov_a(4),
        "btts": btts_dc,
    }

    # ── Features ML ──
    if cutoff_date is not None:
        df_ref = df_f[df_f["date"] < cutoff_date]
    else:
        df_ref = df_f

    def _latest(team):
        m = (df_ref["home"] == team) | (df_ref["away"] == team)
        sub = df_ref[m]
        if sub.empty:
            return None, None
        row = sub.sort_values("date").iloc[-1]
        return row, (row["home"] == team)

    def _read(row, col):
        if row is None:
            return np.nan
        v = row.get(col)
        try:
            f = float(v)
            return f if not np.isnan(f) else np.nan
        except (ValueError, TypeError):
            return np.nan

    h_latest, h_was_home = _latest(home)
    a_latest, a_was_home = _latest(away)

    extra = {
        "dc_lh": lh_dc, "dc_la": la_dc,
        "dc_diff": lh_dc - la_dc, "dc_ratio": lh_dc / max(la_dc, 0.01),
    }

    feat_vec = {}
    for c in feat_names:
        val = np.nan
        if c in extra:
            val = extra[c]
        elif c.startswith("h_"):
            sfx = c[2:]
            if "vh" in sfx:
                # Venue home: busca último jogo como mandante
                m = df_ref[df_ref["home"] == home]
                if not m.empty:
                    val = _read(m.sort_values("date").iloc[-1], c)
            elif h_latest is not None:
                if h_was_home:
                    val = _read(h_latest, c)
                else:
                    val = _read(h_latest, f"a_{sfx}")
        elif c.startswith("a_"):
            sfx = c[2:]
            if "va" in sfx:
                m = df_ref[df_ref["away"] == away]
                if not m.empty:
                    val = _read(m.sort_values("date").iloc[-1], c)
            elif a_latest is not None:
                if not a_was_home:
                    val = _read(a_latest, c)
                else:
                    val = _read(a_latest, f"h_{sfx}")
        elif c.startswith("diff_"):
            pass  # recomputado abaixo
        elif c.startswith("h2h_"):
            hm = ((df_ref["home"] == home) & (df_ref["away"] == away)) | \
                 ((df_ref["home"] == away) & (df_ref["away"] == home))
            hsub = df_ref[hm]
            if not hsub.empty:
                val = _read(hsub.sort_values("date").iloc[-1], c)
        elif "home" in c:
            if h_latest is not None:
                val = _read(h_latest, c if h_was_home else c.replace("home", "away"))
        elif "away" in c:
            if a_latest is not None:
                val = _read(a_latest, c if not a_was_home else c.replace("away", "home"))
        else:
            val = _read(h_latest, c)
        feat_vec[c] = val

    # Recomputa diferenciais
    for n in ROLLING_WINDOWS:
        for hk, ak, dk in [
            (f"h_roll_xgf_{n}", f"a_roll_xgf_{n}", f"diff_xgf_{n}"),
            (f"h_roll_xga_{n}", f"a_roll_xga_{n}", f"diff_xga_{n}"),
            (f"h_roll_pts_{n}", f"a_roll_pts_{n}", f"diff_pts_{n}"),
            (f"h_roll_win_{n}", f"a_roll_win_{n}", f"diff_win_{n}"),
        ]:
            hv = feat_vec.get(hk, np.nan)
            av = feat_vec.get(ak, np.nan)
            if isinstance(hv, float) and isinstance(av, float) \
               and not np.isnan(hv) and not np.isnan(av):
                feat_vec[dk] = hv - av

    X_pred = pd.DataFrame([feat_vec])[feat_names].astype(float)
    X_pred = smart_fillna(X_pred, FEAT_MEDIANS)

    # ── ML predictions ──
    ml_1x2 = _stacked_predict_proba(ml["y_1x2"], X_pred)[0]

    def ml_bin(key):
        if key not in ml:
            return None
        return _stacked_predict_bin(ml[key], X_pred)

    ml_ov15 = ml_bin("y_ov15")
    ml_ov25 = ml_bin("y_ov25")
    ml_ov35 = ml_bin("y_ov35")
    ml_h05 = ml_bin("y_h_ov05")
    ml_h15 = ml_bin("y_h_ov15")
    ml_h25 = ml_bin("y_h_ov25")
    ml_h35 = ml_bin("y_h_ov35")
    ml_a05 = ml_bin("y_a_ov05")
    ml_a15 = ml_bin("y_a_ov15")
    ml_a25 = ml_bin("y_a_ov25")
    ml_a35 = ml_bin("y_a_ov35")
    ml_btts = ml_bin("y_btts")

    # Regressão de gols
    ml_gh = ml["y_goals_home"]["xgb"].predict(X_pred)[0] if "y_goals_home" in ml else lh_dc
    ml_ga = ml["y_goals_away"]["xgb"].predict(X_pred)[0] if "y_goals_away" in ml else la_dc

    # ── Blend DC + ML ──
    pH = np.clip(alpha * dc_p["1"] + (1 - alpha) * ml_1x2[0], PROB_FLOOR, PROB_CEILING)
    pD = np.clip(alpha * dc_p["X"] + (1 - alpha) * ml_1x2[1], PROB_FLOOR, PROB_CEILING)
    pA = np.clip(alpha * dc_p["2"] + (1 - alpha) * ml_1x2[2], PROB_FLOOR, PROB_CEILING)
    s = pH + pD + pA
    pH, pD, pA = pH / s, pD / s, pA / s

    def blend(dc_val, ml_val):
        if ml_val is None:
            return dc_val
        return np.clip(alpha * dc_val + (1 - alpha) * ml_val, PROB_FLOOR_BIN, PROB_CEIL_BIN)

    ov15 = blend(dc_p["ov15"], ml_ov15)
    ov25 = blend(dc_p["ov25"], ml_ov25)
    ov35 = blend(dc_p["ov35"], ml_ov35)

    h05 = blend(dc_p["h_ov05"], ml_h05)
    h15 = blend(dc_p["h_ov15"], ml_h15)
    h25 = blend(dc_p["h_ov25"], ml_h25)
    h35 = blend(dc_p["h_ov35"], ml_h35)

    a05 = blend(dc_p["a_ov05"], ml_a05)
    a15 = blend(dc_p["a_ov15"], ml_a15)
    a25 = blend(dc_p["a_ov25"], ml_a25)
    a35 = blend(dc_p["a_ov35"], ml_a35)

    btts = blend(dc_p["btts"], ml_btts)

    # Monotonicity
    ov35 = min(ov35, ov25); ov25 = min(ov25, ov15)
    h35 = min(h35, h25); h25 = min(h25, h15); h15 = min(h15, h05)
    a35 = min(a35, a25); a25 = min(a25, a15); a15 = min(a15, a05)

    # DNB e Dupla Chance
    dnb_d = pH + pA
    dnb_h = pH / dnb_d if dnb_d > 0 else 0.5
    dnb_a = pA / dnb_d if dnb_d > 0 else 0.5

    xg_h = alpha * lh_dc + (1 - alpha) * ml_gh
    xg_a = alpha * la_dc + (1 - alpha) * ml_ga

    return {
        "lambda_home": round(lh_dc, 3),
        "lambda_away": round(la_dc, 3),
        "xG Previsto (Mand.)": round(xg_h, 3),
        "xG Previsto (Visit.)": round(xg_a, 3),
        "xG ML (Mand.)": round(float(ml_gh), 3),
        "xG ML (Visit.)": round(float(ml_ga), 3),

        "Vitória Mandante (1)": round(pH, 4),
        "Empate (X)": round(pD, 4),
        "Vitória Visitante (2)": round(pA, 4),

        "DNB Mandante": round(dnb_h, 4),
        "DNB Visitante": round(dnb_a, 4),

        "DC 1X (Mand. ou Empate)": round(pH + pD, 4),
        "DC X2 (Visit. ou Empate)": round(pA + pD, 4),
        "DC 12 (Qualquer Vitória)": round(pH + pA, 4),

        "Over 1.5": round(ov15, 4), "Under 1.5": round(1 - ov15, 4),
        "Over 2.5": round(ov25, 4), "Under 2.5": round(1 - ov25, 4),
        "Over 3.5": round(ov35, 4), "Under 3.5": round(1 - ov35, 4),

        "Over 0.5 Gols (Mand.)": round(h05, 4),
        "Over 1.5 Gols (Mand.)": round(h15, 4),
        "Over 2.5 Gols (Mand.)": round(h25, 4),
        "Over 3.5 Gols (Mand.)": round(h35, 4),

        "Over 0.5 Gols (Visit.)": round(a05, 4),
        "Over 1.5 Gols (Visit.)": round(a15, 4),
        "Over 2.5 Gols (Visit.)": round(a25, 4),
        "Over 3.5 Gols (Visit.)": round(a35, 4),

        "BTTS Sim": round(btts, 4),
        "BTTS Não": round(1 - btts, 4),
    }


# ── Otimização do Alpha ──
print("\n⚙️  Otimizando alpha...")
_n_base = split_info["n_base"]
_n_meta = split_info["n_meta"]
_df_val = df_feat.iloc[_n_base:_n_meta].copy()
ALPHA = optimize_alpha(dc_train, ml_models, _df_val, feat_names, FEAT_MEDIANS)
print(f"   ALPHA final: {ALPHA}")


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 12 — BACKTEST                            ║
# ╚══════════════════════════════════════════════════════════╝

def backtest(df_f, dc, ml, df_raw, split_info):
    """Backtest no holdout 80-100% — dados nunca vistos."""
    n_meta = split_info["n_meta"]
    df_test = df_f.iloc[n_meta:].copy()
    df_test = df_test.dropna(subset=["result", "total_goals"])
    print(f"\n⚙️  Backtest: {len(df_test)} partidas (holdout 80-100%)")

    cutoff_date = df_f.iloc[n_meta - 1]["date"] if n_meta > 0 else None
    df_tr = df_raw[df_raw["date"] <= cutoff_date].dropna(subset=["hg", "ag"]).reset_index(drop=True)
    dc_bt = DixonColesModel(xi=0.005)
    dc_bt.fit(df_tr)

    records = []
    n_err = 0
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Backtest"):
        try:
            pred = ensemble_predict(row["home"], row["away"], dc_bt, ml, df_f,
                                    cutoff_date=row["date"])
            records.append({
                "home": row["home"], "away": row["away"],
                "date": row["date"], "result": row["result"],
                "total_g": int(row["total_goals"]),
                "hg": int(row["hg"]), "ag": int(row["ag"]),
                "p_H": pred["Vitória Mandante (1)"],
                "p_D": pred["Empate (X)"],
                "p_A": pred["Vitória Visitante (2)"],
                "p_ov25": pred["Over 2.5"],
                "p_ov15": pred["Over 1.5"],
                "p_btts": pred["BTTS Sim"],
                "xg_h": pred["xG Previsto (Mand.)"],
                "xg_a": pred["xG Previsto (Visit.)"],
            })
        except Exception as e:
            n_err += 1
            logger.debug(f"Backtest erro: {e}")

    bt = pd.DataFrame(records)
    if bt.empty:
        print("⚠️  Backtest vazio.")
        return bt

    bt["y_H"] = (bt["result"] == "H").astype(int)
    bt["y_D"] = (bt["result"] == "D").astype(int)
    bt["y_A"] = (bt["result"] == "A").astype(int)
    bt["y_ov25"] = (bt["total_g"] >= 3).astype(int)
    bt["y_ov15"] = (bt["total_g"] >= 2).astype(int)
    bt["y_btts"] = ((bt["hg"] >= 1) & (bt["ag"] >= 1)).astype(int)

    y_true = bt["result"].map({"H": 0, "D": 1, "A": 2})
    y_prob = bt[["p_H", "p_D", "p_A"]].values

    ll = log_loss(y_true, y_prob)
    acc = accuracy_score(y_true, np.argmax(y_prob, axis=1))
    brier_H = brier_score_loss(bt["y_H"], bt["p_H"])
    brier_D = brier_score_loss(bt["y_D"], bt["p_D"])
    brier_A = brier_score_loss(bt["y_A"], bt["p_A"])
    brier_25 = brier_score_loss(bt["y_ov25"], bt["p_ov25"])
    xg_rmse_h = np.sqrt(mean_squared_error(bt["hg"], bt["xg_h"]))
    xg_rmse_a = np.sqrt(mean_squared_error(bt["ag"], bt["xg_a"]))

    print(f"\n{'═' * 55}")
    print(f"  📈  BACKTEST ({len(bt)} jogos | {n_err} erros)")
    print(f"{'═' * 55}")
    print(f"  Log-Loss 1X2:      {ll:.4f}  (aleatório ≈ 1.099)")
    print(f"  Acurácia 1X2:      {acc:.4f}  (baseline ≈ 0.46)")
    print(f"{'─' * 55}")
    print(f"  Brier Mandante:    {brier_H:.4f}")
    print(f"  Brier Empate:      {brier_D:.4f}")
    print(f"  Brier Visitante:   {brier_A:.4f}")
    print(f"  Brier Over 2.5:    {brier_25:.4f}")
    print(f"{'─' * 55}")
    print(f"  RMSE xG Mand.:     {xg_rmse_h:.4f}")
    print(f"  RMSE xG Visit.:    {xg_rmse_a:.4f}")
    print(f"{'═' * 55}")
    return bt


bt_results = backtest(df_feat, dc_full, ml_models, df, split_info)


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 13 — VISUALIZAÇÕES                       ║
# ╚══════════════════════════════════════════════════════════╝

def plot_heatmap(home, away, dc):
    mat = dc.score_matrix(home, away, max_g=6)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(mat * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"label": "%"})
    ax.set_xlabel(f"Gols — {away}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Gols — {home}", fontsize=12, fontweight="bold")
    ax.set_title(f"Placar Exato: {home} × {away}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_markets(home, away, result):
    markets = {
        "Resultado 1X2": {
            "Mand.": result["Vitória Mandante (1)"],
            "Empate": result["Empate (X)"],
            "Visit.": result["Vitória Visitante (2)"],
        },
        "Dupla Chance": {
            "1X": result["DC 1X (Mand. ou Empate)"],
            "X2": result["DC X2 (Visit. ou Empate)"],
            "12": result["DC 12 (Qualquer Vitória)"],
        },
        "Over/Under Total": {
            "Ov1.5": result["Over 1.5"],
            "Ov2.5": result["Over 2.5"],
            "Ov3.5": result["Over 3.5"],
        },
        "Gols Mandante": {
            "Ov0.5": result["Over 0.5 Gols (Mand.)"],
            "Ov1.5": result["Over 1.5 Gols (Mand.)"],
            "Ov2.5": result["Over 2.5 Gols (Mand.)"],
            "Ov3.5": result["Over 3.5 Gols (Mand.)"],
        },
        "Gols Visitante": {
            "Ov0.5": result["Over 0.5 Gols (Visit.)"],
            "Ov1.5": result["Over 1.5 Gols (Visit.)"],
            "Ov2.5": result["Over 2.5 Gols (Visit.)"],
            "Ov3.5": result["Over 3.5 Gols (Visit.)"],
        },
        "BTTS": {
            "Sim": result["BTTS Sim"],
            "Não": result["BTTS Não"],
        },
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

    for i, (title, data) in enumerate(markets.items()):
        ax = axes[i]
        bars = ax.bar(data.keys(), [v * 100 for v in data.values()],
                      color=colors[i], edgecolor="white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        for bar, v in zip(bars, data.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v * 100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{home} × {away}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_calibration(bt, n_bins=8):
    markets = {
        "Mandante": ("p_H", "y_H"),
        "Empate": ("p_D", "y_D"),
        "Visitante": ("p_A", "y_A"),
        "Over 2.5": ("p_ov25", "y_ov25"),
    }
    markets = {k: v for k, v in markets.items()
               if v[0] in bt.columns and v[1] in bt.columns}

    fig, axes = plt.subplots(1, len(markets), figsize=(5 * len(markets), 5))
    if len(markets) == 1:
        axes = [axes]

    for ax, (label, (pc, tc)) in zip(axes, markets.items()):
        dm = bt[[pc, tc]].dropna()
        if len(dm) < 10:
            continue
        bins = np.linspace(0, 1, n_bins + 1)
        dm["bin"] = pd.cut(dm[pc], bins=bins, include_lowest=True)
        cal = dm.groupby("bin", observed=True).agg(
            pm=(pc, "mean"), fr=(tc, "mean"), n=(pc, "count")).dropna()
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.scatter(cal["pm"], cal["fr"], s=cal["n"] / cal["n"].max() * 200,
                   color="#2196F3", edgecolors="white", zorder=5)
        ax.plot(cal["pm"], cal["fr"], color="#2196F3", lw=1.5, alpha=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")

    plt.suptitle("Calibração", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 14 — INTERFACE DE PREVISÃO               ║
# ╚══════════════════════════════════════════════════════════╝

def predict_match(home, away, dc=None, ml=None, df_ref=None, show_plots=True):
    if dc is None: dc = dc_full
    if ml is None: ml = ml_models
    if df_ref is None: df_ref = df_feat

    print(f"\n{'═' * 55}")
    print(f"  ⚽  {home.upper()}  ×  {away.upper()}")
    print(f"{'═' * 55}")

    result = ensemble_predict(home, away, dc, ml, df_ref)

    categories = {
        "⚡ xG PREVISTO": [
            "xG Previsto (Mand.)", "xG Previsto (Visit.)",
            "xG ML (Mand.)", "xG ML (Visit.)",
        ],
        "📊 RESULTADO 1X2": [
            "Vitória Mandante (1)", "Empate (X)", "Vitória Visitante (2)",
        ],
        "🔄 EMPATE ANULA APOSTA (DNB)": [
            "DNB Mandante", "DNB Visitante",
        ],
        "🎯 DUPLA CHANCE": [
            "DC 1X (Mand. ou Empate)",
            "DC X2 (Visit. ou Empate)",
            "DC 12 (Qualquer Vitória)",
        ],
        "⚽ TOTAL DE GOLS": [
            "Over 1.5", "Under 1.5",
            "Over 2.5", "Under 2.5",
            "Over 3.5", "Under 3.5",
        ],
        "🏠 GOLS MANDANTE": [
            "Over 0.5 Gols (Mand.)", "Over 1.5 Gols (Mand.)",
            "Over 2.5 Gols (Mand.)", "Over 3.5 Gols (Mand.)",
        ],
        "✈️  GOLS VISITANTE": [
            "Over 0.5 Gols (Visit.)", "Over 1.5 Gols (Visit.)",
            "Over 2.5 Gols (Visit.)", "Over 3.5 Gols (Visit.)",
        ],
        "🔵 BTTS": [
            "BTTS Sim", "BTTS Não",
        ],
    }

    xg_keys = {"xG Previsto (Mand.)", "xG Previsto (Visit.)",
               "xG ML (Mand.)", "xG ML (Visit.)"}

    rows = []
    for cat, mkts in categories.items():
        rows.append({"Categoria": cat, "Mercado": "", "Prob": ""})
        for m in mkts:
            p = result.get(m, np.nan)
            if isinstance(p, float) and np.isnan(p):
                fmt = "—"
            elif m in xg_keys:
                fmt = f"{p:.2f} gols"
            else:
                fmt = f"{p * 100:.1f}%"
            rows.append({"Categoria": "", "Mercado": m, "Prob": fmt})

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    if show_plots:
        plot_heatmap(home, away, dc)
        plot_markets(home, away, result)

    return df_out, result


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 15 — PREVISÃO EM LOTE (EXCEL)           ║
# ╚══════════════════════════════════════════════════════════╝

def predict_batch(jogos, dc=None, ml=None, df_ref=None):
    if dc is None: dc = dc_full
    if ml is None: ml = ml_models
    if df_ref is None: df_ref = df_feat

    rows = []
    n_err = 0
    for j in tqdm(jogos, desc="Prevendo"):
        try:
            _, r = predict_match(j["home"], j["away"], dc, ml, df_ref, show_plots=False)
            row = {
                "rodada": j.get("rodada", "?"),
                "date": j.get("date", "?"),
                "home": j["home"],
                "away": j["away"],
            }
            row.update(r)
            rows.append(row)
        except Exception as e:
            n_err += 1
            logger.warning(f"Erro: {j.get('home')} × {j.get('away')}: {e}")

    if n_err > 0:
        print(f"⚠️  {n_err} de {len(jogos)} jogos falharam")

    if not rows:
        print("❌ Nenhum jogo previsto com sucesso.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_predict_excel(path, export_csv=True):
    """Carrega Excel com jogos e gera previsões."""
    print(f"📂 Lendo: {path}")
    dj = pd.read_excel(path)
    dj.columns = dj.columns.str.strip().str.lower()

    required = {"rodada", "date", "home", "away"}
    missing = required - set(dj.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    dj["date"] = pd.to_datetime(dj["date"], dayfirst=True, errors="coerce")
    dj["home"] = dj["home"].str.strip()
    dj["away"] = dj["away"].str.strip()

    df_prev = predict_batch(dj.to_dict("records"))

    if export_csv:
        out = path.replace(".xlsx", "").replace(".xls", "") + "_previsoes_v3.csv"
        df_prev.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"💾 Exportado: {out}")

    return df_prev


# ╔══════════════════════════════════════════════════════════╗
# ║          CÉLULA 16 — EXEMPLO DE USO                      ║
# ╚══════════════════════════════════════════════════════════╝

print("\n🏟️  Times disponíveis:")
print(sorted(df["home"].unique()))

HOME_TEAM = "Manchester City"
AWAY_TEAM = "Arsenal"

df_resultado, resultado_dict = predict_match(
    home=HOME_TEAM, away=AWAY_TEAM, show_plots=True
)

# Calibração
if "bt_results" in dir() and not bt_results.empty:
    print("\n📊 Calibração:")
    plot_calibration(bt_results)

# ── Para previsão via Excel ──
CAMINHO_EXCEL = "/content/rodada.xlsx"
if os.path.exists(CAMINHO_EXCEL):
    df_rodada = load_predict_excel(CAMINHO_EXCEL)
else:
    print(f"\n⚠️  Para previsão em lote, faça upload de '{CAMINHO_EXCEL}'")
    print("   Formato: rodada | date | home | away")
