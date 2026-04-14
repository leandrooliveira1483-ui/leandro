#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  MODELO ML PREDITIVO DE FUTEBOL — GOOGLE COLAB                      ║
║                                                                      ║
║  Abordagem: PURE ML (sem Dixon-Coles)                                ║
║                                                                      ║
║  Arquitetura:                                                        ║
║   1. XGBoost Poisson → prever lambda_home e lambda_away              ║
║   2. Score matrix bivariada com copula gaussiana                     ║
║   3. LightGBM classificadores binários por mercado                   ║
║   4. Meta-ensemble: combina Poisson-derivado + classificador direto  ║
║                                                                      ║
║  Dados: Understat via soccerdata (xG, shots, schedule)               ║
║                                                                      ║
║  Mercados:                                                           ║
║   - 1X2, DNB, Dupla Chance (1X, X2, 12)                             ║
║   - Over/Under 1.5, 2.5, 3.5 total                                  ║
║   - Over 0.5/1.5/2.5/3.5 mandante                                   ║
║   - Over 0.5/1.5/2.5/3.5 visitante                                  ║
║   - BTTS                                                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════
# CÉLULA 1 — INSTALAÇÃO
# ══════════════════════════════════════════════════════════════════════
# !pip install soccerdata xgboost lightgbm scikit-learn scipy matplotlib seaborn tqdm -q

# ══════════════════════════════════════════════════════════════════════
# CÉLULA 2 — IMPORTS
# ══════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import poisson, norm
from scipy.optimize import minimize_scalar
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (log_loss, brier_score_loss,
                             accuracy_score, mean_squared_error)
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os, logging

import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️ LightGBM indisponível, usando apenas XGBoost")

import soccerdata as sd

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("ml_model")

print("✅ Imports OK")


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 3 — CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════

# Liga e temporadas
LEAGUE = "ENG-Premier League"
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

# Janelas de forma
WINDOWS = [3, 5, 10, 20]

# Limites de probabilidade
P_MIN = 0.02
P_MAX = 0.98
P_MIN_1X2 = 0.03
P_MAX_1X2 = 0.92

# Score matrix max gols
MAX_GOALS = 8

# Split temporal
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# Corte de data (None = usar tudo)
CUTOFF_DATE = None

print(f"⚙️ Config: {LEAGUE} | Temporadas: {SEASONS}")
print(f"   Split: {TRAIN_FRAC:.0%} treino / {VAL_FRAC:.0%} val / {1-TRAIN_FRAC-VAL_FRAC:.0%} teste")


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 4 — COLETA DE DADOS DO UNDERSTAT
# ══════════════════════════════════════════════════════════════════════

def fetch_understat(league, seasons):
    """Baixa schedule e shot events do Understat via soccerdata."""
    us = sd.Understat(leagues=league)
    schedules, shots_all = [], []

    print(f"\n📥 Baixando dados: {league}")
    for s in seasons:
        # Schedule
        try:
            sched = us.read_schedule(s)
            sched["season"] = s
            schedules.append(sched)
            print(f"  ✅ {s}: {len(sched)} jogos", end="")
        except Exception as e:
            print(f"  ❌ {s} schedule: {e}")
            continue

        # Shots
        try:
            shots = us.read_shot_events(s)
            shots["season"] = s
            shots_all.append(shots)
            print(f" | {len(shots)} chutes")
        except Exception as e:
            print(f" | ⚠️ shots: {e}")

    df_sched = pd.concat(schedules, ignore_index=True) if schedules else pd.DataFrame()
    df_shots = pd.concat(shots_all, ignore_index=True) if shots_all else pd.DataFrame()

    print(f"\n📊 Total: {len(df_sched)} jogos | {len(df_shots)} chutes")
    return df_sched, df_shots


df_raw, df_shots_raw = fetch_understat(LEAGUE, SEASONS)


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 5 — LIMPEZA E PADRONIZAÇÃO
# ══════════════════════════════════════════════════════════════════════

def clean_data(df_sched, df_shots, cutoff=None):
    """Limpa e padroniza dados do schedule."""
    df = df_sched.copy()

    # Renomear colunas padrão do Understat
    renames = {}
    for old, new in [("home_team", "home"), ("away_team", "away"),
                     ("home_goals", "hg"), ("away_goals", "ag"),
                     ("home_xg", "hxg"), ("away_xg", "axg")]:
        if old in df.columns:
            renames[old] = new
    df = df.rename(columns=renames)

    # Tipos
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["hg", "ag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["hxg", "axg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remover jogos sem resultado
    df = df.dropna(subset=["hg", "ag"]).copy()
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)

    # Derivadas básicas
    df["total_goals"] = df["hg"] + df["ag"]
    df["result"] = np.where(df["hg"] > df["ag"], "H",
                   np.where(df["hg"] < df["ag"], "A", "D"))
    df["gd"] = df["hg"] - df["ag"]  # goal difference do mandante

    # Cutoff
    if cutoff:
        df = df[df["date"] <= pd.to_datetime(cutoff)]

    df = df.sort_values("date").reset_index(drop=True)

    # Shots
    shots = pd.DataFrame()
    if df_shots is not None and not df_shots.empty:
        shots = df_shots.copy()
        date_col = next((c for c in shots.columns if "date" in c.lower()), None)
        if date_col:
            shots[date_col] = pd.to_datetime(shots[date_col], errors="coerce")

    print(f"✅ Dados limpos: {len(df)} jogos finalizados")
    return df, shots


df, shots = clean_data(df_raw, df_shots_raw, CUTOFF_DATE)

# Resumo
print(f"   Temporadas: {sorted(df['season'].unique())}")
print(f"   De {df['date'].min().date()} a {df['date'].max().date()}")
print(f"   Times: {df['home'].nunique()}")
print(f"   Média gols/jogo: {df['total_goals'].mean():.2f}")


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 6 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
#
# Filosofia: cada time tem um "perfil de jogo" capturado por:
#   1. Power ratings: força ofensiva/defensiva relativa à liga
#   2. Forma recente: tendência (momentum) em múltiplas janelas
#   3. Qualidade de chute: métricas de shot-level
#   4. Contexto: descanso, tabela, confronto direto
#
# TUDO com shift(1) para evitar data leakage.

def build_match_features(df, shots):
    """
    Constrói todas as features match-level a partir dos dados brutos.
    Retorna DataFrame com uma linha por jogo + features pré-jogo.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()

    # ── 1. TIMELINE POR TIME ──
    # Cada jogo gera duas linhas: uma para o mandante, uma para o visitante
    rows_h = df[["date", "season", "home", "away", "hg", "ag", "hxg", "axg"]].copy()
    rows_h.columns = ["date", "season", "team", "opp", "gf", "ga", "xgf", "xga"]
    rows_h["is_home"] = 1

    rows_a = df[["date", "season", "away", "home", "ag", "hg", "axg", "hxg"]].copy()
    rows_a.columns = ["date", "season", "team", "opp", "gf", "ga", "xgf", "xga"]
    rows_a["is_home"] = 0

    tl = pd.concat([rows_h, rows_a], ignore_index=True)
    tl = tl.sort_values(["team", "date"]).reset_index(drop=True)

    # Métricas derivadas por jogo
    tl["xgf"] = pd.to_numeric(tl["xgf"], errors="coerce")
    tl["xga"] = pd.to_numeric(tl["xga"], errors="coerce")
    tl["gf"] = tl["gf"].astype(float)
    tl["ga"] = tl["ga"].astype(float)

    tl["pts"] = np.where(tl["gf"] > tl["ga"], 3,
                np.where(tl["gf"] == tl["ga"], 1, 0)).astype(float)
    tl["win"] = (tl["gf"] > tl["ga"]).astype(float)
    tl["clean_sheet"] = (tl["ga"] == 0).astype(float)
    tl["failed_score"] = (tl["gf"] == 0).astype(float)
    tl["btts"] = ((tl["gf"] > 0) & (tl["ga"] > 0)).astype(float)

    # Eficiência: conversão de xG em gols
    tl["conversion"] = np.where(tl["xgf"] > 0.1, tl["gf"] / tl["xgf"], 1.0)
    tl["def_save"] = np.where(tl["xga"] > 0.1, 1.0 - tl["ga"] / tl["xga"], 0.0)

    # xG diferença
    tl["xg_diff"] = tl["xgf"] - tl["xga"]

    # ── 2. ROLLING FEATURES (múltiplas janelas) ──
    roll_cols = ["gf", "ga", "xgf", "xga", "pts", "win", "clean_sheet",
                 "failed_score", "btts", "conversion", "def_save", "xg_diff"]

    for w in WINDOWS:
        for col in roll_cols:
            tl[f"r{w}_{col}"] = (
                tl.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

        # Venue-specific (só quando joga em casa / fora)
        for venue_val, vtag in [(1, "home"), (0, "away")]:
            mask = tl["is_home"] == venue_val
            for col in ["gf", "ga", "xgf", "xga"]:
                cname = f"r{w}_{col}_at_{vtag}"
                tl[cname] = np.nan
                tl.loc[mask, cname] = (
                    tl.loc[mask].groupby("team")[col]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                )

    # ── 3. MOMENTUM (tendência) ──
    # Diferença entre janela curta e longa = time está melhorando ou piorando?
    if 3 in WINDOWS and 20 in WINDOWS:
        for col in ["xgf", "xga", "pts", "xg_diff"]:
            tl[f"trend_{col}"] = tl[f"r3_{col}"] - tl[f"r20_{col}"]

    # ── 4. POWER RATINGS ──
    # Força relativa à média da liga (ataque e defesa separados)
    # Calculado por temporada para refletir mudanças de elenco
    league_avg = tl.groupby("season").agg(
        avg_gf=("gf", "mean"), avg_ga=("ga", "mean"),
        avg_xgf=("xgf", "mean"), avg_xga=("xga", "mean")
    )
    tl = tl.merge(league_avg, on="season", how="left")

    # Cumulative strength: média acumulada até o jogo anterior
    for col, avg_col in [("gf", "avg_gf"), ("ga", "avg_ga"),
                          ("xgf", "avg_xgf"), ("xga", "avg_xga")]:
        cumavg = (tl.groupby(["team", "season"])[col]
                  .transform(lambda x: x.shift(1).expanding(min_periods=3).mean()))
        tl[f"pwr_{col}"] = cumavg / tl[avg_col]  # ratio vs liga

    tl = tl.drop(columns=["avg_gf", "avg_ga", "avg_xgf", "avg_xga"])

    # ── 5. SHOT FEATURES ──
    tl = _add_shot_features(tl, shots)

    # ── 6. MERGE DE VOLTA AO NÍVEL DE JOGO ──
    # Separar features de mandante e visitante
    feat_cols = [c for c in tl.columns if c.startswith(("r", "trend_", "pwr_", "shot_"))]

    tl_home = tl[tl["is_home"] == 1][["team", "date"] + feat_cols].copy()
    tl_home = tl_home.rename(columns={c: f"H_{c}" for c in feat_cols})
    tl_home = tl_home.rename(columns={"team": "home"})
    tl_home = tl_home.drop_duplicates(["home", "date"], keep="last")

    tl_away = tl[tl["is_home"] == 0][["team", "date"] + feat_cols].copy()
    tl_away = tl_away.rename(columns={c: f"A_{c}" for c in feat_cols})
    tl_away = tl_away.rename(columns={"team": "away"})
    tl_away = tl_away.drop_duplicates(["away", "date"], keep="last")

    out = df.merge(tl_home, on=["home", "date"], how="left")
    out = out.merge(tl_away, on=["away", "date"], how="left")
    out = out.drop_duplicates(["home", "away", "date"], keep="first")

    # ── 7. DIFERENCIAIS (mandante - visitante) ──
    for w in WINDOWS:
        for col in ["xgf", "xga", "pts", "win", "xg_diff"]:
            hc = f"H_r{w}_{col}"
            ac = f"A_r{w}_{col}"
            if hc in out.columns and ac in out.columns:
                out[f"diff_{w}_{col}"] = out[hc] - out[ac]

    # Power rating diferencial
    for col in ["pwr_gf", "pwr_ga", "pwr_xgf", "pwr_xga"]:
        hc, ac = f"H_{col}", f"A_{col}"
        if hc in out.columns and ac in out.columns:
            out[f"diff_{col}"] = out[hc] - out[ac]

    # ── 8. CONTEXTO ──
    out = _add_context_features(out, tl)

    # ── 9. H2H ──
    out = _add_h2h(out)

    return out


def _add_shot_features(tl, shots):
    """Agrega métricas de chute por time/jogo e calcula rolling."""
    if shots is None or shots.empty:
        return tl

    # Identifica colunas
    xg_col = next((c for c in shots.columns if c.lower() in ["xg", "xgoal"]), None)
    team_col = next((c for c in shots.columns if c in ["team", "shooter_team"]), None)
    home_col = next((c for c in shots.columns if c in ["home", "home_team"]), None)
    result_col = next((c for c in shots.columns if "result" in c.lower()), None)
    date_col = next((c for c in shots.columns if "date" in c.lower()), None)

    if not all([xg_col, team_col, date_col]):
        return tl

    s = shots.copy()
    s["_date"] = pd.to_datetime(s[date_col], errors="coerce").dt.normalize()
    s["_xg"] = pd.to_numeric(s[xg_col], errors="coerce")
    s["_team"] = s[team_col]

    if result_col:
        s["_on_target"] = s[result_col].astype(str).str.lower().isin(
            ["goal", "savedshot", "blockedshot"]).astype(float)
        s["_goal"] = (s[result_col].astype(str).str.lower() == "goal").astype(float)
    else:
        s["_on_target"] = 0.0
        s["_goal"] = 0.0

    # Big chance: xG > 0.3
    s["_big_chance"] = (s["_xg"] > 0.3).astype(float)

    # Agregar por time/jogo
    agg = s.groupby(["_team", "_date"]).agg(
        shot_n=("_xg", "count"),
        shot_xg_total=("_xg", "sum"),
        shot_xg_avg=("_xg", "mean"),
        shot_on_target=("_on_target", "sum"),
        shot_big_chance=("_big_chance", "sum"),
        shot_goals=("_goal", "sum"),
    ).reset_index()
    agg = agg.rename(columns={"_team": "team", "_date": "date"})

    # Conversion rate do jogo
    agg["shot_conv_rate"] = np.where(
        agg["shot_n"] > 0, agg["shot_goals"] / agg["shot_n"], 0)

    # Merge com timeline
    tl = tl.merge(agg.drop(columns=["shot_goals"]),
                   on=["team", "date"], how="left")

    # Rolling shot features
    shot_metrics = ["shot_n", "shot_xg_total", "shot_xg_avg",
                    "shot_on_target", "shot_big_chance", "shot_conv_rate"]
    for col in shot_metrics:
        if col in tl.columns:
            tl[f"r10_{col}"] = (
                tl.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            )
            # Renomeia para padrão shot_
            tl = tl.rename(columns={f"r10_{col}": f"shot_{col}"})

    # Remove colunas de jogo individual (manter só rolling)
    tl = tl.drop(columns=[c for c in shot_metrics if c in tl.columns], errors="ignore")

    return tl


def _add_context_features(out, tl):
    """Dias de descanso, posição na tabela, fadiga."""
    # Dias de descanso
    tl_dates = tl[["team", "date"]].drop_duplicates().sort_values(["team", "date"])
    tl_dates["prev_date"] = tl_dates.groupby("team")["date"].shift(1)
    tl_dates["rest"] = (tl_dates["date"] - tl_dates["prev_date"]).dt.days

    rest_map = tl_dates.set_index(["team", "date"])["rest"]

    out["rest_home"] = out.apply(
        lambda r: rest_map.get((r["home"], r["date"]), np.nan), axis=1)
    out["rest_away"] = out.apply(
        lambda r: rest_map.get((r["away"], r["date"]), np.nan), axis=1)

    med_rest = out["rest_home"].median()
    out["rest_home"] = out["rest_home"].fillna(med_rest).clip(1, 30)
    out["rest_away"] = out["rest_away"].fillna(med_rest).clip(1, 30)
    out["rest_diff"] = out["rest_home"] - out["rest_away"]
    out["fatigue_home"] = (out["rest_home"] <= 3).astype(int)
    out["fatigue_away"] = (out["rest_away"] <= 3).astype(int)

    # Pontos acumulados na temporada (posição relativa)
    tl_pts = tl[["team", "date", "season", "pts"]].copy()
    tl_pts["cum_pts"] = (
        tl_pts.groupby(["team", "season"])["pts"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )
    pts_map = tl_pts.drop_duplicates(["team", "date"], keep="last") \
                    .set_index(["team", "date"])["cum_pts"]

    out["pts_home"] = out.apply(
        lambda r: pts_map.get((r["home"], r["date"]), 0), axis=1)
    out["pts_away"] = out.apply(
        lambda r: pts_map.get((r["away"], r["date"]), 0), axis=1)
    out["pts_diff"] = out["pts_home"] - out["pts_away"]

    return out


def _add_h2h(out, window=6):
    """Head-to-head entre pares de times."""
    out = out.copy()
    # Cria par ordenado
    t = np.sort(np.stack([out["home"].values, out["away"].values]), axis=0)
    out["_pair"] = [f"{a}_vs_{b}" for a, b in zip(t[0], t[1])]

    out["_hwon"] = (out["result"] == "H").astype(float)
    out["_draw"] = (out["result"] == "D").astype(float)
    out["_tg"] = out["total_goals"].astype(float)

    out = out.sort_values(["_pair", "date"])

    for col, name in [("_hwon", "h2h_hwin"), ("_draw", "h2h_draw"), ("_tg", "h2h_goals")]:
        out[name] = (
            out.groupby("_pair")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    out["h2h_n"] = (
        out.groupby("_pair")["_hwon"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).count())
    )

    out = out.drop(columns=["_pair", "_hwon", "_draw", "_tg"])
    out = out.sort_values("date").reset_index(drop=True)

    return out


# ── EXECUÇÃO ──
print("\n⚙️ Construindo features...")
df_feat = build_match_features(df, shots)
print(f"✅ Features: {df_feat.shape[1]} colunas | {len(df_feat)} jogos")

# Lista de features para o ML
FEATURE_NAMES = [c for c in df_feat.columns
                 if c.startswith(("H_", "A_", "diff_", "rest_", "fatigue_",
                                  "pts_", "h2h_"))
                 and c not in ["home", "away", "date", "season"]]
print(f"   {len(FEATURE_NAMES)} features selecionadas")


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 7 — PREPARAÇÃO DE TARGETS
# ══════════════════════════════════════════════════════════════════════

def prepare_targets(df_f):
    """Cria todos os targets de classificação e regressão."""
    d = df_f.copy()

    # Regressão — gols
    d["y_hg"] = d["hg"].astype(float)
    d["y_ag"] = d["ag"].astype(float)

    # Classificação — 1X2
    d["y_1x2"] = d["result"].map({"H": 0, "D": 1, "A": 2})

    # Binários — over/under totais
    d["y_ov15"] = (d["total_goals"] >= 2).astype(int)
    d["y_ov25"] = (d["total_goals"] >= 3).astype(int)
    d["y_ov35"] = (d["total_goals"] >= 4).astype(int)

    # Binários — over mandante
    d["y_hov05"] = (d["hg"] >= 1).astype(int)
    d["y_hov15"] = (d["hg"] >= 2).astype(int)
    d["y_hov25"] = (d["hg"] >= 3).astype(int)
    d["y_hov35"] = (d["hg"] >= 4).astype(int)

    # Binários — over visitante
    d["y_aov05"] = (d["ag"] >= 1).astype(int)
    d["y_aov15"] = (d["ag"] >= 2).astype(int)
    d["y_aov25"] = (d["ag"] >= 3).astype(int)
    d["y_aov35"] = (d["ag"] >= 4).astype(int)

    # BTTS
    d["y_btts"] = ((d["hg"] >= 1) & (d["ag"] >= 1)).astype(int)

    return d


df_feat = prepare_targets(df_feat)

# Targets
REG_TARGETS = ["y_hg", "y_ag"]
CLS_TARGETS = ["y_1x2", "y_ov15", "y_ov25", "y_ov35",
               "y_hov05", "y_hov15", "y_hov25", "y_hov35",
               "y_aov05", "y_aov15", "y_aov25", "y_aov35",
               "y_btts"]

print(f"✅ Targets: {len(REG_TARGETS)} regressão | {len(CLS_TARGETS)} classificação")


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 8 — TREINAMENTO DOS MODELOS
# ══════════════════════════════════════════════════════════════════════
#
# Arquitetura:
#   Nível 1 (Base):
#     - XGBoost Poisson → lambda_home, lambda_away
#     - XGBoost classificador → cada mercado binário
#     - LightGBM classificador → cada mercado binário (diversidade)
#
#   Nível 2 (Meta):
#     - LogisticRegression combina XGB + LGB por mercado
#
# Split temporal: 70% treino / 15% validação (meta) / 15% teste

def fillna_smart(X, medians=None):
    """Preenche NaN com medianas do treino ou defaults."""
    X = X.copy()
    if medians is not None:
        X = X.fillna(medians.reindex(X.columns))
    for c in X.columns:
        if X[c].isna().any():
            if "fatigue" in c:
                X[c] = X[c].fillna(0)
            elif "pwr" in c:
                X[c] = X[c].fillna(1.0)
            else:
                X[c] = X[c].fillna(0)
    return X


def train_all_models(df_f, features, reg_targets, cls_targets):
    """Treina todos os modelos com split temporal."""
    # Filtrar linhas com dados suficientes
    mask = df_f[features].notna().mean(axis=1) > 0.5
    df_f = df_f[mask].copy().sort_values("date").reset_index(drop=True)

    n = len(df_f)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * (TRAIN_FRAC + VAL_FRAC))

    X = df_f[features].astype(float)
    X_train = X.iloc[:n_train]
    X_val = X.iloc[n_train:n_val]
    X_test = X.iloc[n_val:]

    medians = X_train.median()
    X_train = fillna_smart(X_train, medians)
    X_val = fillna_smart(X_val, medians)
    X_test = fillna_smart(X_test, medians)

    print(f"\n📊 Split: treino={n_train} | val={n_val - n_train} | teste={n - n_val}")

    models = {}

    # ── REGRESSORES POISSON (lambda home/away) ──
    print("\n🔧 REGRESSORES POISSON:")
    for target in reg_targets:
        y_train = df_f[target].iloc[:n_train]
        y_val = df_f[target].iloc[n_train:n_val]

        reg = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        reg.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)

        pred_val = reg.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        mae = np.mean(np.abs(y_val - pred_val))
        print(f"  {target}: RMSE={rmse:.3f} MAE={mae:.3f}")

        models[target] = {"model": reg, "type": "poisson"}

    # ── CLASSIFICADORES BINÁRIOS + META ──
    print("\n🔧 CLASSIFICADORES:")
    for target in cls_targets:
        is_multi = (target == "y_1x2")
        y_train = df_f[target].iloc[:n_train]
        y_val = df_f[target].iloc[n_train:n_val]

        obj_x = "multi:softprob" if is_multi else "binary:logistic"
        met_x = "mlogloss" if is_multi else "logloss"

        # XGBoost base
        xgb_clf = xgb.XGBClassifier(
            objective=obj_x, eval_metric=met_x,
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, reg_alpha=0.1,
            random_state=42, n_jobs=-1
        )
        xgb_clf.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     verbose=False)

        # LightGBM base
        lgb_clf = None
        if HAS_LGB:
            obj_l = "multiclass" if is_multi else "binary"
            extra = {"num_class": 3} if is_multi else {}
            lgb_clf = lgb.LGBMClassifier(
                objective=obj_l, n_estimators=400, max_depth=5,
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                min_child_weight=5, reg_alpha=0.1,
                random_state=42, n_jobs=-1, verbose=-1, **extra
            )
            lgb_clf.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(0)])

        # Meta-learner: treina no val set
        xgb_val_p = xgb_clf.predict_proba(X_val)
        if lgb_clf is not None:
            lgb_val_p = lgb_clf.predict_proba(X_val)
            meta_X = np.hstack([xgb_val_p, lgb_val_p])
        else:
            meta_X = xgb_val_p

        meta = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        meta.fit(meta_X, y_val)

        # Métricas no val
        meta_p = meta.predict_proba(meta_X)
        ll = log_loss(y_val, meta_p)
        if is_multi:
            acc = accuracy_score(y_val, meta.predict(meta_X))
            print(f"  {target}: LL={ll:.4f} Acc={acc:.3f}")
        else:
            bs = brier_score_loss(y_val, meta_p[:, 1])
            print(f"  {target}: LL={ll:.4f} Brier={bs:.4f}")

        models[target] = {
            "xgb": xgb_clf, "lgb": lgb_clf, "meta": meta, "type": "cls"
        }

    info = {
        "n_train": n_train, "n_val": n_val,
        "medians": medians,
        "df_feat": df_f,
        "X_test": X_test,
    }

    return models, info


print("⚙️ Treinando modelos...")
models, train_info = train_all_models(df_feat, FEATURE_NAMES, REG_TARGETS, CLS_TARGETS)
MEDIANS = train_info["medians"]
print("\n✅ Todos os modelos treinados!")


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 9 — SCORE MATRIX & DERIVAÇÃO DE MERCADOS
# ══════════════════════════════════════════════════════════════════════
#
# A partir dos lambdas previstos pelo XGBoost Poisson, constrói a
# score matrix com copula gaussiana para capturar a correlação entre
# gols do mandante e visitante (dependência empírica ≈ -0.05 a 0.15).
# Todos os mercados são derivados desta matrix + classificadores diretos.

def estimate_goal_correlation(df_f, n_train):
    """
    Estima correlação empírica entre gols do mandante e visitante
    nos dados de treino. Usado na copula da score matrix.
    """
    d = df_f.iloc[:n_train]
    hg = d["hg"].astype(float)
    ag = d["ag"].astype(float)
    return float(np.corrcoef(hg, ag)[0, 1])


def build_score_matrix(lam_h, lam_a, rho=0.0, max_g=MAX_GOALS):
    """
    Score matrix bivariada com copula gaussiana.

    Se rho=0, reduz a Poisson independente.
    rho > 0 → mais resultados extremos (ambos marcam ou ambos não marcam)
    rho < 0 → mais resultados com diferença grande
    """
    g = np.arange(max_g + 1)

    if abs(rho) < 0.01:
        # Poisson independente
        ph = poisson.pmf(g, max(lam_h, 0.01))
        pa = poisson.pmf(g, max(lam_a, 0.01))
        mat = np.outer(ph, pa)
    else:
        # Copula gaussiana
        mat = np.zeros((max_g + 1, max_g + 1))
        cdf_h = poisson.cdf(g, max(lam_h, 0.01))
        cdf_a = poisson.cdf(g, max(lam_a, 0.01))

        # Converte para espaço normal
        u_h = np.clip(cdf_h, 1e-8, 1 - 1e-8)
        u_a = np.clip(cdf_a, 1e-8, 1 - 1e-8)
        z_h = norm.ppf(u_h)
        z_a = norm.ppf(u_a)

        for i in range(max_g + 1):
            for j in range(max_g + 1):
                # P(H=i, A=j) via diferenças da CDF bivariada
                z_hi = z_h[i]
                z_lo_h = z_h[i - 1] if i > 0 else -10
                z_aj = z_a[j]
                z_lo_a = z_a[j - 1] if j > 0 else -10

                # CDF bivariada normal
                from scipy.stats import multivariate_normal
                cov = [[1, rho], [rho, 1]]
                mv = multivariate_normal(mean=[0, 0], cov=cov)

                p = (mv.cdf([z_hi, z_aj])
                     - mv.cdf([z_lo_h, z_aj])
                     - mv.cdf([z_hi, z_lo_a])
                     + mv.cdf([z_lo_h, z_lo_a]))
                mat[i, j] = max(p, 0)

    # Normaliza
    s = mat.sum()
    if s > 0:
        mat /= s
    return mat


def markets_from_matrix(mat):
    """Deriva TODOS os mercados a partir da score matrix."""
    max_g = mat.shape[0]

    # 1X2
    pH = float(np.tril(mat, -1).sum())  # home win: abaixo da diagonal
    pA = float(np.triu(mat, 1).sum())   # away win: acima da diagonal
    pD = float(np.diag(mat).sum())       # draw: diagonal

    # Total goals distribution
    total_pmf = np.zeros(2 * max_g - 1)
    for i in range(max_g):
        for j in range(max_g):
            total_pmf[i + j] += mat[i, j]
    total_cdf = np.cumsum(total_pmf)

    # Marginais
    h_marg = mat.sum(axis=1)  # P(home = i)
    a_marg = mat.sum(axis=0)  # P(away = j)
    h_cdf = np.cumsum(h_marg)
    a_cdf = np.cumsum(a_marg)

    def ov_total(k):
        return float(1 - total_cdf[int(k)]) if int(k) < len(total_cdf) else 0.0

    def ov_team(cdf, k):
        k = int(k)
        if k <= 0: return 1.0
        if k > len(cdf): return 0.0
        return float(1 - cdf[k - 1])

    return {
        # 1X2
        "1": pH, "X": pD, "2": pA,
        # Over totais
        "ov15": ov_total(1), "ov25": ov_total(2), "ov35": ov_total(3),
        # Over mandante
        "hov05": ov_team(h_cdf, 1), "hov15": ov_team(h_cdf, 2),
        "hov25": ov_team(h_cdf, 3), "hov35": ov_team(h_cdf, 4),
        # Over visitante
        "aov05": ov_team(a_cdf, 1), "aov15": ov_team(a_cdf, 2),
        "aov25": ov_team(a_cdf, 3), "aov35": ov_team(a_cdf, 4),
        # BTTS
        "btts": float(mat[1:, 1:].sum()),
    }


# Correlação empírica
RHO = estimate_goal_correlation(train_info["df_feat"], train_info["n_train"])
print(f"\n📊 Correlação empírica gols (home, away): {RHO:.4f}")


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 10 — ENSEMBLE PREDICT
# ══════════════════════════════════════════════════════════════════════

def predict_match_probs(X_row, models, rho=RHO, alpha=0.5):
    """
    Prevê TODOS os mercados para uma partida.

    Combina:
      - Score matrix (derivada dos lambdas Poisson do XGBoost)
      - Classificadores diretos (XGB + LGB + Meta)

    alpha: peso da score matrix vs classificador direto
           (otimizado no backtest)
    """
    X = X_row if isinstance(X_row, pd.DataFrame) else pd.DataFrame([X_row])

    # ── Lambdas do Poisson ──
    lam_h = float(models["y_hg"]["model"].predict(X)[0])
    lam_a = float(models["y_ag"]["model"].predict(X)[0])
    lam_h = max(lam_h, 0.05)
    lam_a = max(lam_a, 0.05)

    # ── Score matrix ──
    mat = build_score_matrix(lam_h, lam_a, rho=rho)
    mat_probs = markets_from_matrix(mat)

    # ── Classificadores diretos ──
    def cls_prob(target):
        """Probabilidade via stacking (XGB + LGB → Meta)."""
        m = models.get(target)
        if m is None or m["type"] != "cls":
            return None
        xp = m["xgb"].predict_proba(X)
        if m.get("lgb") is not None:
            lp = m["lgb"].predict_proba(X)
            meta_x = np.hstack([xp, lp])
        else:
            meta_x = xp
        return m["meta"].predict_proba(meta_x)[0]

    # 1X2 do classificador
    cls_1x2 = cls_prob("y_1x2")

    # Binários do classificador: P(classe=1)
    cls_bins = {}
    for t in ["y_ov15", "y_ov25", "y_ov35",
              "y_hov05", "y_hov15", "y_hov25", "y_hov35",
              "y_aov05", "y_aov15", "y_aov25", "y_aov35",
              "y_btts"]:
        p = cls_prob(t)
        cls_bins[t] = float(p[1]) if p is not None else None

    # ── BLEND: score_matrix (alpha) + classificador (1-alpha) ──
    def blend(mat_val, cls_val, floor=P_MIN, ceil=P_MAX):
        if cls_val is None:
            return np.clip(mat_val, floor, ceil)
        return np.clip(alpha * mat_val + (1 - alpha) * cls_val, floor, ceil)

    # 1X2
    pH = blend(mat_probs["1"], cls_1x2[0] if cls_1x2 is not None else None,
               P_MIN_1X2, P_MAX_1X2)
    pD = blend(mat_probs["X"], cls_1x2[1] if cls_1x2 is not None else None,
               P_MIN_1X2, P_MAX_1X2)
    pA = blend(mat_probs["2"], cls_1x2[2] if cls_1x2 is not None else None,
               P_MIN_1X2, P_MAX_1X2)
    s = pH + pD + pA
    pH, pD, pA = pH / s, pD / s, pA / s

    # Over totais
    ov15 = blend(mat_probs["ov15"], cls_bins.get("y_ov15"))
    ov25 = blend(mat_probs["ov25"], cls_bins.get("y_ov25"))
    ov35 = blend(mat_probs["ov35"], cls_bins.get("y_ov35"))

    # Over mandante
    hov05 = blend(mat_probs["hov05"], cls_bins.get("y_hov05"))
    hov15 = blend(mat_probs["hov15"], cls_bins.get("y_hov15"))
    hov25 = blend(mat_probs["hov25"], cls_bins.get("y_hov25"))
    hov35 = blend(mat_probs["hov35"], cls_bins.get("y_hov35"))

    # Over visitante
    aov05 = blend(mat_probs["aov05"], cls_bins.get("y_aov05"))
    aov15 = blend(mat_probs["aov15"], cls_bins.get("y_aov15"))
    aov25 = blend(mat_probs["aov25"], cls_bins.get("y_aov25"))
    aov35 = blend(mat_probs["aov35"], cls_bins.get("y_aov35"))

    # BTTS
    btts = blend(mat_probs["btts"], cls_bins.get("y_btts"))

    # Monotonicity enforcement
    ov35 = min(ov35, ov25); ov25 = min(ov25, ov15)
    hov35 = min(hov35, hov25); hov25 = min(hov25, hov15); hov15 = min(hov15, hov05)
    aov35 = min(aov35, aov25); aov25 = min(aov25, aov15); aov15 = min(aov15, aov05)

    # Dupla chance e DNB
    dnb_s = pH + pA
    dnb_h = pH / dnb_s if dnb_s > 0 else 0.5
    dnb_a = pA / dnb_s if dnb_s > 0 else 0.5

    return {
        "lambda_home": round(lam_h, 3),
        "lambda_away": round(lam_a, 3),

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

        "Over 0.5 Gols (Mand.)": round(hov05, 4),
        "Over 1.5 Gols (Mand.)": round(hov15, 4),
        "Over 2.5 Gols (Mand.)": round(hov25, 4),
        "Over 3.5 Gols (Mand.)": round(hov35, 4),

        "Over 0.5 Gols (Visit.)": round(aov05, 4),
        "Over 1.5 Gols (Visit.)": round(aov15, 4),
        "Over 2.5 Gols (Visit.)": round(aov25, 4),
        "Over 3.5 Gols (Visit.)": round(aov35, 4),

        "BTTS Sim": round(btts, 4),
        "BTTS Não": round(1 - btts, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 11 — OTIMIZAÇÃO DO ALPHA
# ══════════════════════════════════════════════════════════════════════

def optimize_alpha(models, df_f, info, rho):
    """Otimiza alpha (peso matrix vs classificador) no validation set."""
    n_train = info["n_train"]
    n_val = info["n_val"]
    df_val = df_f.iloc[n_train:n_val]

    print("\n⚙️ Otimizando alpha no validation set...")

    results = []
    for _, row in df_val.iterrows():
        try:
            X_row = fillna_smart(
                pd.DataFrame([{c: row.get(c, np.nan) for c in FEATURE_NAMES}]).astype(float),
                MEDIANS
            )
            lam_h = float(models["y_hg"]["model"].predict(X_row)[0])
            lam_a = float(models["y_ag"]["model"].predict(X_row)[0])
            mat = build_score_matrix(max(lam_h, 0.05), max(lam_a, 0.05), rho=rho)
            mp = markets_from_matrix(mat)

            m1x2 = models["y_1x2"]
            xp = m1x2["xgb"].predict_proba(X_row)
            if m1x2.get("lgb") is not None:
                lp = m1x2["lgb"].predict_proba(X_row)
                meta_x = np.hstack([xp, lp])
            else:
                meta_x = xp
            cls_p = m1x2["meta"].predict_proba(meta_x)[0]

            results.append({
                "result": row["result"],
                "mat_H": mp["1"], "mat_D": mp["X"], "mat_A": mp["2"],
                "cls_H": cls_p[0], "cls_D": cls_p[1], "cls_A": cls_p[2],
            })
        except Exception:
            pass

    if len(results) < 30:
        print(f"⚠️ Poucas amostras ({len(results)}), usando alpha=0.45")
        return 0.45

    rv = pd.DataFrame(results)
    y_true = rv["result"].map({"H": 0, "D": 1, "A": 2})

    def neg_ll(a):
        pH = a * rv["mat_H"] + (1 - a) * rv["cls_H"]
        pD = a * rv["mat_D"] + (1 - a) * rv["cls_D"]
        pA = a * rv["mat_A"] + (1 - a) * rv["cls_A"]
        s = pH + pD + pA
        probs = np.column_stack([pH / s, pD / s, pA / s])
        return log_loss(y_true, np.clip(probs, 1e-8, 1 - 1e-8))

    res = minimize_scalar(neg_ll, bounds=(0.0, 1.0), method="bounded")
    alpha_opt = round(res.x, 3)
    print(f"✅ Alpha ótimo: {alpha_opt} (LL={res.fun:.4f}, n={len(results)})")
    return alpha_opt


ALPHA = optimize_alpha(models, train_info["df_feat"], train_info, RHO)
print(f"   Alpha final: {ALPHA}")


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 12 — BACKTEST (holdout final)
# ══════════════════════════════════════════════════════════════════════

def run_backtest(models, df_f, info, rho, alpha):
    """Backtest no holdout final (últimos 15% dos dados)."""
    n_val = info["n_val"]
    df_test = df_f.iloc[n_val:].copy()
    df_test = df_test.dropna(subset=["result", "hg", "ag"])

    print(f"\n⚙️ Backtest: {len(df_test)} jogos (holdout final)")

    records = []
    n_err = 0
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Backtest"):
        try:
            X_row = fillna_smart(
                pd.DataFrame([{c: row.get(c, np.nan) for c in FEATURE_NAMES}]).astype(float),
                MEDIANS
            )
            pred = predict_match_probs(X_row, models, rho=rho, alpha=alpha)
            records.append({
                "home": row["home"], "away": row["away"],
                "date": row["date"], "result": row["result"],
                "hg": int(row["hg"]), "ag": int(row["ag"]),
                "total": int(row["total_goals"]),
                "p_H": pred["Vitória Mandante (1)"],
                "p_D": pred["Empate (X)"],
                "p_A": pred["Vitória Visitante (2)"],
                "p_ov25": pred["Over 2.5"],
                "p_ov15": pred["Over 1.5"],
                "p_btts": pred["BTTS Sim"],
                "lam_h": pred["lambda_home"],
                "lam_a": pred["lambda_away"],
            })
        except Exception as e:
            n_err += 1
            log.debug(f"Backtest erro: {e}")

    if n_err > 0:
        print(f"⚠️ {n_err} erros no backtest")

    bt = pd.DataFrame(records)
    if bt.empty:
        print("❌ Backtest vazio")
        return bt

    # Métricas
    bt["y_H"] = (bt["result"] == "H").astype(int)
    bt["y_D"] = (bt["result"] == "D").astype(int)
    bt["y_A"] = (bt["result"] == "A").astype(int)
    bt["y_ov25"] = (bt["total"] >= 3).astype(int)
    bt["y_ov15"] = (bt["total"] >= 2).astype(int)
    bt["y_btts"] = ((bt["hg"] >= 1) & (bt["ag"] >= 1)).astype(int)

    y_true = bt["result"].map({"H": 0, "D": 1, "A": 2})
    y_prob = bt[["p_H", "p_D", "p_A"]].values

    ll = log_loss(y_true, y_prob)
    acc = accuracy_score(y_true, np.argmax(y_prob, axis=1))

    bs_H = brier_score_loss(bt["y_H"], bt["p_H"])
    bs_D = brier_score_loss(bt["y_D"], bt["p_D"])
    bs_A = brier_score_loss(bt["y_A"], bt["p_A"])
    bs_ov25 = brier_score_loss(bt["y_ov25"], bt["p_ov25"])
    bs_btts = brier_score_loss(bt["y_btts"], bt["p_btts"])

    rmse_h = np.sqrt(mean_squared_error(bt["hg"], bt["lam_h"]))
    rmse_a = np.sqrt(mean_squared_error(bt["ag"], bt["lam_a"]))

    # Por faixa de confiança
    pred_label = np.argmax(y_prob, axis=1)
    conf = np.max(y_prob, axis=1)
    high_conf = conf > 0.50
    acc_high = accuracy_score(y_true[high_conf], pred_label[high_conf]) if high_conf.sum() > 10 else float("nan")

    sep = "═" * 58
    print(f"\n{sep}")
    print(f"  📈 BACKTEST COMPLETO — {len(bt)} jogos | {n_err} erros")
    print(f"{sep}")
    print(f"  Log-Loss 1X2:       {ll:.4f}  (aleatório ≈ 1.099)")
    print(f"  Acurácia 1X2:       {acc:.4f}  (baseline ≈ 0.46)")
    print(f"  Acurácia conf>50%:  {acc_high:.4f}  ({high_conf.sum()} jogos)")
    print(f"{'─' * 58}")
    print(f"  Brier Mandante:     {bs_H:.4f}")
    print(f"  Brier Empate:       {bs_D:.4f}")
    print(f"  Brier Visitante:    {bs_A:.4f}")
    print(f"  Brier Over 2.5:     {bs_ov25:.4f}")
    print(f"  Brier BTTS:         {bs_btts:.4f}")
    print(f"{'─' * 58}")
    print(f"  RMSE λ mandante:    {rmse_h:.4f}")
    print(f"  RMSE λ visitante:   {rmse_a:.4f}")
    print(f"{sep}")

    return bt


bt = run_backtest(models, train_info["df_feat"], train_info, RHO, ALPHA)


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 13 — VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════

def plot_score_heatmap(lam_h, lam_a, home, away, rho=RHO):
    """Heatmap da score matrix."""
    mat = build_score_matrix(lam_h, lam_a, rho=rho, max_g=6)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(mat * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Probabilidade (%)"})
    ax.set_xlabel(f"Gols — {away}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Gols — {home}", fontsize=12, fontweight="bold")
    ax.set_title(f"Placar Exato: {home} × {away}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_market_bars(home, away, pred):
    """Gráficos de barras por categoria de mercado."""
    cats = {
        "1X2": {"Mand.": pred["Vitória Mandante (1)"],
                "Empate": pred["Empate (X)"],
                "Visit.": pred["Vitória Visitante (2)"]},
        "Dupla Chance": {"1X": pred["DC 1X (Mand. ou Empate)"],
                         "X2": pred["DC X2 (Visit. ou Empate)"],
                         "12": pred["DC 12 (Qualquer Vitória)"]},
        "Over/Under": {"Ov1.5": pred["Over 1.5"],
                       "Ov2.5": pred["Over 2.5"],
                       "Ov3.5": pred["Over 3.5"]},
        "Gols Mand.": {"0.5": pred["Over 0.5 Gols (Mand.)"],
                        "1.5": pred["Over 1.5 Gols (Mand.)"],
                        "2.5": pred["Over 2.5 Gols (Mand.)"],
                        "3.5": pred["Over 3.5 Gols (Mand.)"]},
        "Gols Visit.": {"0.5": pred["Over 0.5 Gols (Visit.)"],
                         "1.5": pred["Over 1.5 Gols (Visit.)"],
                         "2.5": pred["Over 2.5 Gols (Visit.)"],
                         "3.5": pred["Over 3.5 Gols (Visit.)"]},
        "BTTS": {"Sim": pred["BTTS Sim"], "Não": pred["BTTS Não"]},
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    cores = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

    for i, (titulo, dados) in enumerate(cats.items()):
        ax = axes[i]
        bars = ax.bar(dados.keys(), [v * 100 for v in dados.values()],
                      color=cores[i], edgecolor="white")
        ax.set_title(titulo, fontweight="bold")
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        for bar, v in zip(bars, dados.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v * 100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{home} × {away}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_calibration(bt, n_bins=8):
    """Diagrama de confiabilidade."""
    if bt is None or bt.empty:
        print("⚠️ Sem dados para calibração")
        return

    mkt = {"Mandante": ("p_H", "y_H"), "Empate": ("p_D", "y_D"),
           "Visitante": ("p_A", "y_A"), "Over 2.5": ("p_ov25", "y_ov25")}
    mkt = {k: v for k, v in mkt.items() if v[0] in bt.columns}

    fig, axes = plt.subplots(1, len(mkt), figsize=(5 * len(mkt), 5))
    if len(mkt) == 1: axes = [axes]

    for ax, (label, (pc, tc)) in zip(axes, mkt.items()):
        dm = bt[[pc, tc]].dropna()
        if len(dm) < 10: continue
        bins = np.linspace(0, 1, n_bins + 1)
        dm["bin"] = pd.cut(dm[pc], bins=bins, include_lowest=True)
        cal = dm.groupby("bin", observed=True).agg(
            pm=(pc, "mean"), fr=(tc, "mean"), n=(pc, "count")).dropna()
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.scatter(cal["pm"], cal["fr"],
                   s=cal["n"] / max(cal["n"].max(), 1) * 200,
                   color="#2196F3", edgecolors="white", zorder=5)
        ax.plot(cal["pm"], cal["fr"], color="#2196F3", lw=1.5, alpha=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
        ax.set_title(label, fontweight="bold")

    plt.suptitle("Calibração do Modelo", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 14 — INTERFACE DE PREVISÃO
# ══════════════════════════════════════════════════════════════════════

def predict_match(home, away, show_plots=True):
    """
    Prevê um jogo específico.
    Busca as features mais recentes de cada time no dataset.
    """
    df_f = train_info["df_feat"]

    # Buscar features mais recentes de cada time
    def latest_feats(team):
        mask = (df_f["home"] == team) | (df_f["away"] == team)
        sub = df_f[mask].sort_values("date")
        if sub.empty:
            return {}
        row = sub.iloc[-1]
        is_h = (row["home"] == team)

        feats = {}
        for c in FEATURE_NAMES:
            val = np.nan
            if c.startswith("H_") and is_h:
                val = row.get(c, np.nan)
            elif c.startswith("H_") and not is_h:
                # Remapear: pegar a versão A_ do mesmo sufixo
                alt = "A_" + c[2:]
                val = row.get(alt, np.nan)
            elif c.startswith("A_") and not is_h:
                val = row.get(c, np.nan)
            elif c.startswith("A_") and is_h:
                alt = "H_" + c[2:]
                val = row.get(alt, np.nan)
            elif not c.startswith(("H_", "A_")):
                val = row.get(c, np.nan)

            try:
                feats[c] = float(val) if pd.notna(val) else np.nan
            except (ValueError, TypeError):
                feats[c] = np.nan
        return feats

    h_feats = latest_feats(home)
    a_feats = latest_feats(away)

    # Combina: para features H_ usa mandante, A_ usa visitante, diff_ recalcula
    feat_row = {}
    for c in FEATURE_NAMES:
        if c.startswith("H_"):
            feat_row[c] = h_feats.get(c, np.nan)
        elif c.startswith("A_"):
            feat_row[c] = a_feats.get(c, np.nan)
        elif c.startswith("diff_"):
            # Tenta recalcular
            parts = c.split("_", 2)  # diff_5_xgf
            if len(parts) >= 3:
                w_col = parts[1]
                metric = "_".join(parts[2:])
                hv = h_feats.get(f"H_r{w_col}_{metric}", np.nan)
                av = a_feats.get(f"A_r{w_col}_{metric}", np.nan)
                if pd.notna(hv) and pd.notna(av):
                    feat_row[c] = hv - av
                else:
                    feat_row[c] = np.nan
            else:
                feat_row[c] = np.nan
        elif "home" in c:
            feat_row[c] = h_feats.get(c, np.nan)
        elif "away" in c:
            feat_row[c] = a_feats.get(c, np.nan)
        elif c.startswith("h2h_"):
            # H2H: pegar do último confronto direto
            h2h_mask = ((df_f["home"] == home) & (df_f["away"] == away)) | \
                       ((df_f["home"] == away) & (df_f["away"] == home))
            h2h_sub = df_f[h2h_mask].sort_values("date")
            if not h2h_sub.empty:
                try:
                    feat_row[c] = float(h2h_sub.iloc[-1].get(c, np.nan))
                except (ValueError, TypeError):
                    feat_row[c] = np.nan
            else:
                feat_row[c] = np.nan
        else:
            feat_row[c] = h_feats.get(c, a_feats.get(c, np.nan))

    X = fillna_smart(pd.DataFrame([feat_row])[FEATURE_NAMES].astype(float), MEDIANS)
    pred = predict_match_probs(X, models, rho=RHO, alpha=ALPHA)

    # ── Exibição ──
    sep = "═" * 58
    print(f"\n{sep}")
    print(f"  ⚽  {home.upper()}  ×  {away.upper()}")
    print(f"{sep}")

    sections = {
        "⚡ xG PREVISTO": [
            ("λ Mandante", pred["lambda_home"], "gol"),
            ("λ Visitante", pred["lambda_away"], "gol"),
        ],
        "📊 RESULTADO 1X2": [
            ("Vitória Mandante (1)", pred["Vitória Mandante (1)"], "%"),
            ("Empate (X)", pred["Empate (X)"], "%"),
            ("Vitória Visitante (2)", pred["Vitória Visitante (2)"], "%"),
        ],
        "🔄 EMPATE ANULA APOSTA": [
            ("DNB Mandante", pred["DNB Mandante"], "%"),
            ("DNB Visitante", pred["DNB Visitante"], "%"),
        ],
        "🎯 DUPLA CHANCE": [
            ("DC 1X", pred["DC 1X (Mand. ou Empate)"], "%"),
            ("DC X2", pred["DC X2 (Visit. ou Empate)"], "%"),
            ("DC 12", pred["DC 12 (Qualquer Vitória)"], "%"),
        ],
        "⚽ TOTAL DE GOLS": [
            ("Over 1.5", pred["Over 1.5"], "%"),
            ("Over 2.5", pred["Over 2.5"], "%"),
            ("Over 3.5", pred["Over 3.5"], "%"),
        ],
        "🏠 GOLS MANDANTE": [
            ("Over 0.5", pred["Over 0.5 Gols (Mand.)"], "%"),
            ("Over 1.5", pred["Over 1.5 Gols (Mand.)"], "%"),
            ("Over 2.5", pred["Over 2.5 Gols (Mand.)"], "%"),
            ("Over 3.5", pred["Over 3.5 Gols (Mand.)"], "%"),
        ],
        "✈️ GOLS VISITANTE": [
            ("Over 0.5", pred["Over 0.5 Gols (Visit.)"], "%"),
            ("Over 1.5", pred["Over 1.5 Gols (Visit.)"], "%"),
            ("Over 2.5", pred["Over 2.5 Gols (Visit.)"], "%"),
            ("Over 3.5", pred["Over 3.5 Gols (Visit.)"], "%"),
        ],
        "🔵 BTTS": [
            ("Sim", pred["BTTS Sim"], "%"),
            ("Não", pred["BTTS Não"], "%"),
        ],
    }

    for section, items in sections.items():
        print(f"\n  {section}")
        for name, val, fmt in items:
            if fmt == "%":
                print(f"    {name:35s} {val * 100:6.1f}%")
            else:
                print(f"    {name:35s} {val:6.2f}")

    print(f"\n{sep}")

    if show_plots:
        plot_score_heatmap(pred["lambda_home"], pred["lambda_away"], home, away)
        plot_market_bars(home, away, pred)

    return pred


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 15 — PREVISÃO EM LOTE
# ══════════════════════════════════════════════════════════════════════

def predict_batch(jogos):
    """Prevê lista de jogos [{home, away, rodada?, date?}]."""
    rows = []
    n_err = 0
    for j in tqdm(jogos, desc="Prevendo"):
        try:
            pred = predict_match(j["home"], j["away"], show_plots=False)
            row = {"rodada": j.get("rodada", ""), "date": j.get("date", ""),
                   "home": j["home"], "away": j["away"]}
            row.update(pred)
            rows.append(row)
        except Exception as e:
            n_err += 1
            log.warning(f"Erro: {j.get('home')} × {j.get('away')}: {e}")

    if n_err:
        print(f"⚠️ {n_err}/{len(jogos)} falharam")
    if not rows:
        print("❌ Nenhuma previsão gerada")
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_predict_excel(path, export_csv=True):
    """Carrega Excel e gera previsões."""
    print(f"\n📂 Carregando: {path}")
    dj = pd.read_excel(path)
    dj.columns = dj.columns.str.strip().str.lower()

    for old, new in [("mandante", "home"), ("visitante", "away")]:
        if old in dj.columns:
            dj = dj.rename(columns={old: new})

    required = {"home", "away"}
    missing = required - set(dj.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    dj["home"] = dj["home"].str.strip()
    dj["away"] = dj["away"].str.strip()
    if "date" in dj.columns:
        dj["date"] = pd.to_datetime(dj["date"], dayfirst=True, errors="coerce")

    jogos = dj.to_dict("records")
    df_prev = predict_batch(jogos)

    if export_csv and not df_prev.empty:
        out_path = path.replace(".xlsx", "").replace(".xls", "") + "_previsoes.csv"
        df_prev.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"💾 Exportado: {out_path}")

    return df_prev


# ══════════════════════════════════════════════════════════════════════
# CÉLULA 16 — EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════

# Times disponíveis
print("\n🏟️ Times disponíveis:")
print(sorted(df["home"].unique()))

# Exemplo de previsão
HOME = "Manchester City"
AWAY = "Arsenal"
pred_exemplo = predict_match(HOME, AWAY, show_plots=True)

# Calibração
if bt is not None and not bt.empty:
    plot_calibration(bt)

# Previsão via Excel (se existir)
EXCEL_PATH = "/content/rodada.xlsx"
if os.path.exists(EXCEL_PATH):
    df_excel = load_predict_excel(EXCEL_PATH)
else:
    print(f"\n💡 Para previsão em lote: faça upload de '{EXCEL_PATH}'")
    print("   Colunas: home | away | rodada (opcional) | date (opcional)")
