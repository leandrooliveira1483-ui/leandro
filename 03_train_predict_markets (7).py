# Etapa 03 - Treino, calibração, avaliação probabilística e backtest com odds.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path("data")
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "model_dataset.parquet"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}. Rode primeiro a etapa 02_build_features.py")
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    if "season" in df.columns:
        season_str = df["season"].astype(str)
        df["season_num"] = pd.to_numeric(season_str.str.extract(r"(\d{4})", expand=False), errors="coerce")
    return df


def build_feature_list(df: pd.DataFrame) -> Tuple[list[str], list[str], list[str]]:
    base_num = [c for c in df.columns if ("avg_" in c or c.startswith("diff_") or "rest_" in c or "games_last_" in c)]
    if "season_num" in df.columns:
        base_num.append("season_num")
    cat_cols = [c for c in ["league", "home_team", "away_team"] if c in df.columns]

    must_have = ["home_goals", "away_goals", "date"]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset sem colunas necessárias: {missing}")

    feature_cols = base_num + cat_cols
    if not feature_cols:
        raise ValueError("Nenhuma feature foi selecionada. Verifique etapa 02.")

    return feature_cols, base_num, cat_cols


def build_model(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    model = Pipeline(steps=[("prep", preprocessor), ("reg", PoissonRegressor(alpha=0.2, max_iter=500))])
    return model


def fit_models(train_df: pd.DataFrame, num_cols: list[str], cat_cols: list[str], features: list[str], time_decay_xi: float = 0.003):
    clean_train = train_df.dropna(subset=["home_goals", "away_goals"]).copy()
    x_train = clean_train[features]
    days_since = (clean_train["date"].max() - clean_train["date"]).dt.days.fillna(0).clip(lower=0)
    sample_weights = np.exp(-time_decay_xi * days_since.to_numpy())

    m_home = build_model(num_cols, cat_cols)
    m_away = build_model(num_cols, cat_cols)

    m_home.fit(x_train, clean_train["home_goals"], reg__sample_weight=sample_weights)
    m_away.fit(x_train, clean_train["away_goals"], reg__sample_weight=sample_weights)
    return m_home, m_away


def poisson_probs(lam: np.ndarray, max_goals: int = 8) -> np.ndarray:
    goals = np.arange(0, max_goals + 1)
    fact = np.array([math.factorial(int(g)) for g in goals])
    probs = np.exp(-lam[:, None]) * (lam[:, None] ** goals[None, :]) / fact[None, :]
    tail = 1 - probs.sum(axis=1, keepdims=True)
    probs[:, -1:] = probs[:, -1:] + np.clip(tail, 0, None)
    return probs


def apply_dixon_coles_adjustment(score_matrix: np.ndarray, home_lambda: np.ndarray, away_lambda: np.ndarray, rho: float) -> np.ndarray:
    out = score_matrix.copy()
    lh = home_lambda.astype(float)
    la = away_lambda.astype(float)
    tau00 = np.clip(1 - lh * la * rho, 1e-9, None)
    tau01 = np.clip(1 + lh * rho, 1e-9, None)
    tau10 = np.clip(1 + la * rho, 1e-9, None)
    tau11 = np.clip(1 - rho, 1e-9, None)

    out[:, 0, 0] *= tau00
    out[:, 0, 1] *= tau01
    out[:, 1, 0] *= tau10
    out[:, 1, 1] *= tau11
    out /= np.clip(out.sum(axis=(1, 2), keepdims=True), 1e-12, None)
    return out


def estimate_dixon_coles_rho(y_home: np.ndarray, y_away: np.ndarray, home_lambda: np.ndarray, away_lambda: np.ndarray) -> float:
    p_home = poisson_probs(home_lambda)
    p_away = poisson_probs(away_lambda)
    base_matrix = p_home[:, :, None] * p_away[:, None, :]

    # Busca em duas etapas: coarse + refinamento local (mais robusto que grid único fixo)
    coarse = np.linspace(-0.2, 0.2, 81)
    best_rho = 0.0
    best_ll = -1e18

    for rho in coarse:
        adj = apply_dixon_coles_adjustment(base_matrix, home_lambda, away_lambda, float(rho))
        ll = 0.0
        for i in range(len(y_home)):
            gh = int(min(max(y_home[i], 0), adj.shape[1] - 1))
            ga = int(min(max(y_away[i], 0), adj.shape[2] - 1))
            ll += math.log(max(adj[i, gh, ga], 1e-12))
        if ll > best_ll:
            best_ll = ll
            best_rho = float(rho)
    fine = np.linspace(best_rho - 0.02, best_rho + 0.02, 81)
    for rho in fine:
        adj = apply_dixon_coles_adjustment(base_matrix, home_lambda, away_lambda, float(rho))
        ll = 0.0
        for i in range(len(y_home)):
            gh = int(min(max(y_home[i], 0), adj.shape[1] - 1))
            ga = int(min(max(y_away[i], 0), adj.shape[2] - 1))
            ll += math.log(max(adj[i, gh, ga], 1e-12))
        if ll > best_ll:
            best_ll = ll
            best_rho = float(rho)
    return best_rho


def market_probabilities(home_lambda: np.ndarray, away_lambda: np.ndarray, max_goals: int = 8, rho: float = 0.0) -> Dict[str, np.ndarray]:
    p_home = poisson_probs(home_lambda, max_goals=max_goals)
    p_away = poisson_probs(away_lambda, max_goals=max_goals)

    score_matrix = p_home[:, :, None] * p_away[:, None, :]
    if abs(rho) > 1e-12:
        score_matrix = apply_dixon_coles_adjustment(score_matrix, home_lambda, away_lambda, rho)

    idx = np.arange(max_goals + 1)
    home_win = (idx[:, None] > idx[None, :]).astype(float)
    draw = (idx[:, None] == idx[None, :]).astype(float)
    away_win = (idx[:, None] < idx[None, :]).astype(float)
    total_goals = idx[:, None] + idx[None, :]
    btts_yes_mask = ((idx[:, None] >= 1) & (idx[None, :] >= 1)).astype(float)

    def mat_prob(mask: np.ndarray) -> np.ndarray:
        return (score_matrix * mask[None, :, :]).sum(axis=(1, 2))

    p_home_win = mat_prob(home_win)
    p_draw = mat_prob(draw)
    p_away_win = mat_prob(away_win)

    p_1x = p_home_win + p_draw
    p_x2 = p_away_win + p_draw
    p_12 = p_home_win + p_away_win

    p_over_15 = mat_prob((total_goals > 1.5).astype(float))
    p_over_25 = mat_prob((total_goals > 2.5).astype(float))
    p_over_35 = mat_prob((total_goals > 3.5).astype(float))
    p_under_15 = 1 - p_over_15
    p_under_25 = 1 - p_over_25
    p_under_35 = 1 - p_over_35

    home_marginal = score_matrix.sum(axis=2)
    away_marginal = score_matrix.sum(axis=1)
    p_home_over_05 = 1 - home_marginal[:, 0]
    p_home_over_15 = 1 - (home_marginal[:, 0] + home_marginal[:, 1])
    p_home_over_25 = 1 - (home_marginal[:, 0] + home_marginal[:, 1] + home_marginal[:, 2])
    p_home_over_35 = 1 - (home_marginal[:, 0] + home_marginal[:, 1] + home_marginal[:, 2] + home_marginal[:, 3])

    p_away_over_05 = 1 - away_marginal[:, 0]
    p_away_over_15 = 1 - (away_marginal[:, 0] + away_marginal[:, 1])
    p_away_over_25 = 1 - (away_marginal[:, 0] + away_marginal[:, 1] + away_marginal[:, 2])
    p_away_over_35 = 1 - (away_marginal[:, 0] + away_marginal[:, 1] + away_marginal[:, 2] + away_marginal[:, 3])
    p_btts_yes = mat_prob(btts_yes_mask)
    p_btts_no = 1 - p_btts_yes

    eps = 1e-12
    p_home_dnb = p_home_win / np.clip((1 - p_draw), eps, None)
    p_away_dnb = p_away_win / np.clip((1 - p_draw), eps, None)

    return {
        "p_home_win": p_home_win,
        "p_draw": p_draw,
        "p_away_win": p_away_win,
        "p_home_dnb": p_home_dnb,
        "p_away_dnb": p_away_dnb,
        "p_double_chance_1x": p_1x,
        "p_double_chance_x2": p_x2,
        "p_double_chance_12": p_12,
        "p_over_1_5": p_over_15,
        "p_over_2_5": p_over_25,
        "p_over_3_5": p_over_35,
        "p_under_1_5": p_under_15,
        "p_under_2_5": p_under_25,
        "p_under_3_5": p_under_35,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": p_btts_no,
        "p_home_over_0_5": p_home_over_05,
        "p_home_over_1_5": p_home_over_15,
        "p_home_over_2_5": p_home_over_25,
        "p_home_over_3_5": p_home_over_35,
        "p_away_over_0_5": p_away_over_05,
        "p_away_over_1_5": p_away_over_15,
        "p_away_over_2_5": p_away_over_25,
        "p_away_over_3_5": p_away_over_35,
    }


def split_train_calib(train_df: pd.DataFrame, calib_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sort_values("date").reset_index(drop=True)
    cut = int(len(train_df) * (1 - calib_frac))
    return train_df.iloc[:cut].copy(), train_df.iloc[cut:].copy()


def fit_probability_calibrators(calib_df: pd.DataFrame, raw_prob_df: pd.DataFrame, method: str = "isotonic") -> Dict[str, object]:
    calibrators: Dict[str, object] = {}
    markets_targets = {
        "p_home_win": (calib_df["home_goals"] > calib_df["away_goals"]).astype(int).to_numpy(),
        "p_draw": (calib_df["home_goals"] == calib_df["away_goals"]).astype(int).to_numpy(),
        "p_away_win": (calib_df["home_goals"] < calib_df["away_goals"]).astype(int).to_numpy(),
        "p_over_2_5": ((calib_df["home_goals"] + calib_df["away_goals"]) > 2.5).astype(int).to_numpy(),
    }

    for market, y in markets_targets.items():
        p = np.clip(raw_prob_df[market].to_numpy(), 1e-6, 1 - 1e-6)
        if len(np.unique(y)) < 2:
            continue

        if method == "platt":
            model = LogisticRegression()
            model.fit(p.reshape(-1, 1), y)
            calibrators[market] = model
        else:
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p, y)
            calibrators[market] = model
    return calibrators


def apply_calibration(prob_df: pd.DataFrame, calibrators: Dict[str, object], method: str = "isotonic") -> pd.DataFrame:
    out = prob_df.copy()
    for market, model in calibrators.items():
        p = np.clip(out[market].to_numpy(), 1e-6, 1 - 1e-6)
        if method == "platt":
            out[f"{market}_cal"] = model.predict_proba(p.reshape(-1, 1))[:, 1]
        else:
            out[f"{market}_cal"] = model.predict(p)

    # Re-normalização 1X2 após calibração (consistência probabilística)
    cols_1x2 = ["p_home_win_cal", "p_draw_cal", "p_away_win_cal"]
    if all(c in out.columns for c in cols_1x2):
        s = out[cols_1x2].sum(axis=1).replace(0, np.nan)
        out["p_home_win_cal"] = (out["p_home_win_cal"] / s).clip(1e-6, 1 - 1e-6)
        out["p_draw_cal"] = (out["p_draw_cal"] / s).clip(1e-6, 1 - 1e-6)
        out["p_away_win_cal"] = (out["p_away_win_cal"] / s).clip(1e-6, 1 - 1e-6)
        out["p_double_chance_1x_cal"] = out["p_home_win_cal"] + out["p_draw_cal"]
        out["p_double_chance_x2_cal"] = out["p_away_win_cal"] + out["p_draw_cal"]
        out["p_double_chance_12_cal"] = out["p_home_win_cal"] + out["p_away_win_cal"]
        denom = np.clip(1 - out["p_draw_cal"], 1e-12, None)
        out["p_home_dnb_cal"] = out["p_home_win_cal"] / denom
        out["p_away_dnb_cal"] = out["p_away_win_cal"] / denom
    return out


def predict_markets_for_matches(df_matches: pd.DataFrame, model_home: Pipeline, model_away: Pipeline, features: list[str], rho: float = 0.0) -> pd.DataFrame:
    if df_matches.empty:
        return df_matches.copy()

    x_data = df_matches[features]
    pred_home_lambda = model_home.predict(x_data)
    pred_away_lambda = model_away.predict(x_data)

    markets = market_probabilities(pred_home_lambda, pred_away_lambda, max_goals=8, rho=rho)

    out = df_matches[["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    out["pred_home_lambda"] = pred_home_lambda
    out["pred_away_lambda"] = pred_away_lambda
    for k, v in markets.items():
        out[k] = v
    return out


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def log_loss_binary(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": i,
            "p_mean": float(p[mask].mean()),
            "y_mean": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return rows


def evaluate_probabilities(eval_df: pd.DataFrame, use_calibrated: bool = True) -> dict:
    if eval_df.empty:
        return {}

    report: dict = {}
    target_map = {
        "p_home_win": (eval_df["home_goals"] > eval_df["away_goals"]).astype(int).to_numpy(),
        "p_draw": (eval_df["home_goals"] == eval_df["away_goals"]).astype(int).to_numpy(),
        "p_away_win": (eval_df["home_goals"] < eval_df["away_goals"]).astype(int).to_numpy(),
        "p_over_2_5": ((eval_df["home_goals"] + eval_df["away_goals"]) > 2.5).astype(int).to_numpy(),
        "p_under_2_5": ((eval_df["home_goals"] + eval_df["away_goals"]) < 2.5).astype(int).to_numpy(),
        "p_btts_yes": ((eval_df["home_goals"] > 0) & (eval_df["away_goals"] > 0)).astype(int).to_numpy(),
        "p_btts_no": ((eval_df["home_goals"] == 0) | (eval_df["away_goals"] == 0)).astype(int).to_numpy(),
    }

    for market, y in target_map.items():
        col = f"{market}_cal" if use_calibrated and f"{market}_cal" in eval_df.columns else market
        p = eval_df[col].to_numpy()
        report[market] = {
            "prob_col_used": col,
            "brier": brier_score(y, p),
            "log_loss": log_loss_binary(y, p),
            "calibration": calibration_table(y, p, n_bins=10),
        }
    return report


def compute_psi(train_values: pd.Series, score_values: pd.Series, bins: int = 10) -> float:
    train_values = train_values.dropna().to_numpy()
    score_values = score_values.dropna().to_numpy()
    if len(train_values) < 10 or len(score_values) < 10:
        return float("nan")

    qs = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(train_values, qs)
    cuts = np.unique(cuts)
    if len(cuts) < 3:
        return float("nan")

    train_hist, _ = np.histogram(train_values, bins=cuts)
    score_hist, _ = np.histogram(score_values, bins=cuts)

    train_pct = np.clip(train_hist / max(train_hist.sum(), 1), 1e-6, 1)
    score_pct = np.clip(score_hist / max(score_hist.sum(), 1), 1e-6, 1)
    psi = np.sum((score_pct - train_pct) * np.log(score_pct / train_pct))
    return float(psi)


def drift_report(train_df: pd.DataFrame, score_df: pd.DataFrame, numeric_features: list[str]) -> dict:
    report = {}
    for col in numeric_features:
        if col in train_df.columns and col in score_df.columns:
            report[col] = compute_psi(train_df[col], score_df[col], bins=10)
    return report


def merge_odds_and_backtest(eval_df: pd.DataFrame, odds_file: str | None) -> tuple[pd.DataFrame, dict]:
    if not odds_file:
        return eval_df, {}

    odds_path = Path(odds_file)
    if not odds_path.exists():
        return eval_df, {"warning": f"odds_file não encontrado: {odds_file}"}

    odds = pd.read_csv(odds_path)
    if "date" in odds.columns:
        odds["date"] = pd.to_datetime(odds["date"], errors="coerce")

    key_cols = ["date", "league", "home_team", "away_team"]
    required = set(key_cols + ["odds_home", "odds_draw", "odds_away"])
    if not required.issubset(set(odds.columns)):
        return eval_df, {"warning": f"odds_file precisa de colunas: {sorted(required)}"}

    merged = eval_df.merge(odds, on=key_cols, how="left")

    # Estratégia simples: aposta 1 unidade quando EV > 0
    p_home_col = "p_home_win_cal" if "p_home_win_cal" in merged.columns else "p_home_win"
    p_draw_col = "p_draw_cal" if "p_draw_cal" in merged.columns else "p_draw"
    p_away_col = "p_away_win_cal" if "p_away_win_cal" in merged.columns else "p_away_win"

    merged["ev_home"] = merged[p_home_col] * merged["odds_home"] - 1
    merged["ev_draw"] = merged[p_draw_col] * merged["odds_draw"] - 1
    merged["ev_away"] = merged[p_away_col] * merged["odds_away"] - 1

    merged["bet_home"] = (merged["ev_home"] > 0).astype(int)
    merged["bet_draw"] = (merged["ev_draw"] > 0).astype(int)
    merged["bet_away"] = (merged["ev_away"] > 0).astype(int)

    home_win = (merged["home_goals"] > merged["away_goals"]).astype(int)
    draw = (merged["home_goals"] == merged["away_goals"]).astype(int)
    away_win = (merged["home_goals"] < merged["away_goals"]).astype(int)

    merged["profit_home"] = merged["bet_home"] * (home_win * (merged["odds_home"] - 1) - (1 - home_win))
    merged["profit_draw"] = merged["bet_draw"] * (draw * (merged["odds_draw"] - 1) - (1 - draw))
    merged["profit_away"] = merged["bet_away"] * (away_win * (merged["odds_away"] - 1) - (1 - away_win))

    total_bets = float(merged[["bet_home", "bet_draw", "bet_away"]].sum().sum())
    total_profit = float(merged[["profit_home", "profit_draw", "profit_away"]].sum().sum())

    clv_report = {}
    if {"open_odds_home", "open_odds_draw", "open_odds_away"}.issubset(merged.columns):
        merged["clv_home"] = (1 / merged["open_odds_home"]) - (1 / merged["odds_home"])
        merged["clv_draw"] = (1 / merged["open_odds_draw"]) - (1 / merged["odds_draw"])
        merged["clv_away"] = (1 / merged["open_odds_away"]) - (1 / merged["odds_away"])
        clv_report = {
            "mean_clv_home": float(merged.loc[merged["bet_home"] == 1, "clv_home"].mean()),
            "mean_clv_draw": float(merged.loc[merged["bet_draw"] == 1, "clv_draw"].mean()),
            "mean_clv_away": float(merged.loc[merged["bet_away"] == 1, "clv_away"].mean()),
        }

    report = {
        "n_matches_with_odds": int(merged["odds_home"].notna().sum()),
        "prob_columns_used_for_ev": {"home": p_home_col, "draw": p_draw_col, "away": p_away_col},
        "total_bets": total_bets,
        "total_profit_units": total_profit,
        "roi": float(total_profit / total_bets) if total_bets > 0 else None,
        **clv_report,
    }

    # Benchmark vs mercado (implied probabilities de closing odds)
    valid = merged[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1)
    if valid.any():
        imp_home = 1 / merged.loc[valid, "odds_home"].to_numpy()
        imp_draw = 1 / merged.loc[valid, "odds_draw"].to_numpy()
        imp_away = 1 / merged.loc[valid, "odds_away"].to_numpy()
        imp_sum = np.clip(imp_home + imp_draw + imp_away, 1e-12, None)
        imp_home, imp_draw, imp_away = imp_home / imp_sum, imp_draw / imp_sum, imp_away / imp_sum

        y_home = (merged.loc[valid, "home_goals"] > merged.loc[valid, "away_goals"]).astype(int).to_numpy()
        y_draw = (merged.loc[valid, "home_goals"] == merged.loc[valid, "away_goals"]).astype(int).to_numpy()
        y_away = (merged.loc[valid, "home_goals"] < merged.loc[valid, "away_goals"]).astype(int).to_numpy()

        model_home = merged.loc[valid, p_home_col].to_numpy()
        model_draw = merged.loc[valid, p_draw_col].to_numpy()
        model_away = merged.loc[valid, p_away_col].to_numpy()

        model_brier_multi = float(np.mean((model_home - y_home) ** 2 + (model_draw - y_draw) ** 2 + (model_away - y_away) ** 2))
        market_brier_multi = float(np.mean((imp_home - y_home) ** 2 + (imp_draw - y_draw) ** 2 + (imp_away - y_away) ** 2))

        y_idx = np.where(y_home == 1, 0, np.where(y_draw == 1, 1, 2))
        model_probs = np.vstack([model_home, model_draw, model_away]).T
        market_probs = np.vstack([imp_home, imp_draw, imp_away]).T
        eps = 1e-12
        model_logloss_multi = float(-np.mean(np.log(np.clip(model_probs[np.arange(len(y_idx)), y_idx], eps, 1))))
        market_logloss_multi = float(-np.mean(np.log(np.clip(market_probs[np.arange(len(y_idx)), y_idx], eps, 1))))

        report["market_benchmark"] = {
            "n_matches": int(valid.sum()),
            "model_brier_multi_1x2": model_brier_multi,
            "market_brier_multi_1x2": market_brier_multi,
            "model_logloss_multi_1x2": model_logloss_multi,
            "market_logloss_multi_1x2": market_logloss_multi,
        }

    return merged, report


def rolling_backtest_summary(
    played_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    calibration_method: str,
    n_windows: int = 4,
    window_size: int = 120,
) -> dict:
    played_df = played_df.sort_values("date").reset_index(drop=True)
    results = []

    for w in range(n_windows):
        test_end = len(played_df) - w * window_size
        test_start = test_end - window_size
        if test_start <= 300:
            break

        train_df = played_df.iloc[:test_start].copy()
        test_df = played_df.iloc[test_start:test_end].copy()

        usable_num_cols = [c for c in num_cols if c in train_df.columns and train_df[c].notna().any()]
        features = usable_num_cols + cat_cols
        if not features:
            continue

        fit_df, calib_df = split_train_calib(train_df, calib_frac=0.2)
        hm, am = fit_models(fit_df, usable_num_cols, cat_cols, features)

        calib_pred = predict_markets_for_matches(calib_df, hm, am, features, rho=0.0)
        rho = estimate_dixon_coles_rho(
            calib_df["home_goals"].to_numpy(dtype=int),
            calib_df["away_goals"].to_numpy(dtype=int),
            calib_pred["pred_home_lambda"].to_numpy(),
            calib_pred["pred_away_lambda"].to_numpy(),
        )
        calib_pred = predict_markets_for_matches(calib_df, hm, am, features, rho=rho)
        calibrators = fit_probability_calibrators(calib_df, calib_pred, method=calibration_method)

        pred = predict_markets_for_matches(test_df, hm, am, features, rho=rho)
        pred = apply_calibration(pred, calibrators, method=calibration_method)

        rep = evaluate_probabilities(pred, use_calibrated=True)
        results.append({
            "window": w + 1,
            "date_from": str(test_df["date"].min()),
            "date_to": str(test_df["date"].max()),
            "n_matches": int(len(test_df)),
            "mae_home": float(mean_absolute_error(test_df["home_goals"], pred["pred_home_lambda"])),
            "mae_away": float(mean_absolute_error(test_df["away_goals"], pred["pred_away_lambda"])),
            "brier_home_win": rep.get("p_home_win", {}).get("brier"),
            "brier_draw": rep.get("p_draw", {}).get("brier"),
            "brier_away_win": rep.get("p_away_win", {}).get("brier"),
        })

    return {"n_windows": len(results), "window_size": window_size, "results": results}


def filter_by_date_range(df: pd.DataFrame, start_date: pd.Timestamp | None = None, end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    if start_date is not None:
        out = out[out["date"] >= start_date]
    if end_date is not None:
        out = out[out["date"] <= end_date]
    return out


def _parse_optional_date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.Timestamp(value)


def run_pipeline(
    train_end_date: str | None = None,
    predict_start_date: str | None = None,
    predict_end_date: str | None = None,
    calibration_method: str = "isotonic",
    odds_file: str | None = None,
    rolling_windows: int = 4,
    rolling_window_size: int = 120,
) -> None:
    df = load_data()
    _features, num_cols, cat_cols = build_feature_list(df)

    played_mask = df["home_goals"].notna() & df["away_goals"].notna()
    played_df = df[played_mask].copy()

    train_end_ts = _parse_optional_date(train_end_date)
    predict_start_ts = _parse_optional_date(predict_start_date)
    predict_end_ts = _parse_optional_date(predict_end_date)

    if len(played_df) < 100:
        raise ValueError("Poucos jogos finalizados para treino. Verifique os dados extraídos.")

    if train_end_ts is not None:
        train_df = played_df[played_df["date"] <= train_end_ts].copy()
        scoring_pool = df[df["date"] > train_end_ts].copy()
    else:
        train_df = played_df.sort_values("date").iloc[:-500].copy()
        scoring_pool = played_df.sort_values("date").iloc[-500:].copy()

    if len(train_df) < 200:
        raise ValueError("Treino muito pequeno para etapa avançada.")

    scoring_pool = filter_by_date_range(scoring_pool, start_date=predict_start_ts, end_date=predict_end_ts)

    usable_num_cols = [c for c in num_cols if c in train_df.columns and train_df[c].notna().any()]
    features = usable_num_cols + cat_cols
    if not features:
        raise ValueError("Nenhuma feature disponível após filtrar colunas totalmente vazias no treino.")

    fit_df, calib_df = split_train_calib(train_df, calib_frac=0.2)
    home_model, away_model = fit_models(fit_df, usable_num_cols, cat_cols, features)

    # Estima rho de Dixon-Coles usando janela de calibração
    calib_pred = predict_markets_for_matches(calib_df, home_model, away_model, features, rho=0.0)
    rho = estimate_dixon_coles_rho(
        calib_df["home_goals"].to_numpy(dtype=int),
        calib_df["away_goals"].to_numpy(dtype=int),
        calib_pred["pred_home_lambda"].to_numpy(),
        calib_pred["pred_away_lambda"].to_numpy(),
    )

    # Refaz previsões com ajuste Dixon-Coles
    calib_pred = predict_markets_for_matches(calib_df, home_model, away_model, features, rho=rho)
    calibrators = fit_probability_calibrators(calib_df, calib_pred, method=calibration_method)

    scored_out = predict_markets_for_matches(scoring_pool, home_model, away_model, features, rho=rho)
    scored_out = apply_calibration(scored_out, calibrators, method=calibration_method)
    scored_out = scored_out.sort_values(["date", "league", "home_team", "away_team"]).reset_index(drop=True)

    window_all_csv = OUTPUT_DIR / "predictions_markets_window_all.csv"
    window_all_xlsx = OUTPUT_DIR / "predictions_markets_window_all.xlsx"
    scored_out.to_csv(window_all_csv, index=False)
    scored_out.to_excel(window_all_xlsx, index=False)

    eval_out = scored_out[scored_out["home_goals"].notna() & scored_out["away_goals"].notna()].copy()
    future_out = scored_out[scored_out["home_goals"].isna() | scored_out["away_goals"].isna()].copy()

    eval_out_with_odds, odds_report = merge_odds_and_backtest(eval_out, odds_file=odds_file)

    eval_csv = OUTPUT_DIR / "predictions_markets_backtest.csv"
    eval_xlsx = OUTPUT_DIR / "predictions_markets_backtest.xlsx"
    eval_out_with_odds.to_csv(eval_csv, index=False)
    eval_out_with_odds.to_excel(eval_xlsx, index=False)

    future_csv = OUTPUT_DIR / "predictions_markets_future.csv"
    future_xlsx = OUTPUT_DIR / "predictions_markets_future.xlsx"
    future_out.to_csv(future_csv, index=False)
    future_out.to_excel(future_xlsx, index=False)

    future_out.to_csv(OUTPUT_DIR / "predictions_markets.csv", index=False)
    future_out.to_excel(OUTPUT_DIR / "predictions_markets.xlsx", index=False)

    prob_eval_report = evaluate_probabilities(eval_out_with_odds, use_calibrated=True)
    drift = drift_report(train_df, scoring_pool, usable_num_cols)
    rolling_report = rolling_backtest_summary(
        played_df=played_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        calibration_method=calibration_method,
        n_windows=rolling_windows,
        window_size=rolling_window_size,
    )

    metrics = {
        "train_end_date": str(train_end_ts.date()) if train_end_ts is not None else None,
        "predict_start_date": str(predict_start_ts.date()) if predict_start_ts is not None else None,
        "predict_end_date": str(predict_end_ts.date()) if predict_end_ts is not None else None,
        "calibration_method": calibration_method,
        "dixon_coles_rho": rho,
        "n_train": int(len(train_df)),
        "n_fit": int(len(fit_df)),
        "n_calib": int(len(calib_df)),
        "n_scoring_pool": int(len(scored_out)),
        "n_backtest_eval_matches": int(len(eval_out_with_odds)),
        "n_future_matches": int(len(future_out)),
        "features_used": features,
        "mae_home_goals_backtest": float(mean_absolute_error(eval_out_with_odds["home_goals"], eval_out_with_odds["pred_home_lambda"])) if len(eval_out_with_odds) else None,
        "mae_away_goals_backtest": float(mean_absolute_error(eval_out_with_odds["away_goals"], eval_out_with_odds["pred_away_lambda"])) if len(eval_out_with_odds) else None,
        "probabilistic_report": prob_eval_report,
        "odds_backtest_report": odds_report,
        "drift_psi_report": drift,
        "rolling_backtest_report": rolling_report,
    }

    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=== Treino e previsão concluídos ===")
    print(f"Backtest (jogos já encerrados): {eval_csv}")
    print(f"Futuros (arquivo principal):    {future_csv}")
    print(f"Janela completa simulada:       {window_all_csv}")
    print(f"Métricas avançadas:             {OUTPUT_DIR / 'model_metrics.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina modelo e gera previsões de mercados.")
    parser.add_argument("--train-end-date", type=str, default=None, help="Data limite de treino (YYYY-MM-DD).")
    parser.add_argument("--predict-start-date", type=str, default=None, help="Data mínima da janela de previsão (YYYY-MM-DD).")
    parser.add_argument("--predict-end-date", type=str, default=None, help="Data máxima da janela de previsão (YYYY-MM-DD).")
    parser.add_argument("--calibration-method", type=str, choices=["isotonic", "platt"], default="isotonic", help="Método de calibração")
    parser.add_argument("--odds-file", type=str, default=None, help="CSV com odds históricas para EV/ROI/CLV")
    parser.add_argument("--rolling-windows", type=int, default=4, help="Número de janelas no rolling backtest")
    parser.add_argument("--rolling-window-size", type=int, default=120, help="Quantidade de jogos por janela no rolling backtest")

    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        train_end_date=args.train_end_date,
        predict_start_date=args.predict_start_date,
        predict_end_date=args.predict_end_date,
        calibration_method=args.calibration_method,
        odds_file=args.odds_file,
        rolling_windows=args.rolling_windows,
        rolling_window_size=args.rolling_window_size,
    )
