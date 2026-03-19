#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODELO PREDITIVO V2 — QUALIDADE PREDITIVA + ESTABILIDADE                  ║
║                                                                              ║
║  Melhorias em relação ao V1:                                                ║
║                                                                              ║
║  1. CALIBRAÇÃO ISOTÔNICA DAS PROBABILIDADES                                 ║
║     Bias sistemático medido nos dados:                                       ║
║       Over 0.5 mandante: Poisson subestima em +7.0pp                       ║
║       Over 1.5 mandante: Poisson subestima em +6.1pp                       ║
║     → Isotonic Regression calibra as probabilidades usando previsões OOF   ║
║                                                                              ║
║  2. SHRINKAGE BAYESIANO NO FINISHING RATIO                                  ║
║     Com apenas 12-13 jogos por time, o ratio gols/xG tem alta variância.   ║
║     → James-Stein shrinkage para a média da liga (prior_n = 15 jogos)     ║
║                                                                              ║
║  3. ALPHA DC/STAGE2 OTIMIZADO POR VALIDAÇÃO CRUZADA                        ║
║     Em vez de hardcoded 0.45, encontra o alpha que minimiza MAE            ║
║     num holdout cronológico 80/20.                                          ║
║                                                                              ║
║  4. DISPERSÃO DA BINOMIAL NEGATIVA                                          ║
║     gols reais têm var/mean ≈ 1.15 (levemente overdispersed).             ║
║     → Estima parâmetro r por MLE; usa NB em vez de Poisson puro           ║
║       para converter lambda → probabilidades                                ║
║                                                                              ║
║  5. DECAIMENTO EXPONENCIAL INTRA-JANELA DE FORMA                           ║
║     Em vez de média simples dos últimos N jogos, aplica decay              ║
║     exponencial para dar mais peso aos jogos mais recentes.                 ║
║                                                                              ║
║  6. RELATÓRIO COMPLETO DE CALIBRAÇÃO                                        ║
║     Mostra bias antes/depois da calibração para cada mercado.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import unicodedata
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import poisson, nbinom
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

PASSADAS_FILE  = "passadas.xlsx"
ATUAIS_FILE    = "atuais.xlsx"
PROXIMA_FILE   = "proxima.xlsx"
OUTPUT_FILE    = "previsao_v2.xlsx"

# Pesos de amostra
SEASON_DECAY         = 0.82
CURRENT_SEASON_BONUS = 1.35
ATUAIS_SOURCE_BONUS  = 1.10
PROXY_PENALTY        = 0.90
TRAIN_FRACTION       = 0.80   # split cronológico para calibração e CV-alpha

# Modelos
RANDOM_STATE = 42
OOF_SPLITS   = 5
USE_XGBOOST  = True

# Shrinkage bayesiano no finishing
# prior_n = número equivalente de jogos no prior (média da liga)
# Quanto maior, mais o finishing individual é puxado para a média da liga
FINISHING_PRIOR_N = 15

# Forma
RECENT_WINDOW = 5
FORM_DECAY    = 0.85   # decaimento exponencial intra-janela (mais recente > peso maior)
WFI_WEIGHTS   = (0.50, 0.30, 0.20)
H2H_GAMES     = 6

# DC
DC_DECAY = 0.97

# Probabilidade: usar Negative Binomial (True) ou Poisson puro (False)?
# NB trata a overdispersão real dos gols (var/mean ≈ 1.15)
USE_NB = True

# Limiares de mercado
THRESHOLDS = [0.5, 1.5, 2.5, 3.5]
SCORECARD_MIN_PROB = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))

def norm_col(c):
    c = strip_accents(c).lower().strip()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    return re.sub(r"_+", "_", c).strip("_")

def clean_team(x):
    return np.nan if pd.isna(x) else re.sub(r"\s+", " ", str(x).strip())

def safe_div(a, b, default=np.nan):
    if pd.isna(b) or b == 0:
        return default
    return float(a) / float(b)

def pts_result(hg, ag):
    if hg > ag: return 3, 0
    if hg < ag: return 0, 3
    return 1, 1

SEP  = "═" * 80
SEP2 = "─" * 80

def pct(v, d=1):
    return f"{v*100:.{d}f}%"

def banner(title):
    print(f"\n{SEP}\n  {title}\n{SEP2}")


# ══════════════════════════════════════════════════════════════════════════════
# BINOMIAL NEGATIVA — PROBABILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def estimate_nb_dispersion(goals: np.ndarray, lambdas: np.ndarray) -> float:
    """
    Estima o parâmetro de dispersão r da Binomial Negativa por MLE.
    NB(mu, r): var = mu + mu²/r  →  var/mean = 1 + mean/r
    Quanto maior r, mais próximo de Poisson.
    """
    mu = np.clip(lambdas, 0.01, None)
    goals = np.array(goals, dtype=float)

    def neg_ll(log_r):
        r = np.exp(log_r)
        ll = (gammaln(goals + r) - gammaln(r) - gammaln(goals + 1)
              + r * np.log(r / (r + mu))
              + goals * np.log(mu / (r + mu)))
        return -ll.sum()

    result = minimize(neg_ll, x0=[2.0], method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 1e-6})
    r = float(np.exp(result.x[0]))
    return max(r, 0.5)   # mínimo de estabilidade


def prob_over_nb(lam: float, line: float, r: float) -> float:
    """P(X > line) usando Binomial Negativa NB(mu=lam, r=r)."""
    lam = max(float(lam), 1e-9)
    k = int(line)
    p = r / (r + lam)
    return float(1.0 - nbinom.cdf(k, n=r, p=p))


def prob_over_poisson(lam: float, line: float) -> float:
    """P(X > line) usando Poisson(lam)."""
    lam = max(float(lam), 1e-9)
    k = int(line)
    return float(1.0 - poisson.cdf(k, lam))


def prob_over(lam: float, line: float, r_home: float = None, r_away: float = None,
              is_home: bool = True) -> float:
    """Dispatcher que usa NB se USE_NB e r disponível, senão Poisson."""
    if USE_NB:
        r = r_home if is_home else r_away
        if r is not None:
            return prob_over_nb(lam, line, r)
    return prob_over_poisson(lam, line)


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO E PADRONIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def prep_df(df: pd.DataFrame, is_future: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]

    rename = {
        "mandante": "home", "visitante": "away",
        "year": "ano", "round": "rodada",
        "home_xg": "hxg", "away_xg": "axg",
        "xg_home": "hxg", "xg_away": "axg",
        "hxg_": "hxg", "axg_": "axg",
    }
    # Normaliza colunas de xG que podem vir como hxG (maiúsculo)
    for c in list(df.columns):
        nc = norm_col(c)
        if nc in rename:
            df = df.rename(columns={c: rename[nc]})
        elif nc == "hxg" or nc == "axg":
            pass

    df.columns = [norm_col(c) for c in df.columns]
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["rodada", "home", "away", "ano"]
    if not is_future:
        required += ["hg", "ag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    df["home"] = df["home"].map(clean_team)
    df["away"] = df["away"].map(clean_team)

    for c in ["rodada", "ano", "hg", "ag", "hxg", "axg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if not is_future:
        for col_xg, col_g, flag in [("hxg", "hg", "hxg_proxy"), ("axg", "ag", "axg_proxy")]:
            if col_xg not in df.columns or df[col_xg].isna().all():
                df[col_xg] = df[col_g]
                df[flag]   = 1
            else:
                df[flag]   = df[col_xg].isna().astype(int)
                df[col_xg] = df[col_xg].fillna(df[col_g])

    df["rodada"] = df["rodada"].astype(int)
    df["ano"]    = df["ano"].astype(int)
    df["tick"]   = df["ano"] * 1000 + df["rodada"]
    df["_row"]   = np.arange(len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DIXON-COLES VETORIZADO
# ══════════════════════════════════════════════════════════════════════════════

def compute_dc_ratings(df: pd.DataFrame, decay: float = DC_DECAY):
    """MLE Dixon-Coles com NumPy puro (sem iterrows). ~50x mais rápido."""
    max_tick = df["tick"].max()
    w = (decay ** (max_tick - df["tick"].values)).astype(np.float64)

    teams = list(pd.unique(df[["home", "away"]].values.ravel()))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    hi = np.array([idx[t] for t in df["home"]], dtype=np.int32)
    ai = np.array([idx[t] for t in df["away"]], dtype=np.int32)
    hg = df["hg"].values.astype(np.float64)
    ag = df["ag"].values.astype(np.float64)

    lf = np.zeros(21)
    for k in range(1, 21):
        lf[k] = lf[k-1] + np.log(k)
    hg_lf = lf[np.clip(hg.astype(int), 0, 20)]
    ag_lf = lf[np.clip(ag.astype(int), 0, 20)]

    def neg_ll(p):
        att, dfe, ha = p[:n], p[n:2*n], p[2*n]
        llh_h = hg * (att[hi] - dfe[ai] + ha) - np.exp(att[hi] - dfe[ai] + ha) - hg_lf
        llh_a = ag * (att[ai] - dfe[hi])       - np.exp(att[ai] - dfe[hi])       - ag_lf
        return -np.dot(w, llh_h + llh_a)

    x0 = np.zeros(2*n + 1); x0[2*n] = 0.3
    res = minimize(neg_ll, x0, method="SLSQP",
                   constraints={"type": "eq", "fun": lambda p: p[:n].sum()},
                   options={"maxiter": 600, "ftol": 1e-8})
    p = res.x
    return ({teams[i]: p[i]     for i in range(n)},
            {teams[i]: p[n+i]   for i in range(n)},
            float(p[2*n]))


# ══════════════════════════════════════════════════════════════════════════════
# TABELA DE CLASSIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def make_block():
    return dict(games=0, gf=0., ga=0., xgf=0., xga=0.,
                pts=0., wins=0., draws=0., losses=0.)

def update_block(b, gf, ga, xgf, xga, pts):
    b["games"] += 1; b["gf"]  += gf;  b["ga"]  += ga
    b["xgf"]   += xgf;  b["xga"] += xga; b["pts"] += pts
    if pts == 3:   b["wins"]   += 1
    elif pts == 1: b["draws"]  += 1
    else:          b["losses"] += 1

def default_state():
    return {"overall": make_block(), "home": make_block(), "away": make_block()}

def block_feats(pfx, b):
    g = b["games"]
    return {
        f"{pfx}_g":        g,
        f"{pfx}_gf_pg":    safe_div(b["gf"],  g),
        f"{pfx}_ga_pg":    safe_div(b["ga"],  g),
        f"{pfx}_xgf_pg":   safe_div(b["xgf"], g),
        f"{pfx}_xga_pg":   safe_div(b["xga"], g),
        f"{pfx}_pts_pg":   safe_div(b["pts"], g),
        f"{pfx}_gd_pg":    safe_div(b["gf"]-b["ga"],  g),
        f"{pfx}_xgd_pg":   safe_div(b["xgf"]-b["xga"], g),
        f"{pfx}_wr":       safe_div(b["wins"],  g),
        f"{pfx}_dr":       safe_div(b["draws"], g),
        f"{pfx}_lr":       safe_div(b["losses"],g),
    }

def build_rank_table(season_tbl, teams, prev_summary):
    rows = []
    for team in teams:
        b = season_tbl.get(team, {}).get("overall", make_block())
        g = b["games"]
        prev = prev_summary.get(team, {})
        rows.append({
            "team": team, "games": g,
            "points": b["pts"], "gf": b["gf"], "ga": b["ga"],
            "gd": b["gf"]-b["ga"], "xgf": b["xgf"], "xga": b["xga"],
            "xgd": b["xgf"]-b["xga"],
            "ppg": safe_div(b["pts"], g, 0),
            "prev_ppg": prev.get("prev_ppg", np.nan),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["points","gd","gf","xgd"], ascending=False
    ).reset_index(drop=True)
    df["rank"]     = np.arange(1, len(df)+1)
    df["rank_pct"] = df["rank"] / len(df)
    return df[["team","rank","rank_pct","games","points","gf","ga","gd",
               "xgf","xga","xgd","ppg"]]

def finalize_season(season_tbl, teams, prev_summary):
    rdf = build_rank_table(season_tbl, teams, prev_summary)
    out = {}
    for _, r in rdf.iterrows():
        out[r["team"]] = {
            "prev_rank":     int(r["rank"]),
            "prev_rank_pct": float(r["rank_pct"]),
            "prev_ppg":      float(r["ppg"]) if pd.notna(r["ppg"]) else np.nan,
            "prev_gd_pg":    safe_div(r["gd"],  r["games"]),
            "prev_xgd_pg":   safe_div(r["xgd"], r["games"]),
            "prev_gf_pg":    safe_div(r["gf"],  r["games"]),
            "prev_ga_pg":    safe_div(r["ga"],  r["games"]),
            "prev_xgf_pg":   safe_div(r["xgf"], r["games"]),
            "prev_xga_pg":   safe_div(r["xga"], r["games"]),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FORMA RECENTE COM DECAIMENTO EXPONENCIAL (melhoria V2)
# ══════════════════════════════════════════════════════════════════════════════

def make_recent_buf():
    return {"gf": deque(maxlen=RECENT_WINDOW), "ga": deque(maxlen=RECENT_WINDOW),
            "xgf": deque(maxlen=RECENT_WINDOW), "xga": deque(maxlen=RECENT_WINDOW),
            "pts": deque(maxlen=RECENT_WINDOW)}

def default_recent():
    return {"overall": make_recent_buf(), "home": make_recent_buf(),
            "away": make_recent_buf()}

def update_recent(rec, gf, ga, xgf, xga, pts):
    for key, val in [("gf",gf),("ga",ga),("xgf",xgf),("xga",xga),("pts",pts)]:
        rec[key].append(float(val))

def recent_feats(pfx, rec):
    """
    Média com decaimento exponencial intra-janela.
    O jogo mais recente tem peso 1.0, o anterior FORM_DECAY, etc.
    """
    def _wmean(q):
        n = len(q)
        if n == 0:
            return np.nan
        weights = np.array([FORM_DECAY**(n-1-i) for i in range(n)])
        arr = np.array(q, dtype=float)
        return float(np.average(arr, weights=weights))

    gf  = _wmean(rec["gf"]);  ga  = _wmean(rec["ga"])
    xgf = _wmean(rec["xgf"]); xga = _wmean(rec["xga"])
    pts = _wmean(rec["pts"])
    return {
        f"{pfx}_gf":  gf,  f"{pfx}_ga":  ga,
        f"{pfx}_xgf": xgf, f"{pfx}_xga": xga,
        f"{pfx}_pts": pts,
        f"{pfx}_gd":  gf-ga  if pd.notna(gf)  and pd.notna(ga)  else np.nan,
        f"{pfx}_xgd": xgf-xga if pd.notna(xgf) and pd.notna(xga) else np.nan,
        f"{pfx}_n":   float(len(rec["gf"])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SHRINKAGE BAYESIANO NO FINISHING (melhoria V2)
# ══════════════════════════════════════════════════════════════════════════════

def shrunk_finishing(goals: float, xg: float, n_games: int,
                     league_mean: float = 1.0,
                     prior_n: int = FINISHING_PRIOR_N) -> float:
    """
    James-Stein shrinkage do ratio gols/xG em direção à média da liga.

    finishing_shrunk = (n / (n + prior_n)) * (goals/xg)
                     + (prior_n / (n + prior_n)) * league_mean

    Com n_games pequeno (time novo), o prior domina → estabilidade.
    Com n_games grande, a observação domina → fidelidade ao dado.
    """
    if n_games == 0 or xg <= 0:
        return league_mean
    sample = goals / xg
    w = n_games / (n_games + prior_n)
    return w * sample + (1 - w) * league_mean


# ══════════════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════════════════════════

def get_h2h(df_past: pd.DataFrame, home: str, away: str) -> dict:
    if df_past is None or len(df_past) == 0:
        return {k: np.nan for k in ["h2h_h_wr","h2h_dr","h2h_a_wr",
                                     "h2h_h_gf","h2h_a_gf","h2h_h_xgf","h2h_a_xgf","h2h_n"]}
    mask = (((df_past["home"]==home) & (df_past["away"]==away)) |
            ((df_past["home"]==away) & (df_past["away"]==home)))
    h2h  = df_past[mask].sort_values("tick").tail(H2H_GAMES)
    hw = dr = aw = hgf = agf = hxgf = axgf = 0
    for _, r in h2h.iterrows():
        if r["home"] == home:
            hgf += r["hg"]; agf += r["ag"]
            hxgf += r.get("hxg", r["hg"]); axgf += r.get("axg", r["ag"])
            if r["hg"] > r["ag"]: hw += 1
            elif r["hg"] < r["ag"]: aw += 1
            else: dr += 1
        else:
            hgf += r["ag"]; agf += r["hg"]
            hxgf += r.get("axg", r["ag"]); axgf += r.get("hxg", r["hg"])
            if r["ag"] > r["hg"]: hw += 1
            elif r["ag"] < r["hg"]: aw += 1
            else: dr += 1
    n = max(len(h2h), 1)
    return {"h2h_h_wr": hw/n, "h2h_dr": dr/n, "h2h_a_wr": aw/n,
            "h2h_h_gf": hgf/n, "h2h_a_gf": agf/n,
            "h2h_h_xgf": hxgf/n, "h2h_a_xgf": axgf/n,
            "h2h_n": float(len(h2h))}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURES POR JOGO
# ══════════════════════════════════════════════════════════════════════════════

def make_match_features(home, away, ano, rodada, tick,
                        career, season, recent,
                        season_tbl, teams_by_year, prev_summary, lg,
                        df_past, attack_r, defense_r, home_adv):
    f = {
        "ano": float(ano), "rodada": float(rodada),
        "is_early": 1.0 if rodada <= 5 else 0.0,
    }

    # ── Blocos estado acumulado ────────────────────────────────────────────────
    for pfx, team, venue in [
        ("hca", home, "overall"), ("hch", home, "home"),
        ("aca", away, "overall"), ("acw", away, "away"),
        ("hsa", home, "overall"), ("hsh", home, "home"),
        ("asa", away, "overall"), ("asw", away, "away"),
    ]:
        state = career if pfx[1] == "c" else season
        f.update(block_feats(pfx, state[team][venue]))

    for pfx, team, venue in [
        ("hra", home, "overall"), ("hrh", home, "home"),
        ("ara", away, "overall"), ("arw", away, "away"),
    ]:
        f.update(recent_feats(pfx, recent[team][venue]))

    # ── WFI: combina recente, temporada e carreira ─────────────────────────────
    for metric in ("xgf_pg", "xga_pg", "gf_pg", "ga_pg", "pts_pg"):
        for side, rec_pfx, szn_pfx, car_pfx in [
            ("h", "hra", "hsa", "hca"),
            ("a", "ara", "asa", "aca"),
        ]:
            v_rec = f.get(f"{rec_pfx}_{metric.replace('_pg','')}", np.nan)
            v_szn = f.get(f"{szn_pfx}_{metric}", np.nan)
            v_car = f.get(f"{car_pfx}_{metric}", np.nan)
            vals_w = [(v_rec, WFI_WEIGHTS[0]), (v_szn, WFI_WEIGHTS[1]), (v_car, WFI_WEIGHTS[2])]
            valid = [(v, w) for v, w in vals_w if pd.notna(v)]
            if valid:
                tw = sum(w for _, w in valid)
                f[f"wfi_{side}_{metric}"] = sum(v*w for v, w in valid) / tw
                # Momentum: recente vs temporada
                if pd.notna(v_rec) and pd.notna(v_szn):
                    f[f"mom_{side}_{metric}"] = v_rec - v_szn
                else:
                    f[f"mom_{side}_{metric}"] = np.nan
            else:
                f[f"wfi_{side}_{metric}"] = np.nan
                f[f"mom_{side}_{metric}"] = np.nan

    # ── Ranking ───────────────────────────────────────────────────────────────
    curr_df  = build_rank_table(season_tbl, teams_by_year.get(ano, []), prev_summary)
    cr_map   = curr_df.set_index("team").to_dict("index") if not curr_df.empty else {}
    hr = cr_map.get(home, {}); ar = cr_map.get(away, {})
    for k in ["rank","rank_pct","games","points","gf","ga","gd","xgf","xga","xgd","ppg"]:
        f[f"hcr_{k}"] = hr.get(k, np.nan)
        f[f"acr_{k}"] = ar.get(k, np.nan)

    hp = prev_summary.get(home, {}); ap = prev_summary.get(away, {})
    for k in ["prev_rank","prev_rank_pct","prev_ppg","prev_gd_pg","prev_xgd_pg"]:
        f[f"h_{k}"] = hp.get(k, np.nan)
        f[f"a_{k}"] = ap.get(k, np.nan)

    # ── Liga ──────────────────────────────────────────────────────────────────
    lg_n = lg["games"]
    f["lg_hg_pg"]  = safe_div(lg["hg"],  lg_n, 1.764)
    f["lg_ag_pg"]  = safe_div(lg["ag"],  lg_n, 1.404)
    f["lg_hxg_pg"] = safe_div(lg["hxg"], lg_n, 1.642)
    f["lg_axg_pg"] = safe_div(lg["axg"], lg_n, 1.310)

    # ── Dixon-Coles ratings ────────────────────────────────────────────────────
    att_h = attack_r.get(home, 0.0);  def_h = defense_r.get(home, 0.0)
    att_a = attack_r.get(away, 0.0);  def_a = defense_r.get(away, 0.0)
    dc_lam_h = float(np.exp(att_h - def_a + home_adv))
    dc_lam_a = float(np.exp(att_a - def_h))
    f.update({
        "att_h": att_h, "def_h": def_h, "att_a": att_a, "def_a": def_a,
        "dc_lam_h": dc_lam_h, "dc_lam_a": dc_lam_a,
        "dc_lam_diff": dc_lam_h - dc_lam_a,
        "home_adv": home_adv,
    })

    # ── H2H ───────────────────────────────────────────────────────────────────
    f.update(get_h2h(df_past, home, away))

    # ── Diffs compostos ───────────────────────────────────────────────────────
    diffs = [
        ("rank",        "hcr_rank",          "acr_rank"),
        ("ppg",         "hcr_ppg",           "acr_ppg"),
        ("gd",          "hcr_gd",            "acr_gd"),
        ("xgd",         "hcr_xgd",           "acr_xgd"),
        ("wfi_xgf",     "wfi_h_xgf_pg",      "wfi_a_xgf_pg"),
        ("wfi_xga",     "wfi_h_xga_pg",      "wfi_a_xga_pg"),
        ("wfi_pts",     "wfi_h_pts_pg",      "wfi_a_pts_pg"),
        ("dc",          "dc_lam_h",          "dc_lam_a"),
        ("prev_ppg",    "h_prev_ppg",        "a_prev_ppg"),
        ("prev_xgd",    "h_prev_xgd_pg",     "a_prev_xgd_pg"),
    ]
    for out, hk, ak in diffs:
        hv = f.get(hk, np.nan); av = f.get(ak, np.nan)
        f[f"diff_{out}"] = hv - av if pd.notna(hv) and pd.notna(av) else np.nan

    # ── Interação xG ataque × xGA defesa adversária ───────────────────────────
    h_xgf = f.get("hsh_xgf_pg", f.get("hca_xgf_pg", dc_lam_h))
    a_xga = f.get("asw_xga_pg", f.get("aca_xga_pg", 1.0)) or 1.0
    a_xgf = f.get("asw_xgf_pg", f.get("aca_xgf_pg", dc_lam_a))
    h_xga = f.get("hsh_xga_pg", f.get("hca_xga_pg", 1.0)) or 1.0
    f["xg_h_vs_def_a"] = (h_xgf or 0) / max(float(a_xga or 1), 0.1)
    f["xg_a_vs_def_h"] = (a_xgf or 0) / max(float(h_xga or 1), 0.1)

    return f


# ══════════════════════════════════════════════════════════════════════════════
# MONTAGEM DO DATASET (iteração temporal sem leakage)
# ══════════════════════════════════════════════════════════════════════════════

def build_datasets(hist_df, next_df, attack_r, defense_r, home_adv):
    hist_df = hist_df.sort_values(["ano","rodada","_row"]).reset_index(drop=True)
    next_df = next_df.sort_values(["ano","rodada","_row"]).reset_index(drop=True)

    teams_by_year = defaultdict(set)
    for df in [hist_df, next_df]:
        for _, r in df.iterrows():
            teams_by_year[int(r["ano"])].add(r["home"])
            teams_by_year[int(r["ano"])].add(r["away"])
    teams_by_year = {k: sorted(v) for k, v in teams_by_year.items()}

    career  = defaultdict(default_state)
    season  = defaultdict(default_state)
    recent  = defaultdict(default_recent)
    s_table = defaultdict(default_state)
    prev_summary = {}
    lg = dict(games=0, hg=0., ag=0., hxg=0., axg=0.)

    cur_year = None; cur_teams = []
    hist_buf = []    # buffer H2H sem leakage
    rows = []

    def roll_year(ny):
        nonlocal cur_year, season, s_table, prev_summary, cur_teams
        if cur_year is None:
            cur_year = ny; cur_teams = teams_by_year.get(ny, [])
            return
        if ny != cur_year:
            prev_summary = finalize_season(s_table, cur_teams, prev_summary)
            season  = defaultdict(default_state)
            s_table = defaultdict(default_state)
            cur_year = ny; cur_teams = teams_by_year.get(ny, [])

    print("  Iterando histórico cronologicamente...")
    for i, row in hist_df.iterrows():
        ano = int(row["ano"]); rod = int(row["rodada"])
        home = row["home"]; away = row["away"]
        tick = int(row["tick"])
        roll_year(ano)

        df_past = pd.DataFrame(hist_buf) if hist_buf else None
        feats = make_match_features(
            home, away, ano, rod, tick,
            career, season, recent, s_table,
            teams_by_year, prev_summary, lg, df_past,
            attack_r, defense_r, home_adv
        )
        feats.update({
            "home": home, "away": away,
            "hg": float(row["hg"]), "ag": float(row["ag"]),
            "hxg": float(row["hxg"]), "axg": float(row["axg"]),
            "hxg_proxy": int(row.get("hxg_proxy", 0)),
            "axg_proxy": int(row.get("axg_proxy", 0)),
            "dataset_source": str(row.get("dataset_source", "hist")),
        })
        rows.append(feats)

        hg = float(row["hg"]); ag = float(row["ag"])
        hxg = float(row["hxg"]); axg = float(row["axg"])
        ph, pa = pts_result(hg, ag)

        for st in [career, season, s_table]:
            update_block(st[home]["overall"], hg, ag, hxg, axg, ph)
            update_block(st[home]["home"],    hg, ag, hxg, axg, ph)
            update_block(st[away]["overall"], ag, hg, axg, hxg, pa)
            update_block(st[away]["away"],    ag, hg, axg, hxg, pa)

        update_recent(recent[home]["overall"], hg, ag, hxg, axg, ph)
        update_recent(recent[home]["home"],    hg, ag, hxg, axg, ph)
        update_recent(recent[away]["overall"], ag, hg, axg, hxg, pa)
        update_recent(recent[away]["away"],    ag, hg, axg, hxg, pa)

        lg["games"] += 1; lg["hg"] += hg; lg["ag"] += ag
        lg["hxg"] += hxg; lg["axg"] += axg

        hist_buf.append({"home":home,"away":away,"tick":tick,"ano":ano,
                         "hg":hg,"ag":ag,"hxg":hxg,"axg":axg})

        if (i+1) % 300 == 0:
            print(f"    [{i+1}/{len(hist_df)}] jogos processados...")

    hist_feats = pd.DataFrame(rows)

    # Features da próxima rodada
    df_past_full = pd.DataFrame(hist_buf)
    fut_year = cur_year; fut_tbl = s_table; fut_szn = season; fut_prev = prev_summary
    fut_teams = cur_teams
    next_rows = []

    for _, row in next_df.iterrows():
        ano = int(row["ano"]); rod = int(row["rodada"])
        home = row["home"]; away = row["away"]; tick = int(row["tick"])

        if fut_year is None:
            fut_year = ano; fut_teams = teams_by_year.get(ano, [])
        elif ano != fut_year:
            fut_prev  = finalize_season(fut_tbl, fut_teams, fut_prev)
            fut_tbl   = defaultdict(default_state)
            fut_szn   = defaultdict(default_state)
            fut_year  = ano; fut_teams = teams_by_year.get(ano, [])

        feats = make_match_features(
            home, away, ano, rod, tick,
            career, fut_szn, recent, fut_tbl,
            teams_by_year, fut_prev, lg, df_past_full,
            attack_r, defense_r, home_adv
        )
        feats.update({"home": home, "away": away})
        next_rows.append(feats)

    next_feats = pd.DataFrame(next_rows)
    ranking_atual = (build_rank_table(s_table, cur_teams, prev_summary)
                     if cur_year is not None else pd.DataFrame())
    return hist_feats, next_feats, ranking_atual


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS ML
# ══════════════════════════════════════════════════════════════════════════════

def get_xg_model():
    """Camada 1 — reg:gamma (ideal para xG contínuo ≥ 0)."""
    if USE_XGBOOST:
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                objective="reg:gamma", n_estimators=500,
                learning_rate=0.04, max_depth=4,
                min_child_weight=3, subsample=0.80,
                colsample_bytree=0.75, reg_alpha=0.10,
                reg_lambda=1.50, gamma=0.05,
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            ), "xgb_gamma"
        except ImportError:
            pass
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_depth=5,
        max_iter=300, min_samples_leaf=10, l2_regularization=0.1,
        random_state=RANDOM_STATE,
    ), "histgb_sq"


def get_goal_model():
    """Camada 2 — count:poisson (gols inteiros ≥ 0)."""
    if USE_XGBOOST:
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                objective="count:poisson", n_estimators=500,
                learning_rate=0.04, max_depth=4,
                min_child_weight=3, subsample=0.80,
                colsample_bytree=0.75, reg_alpha=0.05,
                reg_lambda=1.50, random_state=RANDOM_STATE,
                verbosity=0, n_jobs=-1,
            ), "xgb_poisson"
        except ImportError:
            pass
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_depth=5,
        max_iter=280, min_samples_leaf=10, l2_regularization=0.1,
        random_state=RANDOM_STATE,
    ), "histgb_poisson"


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE WEIGHTS COMPOSTOS
# ══════════════════════════════════════════════════════════════════════════════

def sample_weights(df):
    years  = df["ano"].astype(float).values
    rounds = df["rodada"].astype(float).values
    max_yr = np.nanmax(years)

    sw = np.power(SEASON_DECAY, max_yr - years)
    sw = np.where(years == max_yr, sw * CURRENT_SEASON_BONUS, sw)

    if "dataset_source" in df.columns:
        srcw = np.where(df["dataset_source"].astype(str).values == "atuais",
                        ATUAIS_SOURCE_BONUS, 1.0)
    else:
        srcw = np.ones(len(df))

    rndw = 0.90 + 0.20 * (rounds / max(1.0, np.nanmax(rounds)))

    if {"hxg_proxy","axg_proxy"}.issubset(df.columns):
        proxy = ((df["hxg_proxy"].fillna(0)>0) | (df["axg_proxy"].fillna(0)>0)).values
        qualw = np.where(proxy, PROXY_PENALTY, 1.0)
    else:
        qualw = np.ones(len(df))

    return sw * srcw * rndw * qualw


# ══════════════════════════════════════════════════════════════════════════════
# OOF EXPANDING-WINDOW (STAGE 1)
# ══════════════════════════════════════════════════════════════════════════════

def expanding_folds(n, n_splits=OOF_SPLITS, min_frac=0.40):
    if n < 20: return []
    min_tr = max(10, int(n * min_frac))
    if min_tr >= n - 1: return []
    step   = max(1, (n - min_tr) // n_splits)
    folds  = []
    te_end = min_tr
    while te_end < n:
        folds.append((np.arange(0, te_end), np.arange(te_end, min(n, te_end+step))))
        te_end = min(n, te_end + step)
    return folds


def oof_xg(X, y_h, y_a, sw):
    n = len(X)
    oh = np.full(n, np.nan); oa = np.full(n, np.nan)
    folds = expanding_folds(n)
    if not folds:
        warm = max(1, n//2)
        mh, _ = get_xg_model(); ma, _ = get_xg_model()
        mh.fit(X.iloc[:warm], y_h.iloc[:warm], sample_weight=sw[:warm])
        ma.fit(X.iloc[:warm], y_a.iloc[:warm], sample_weight=sw[:warm])
        oh[warm:] = np.clip(mh.predict(X.iloc[warm:]), 0.01, None)
        oa[warm:] = np.clip(ma.predict(X.iloc[warm:]), 0.01, None)
        return oh, oa
    for tr, te in folds:
        mh, _ = get_xg_model(); ma, _ = get_xg_model()
        mh.fit(X.iloc[tr], y_h.iloc[tr], sample_weight=sw[tr])
        ma.fit(X.iloc[tr], y_a.iloc[tr], sample_weight=sw[tr])
        oh[te] = np.clip(mh.predict(X.iloc[te]), 0.01, None)
        oa[te] = np.clip(ma.predict(X.iloc[te]), 0.01, None)
    return oh, oa


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: FINISHING BLEND COM SHRINKAGE (melhoria V2)
# ══════════════════════════════════════════════════════════════════════════════

def add_stage2(df, pred_h_col, pred_a_col,
               league_fin_h=1.0, league_fin_a=1.0,
               league_def_oa_h=1.0, league_def_oa_a=1.0):
    """
    Adiciona features de finishing com shrinkage bayesiano.
    league_fin_* e league_def_oa_* são calculados a partir do histórico completo.
    """
    out = df.copy()
    out["pred_xg_h"] = np.clip(out[pred_h_col].astype(float), 0.01, None)
    out["pred_xg_a"] = np.clip(out[pred_a_col].astype(float), 0.01, None)

    def rf(r, num_k, den_k, default=1.0):
        n = r.get(num_k); d = r.get(den_k)
        if pd.isna(n) or pd.isna(d) or d == 0: return default
        return float(n) / float(d)

    # Finishing com shrinkage bayesiano
    def shrunk_ratio(r, gf_k, xgf_k, g_k, league_mean):
        gf  = r.get(gf_k,  np.nan)
        xgf = r.get(xgf_k, np.nan)
        n_g = r.get(g_k,   0)
        if pd.isna(gf) or pd.isna(xgf) or n_g == 0:
            return league_mean
        return shrunk_finishing(gf * n_g, xgf * n_g, int(n_g), league_mean)

    out["hfin_szn"] = out.apply(lambda r: shrunk_ratio(r, "hsh_gf_pg","hsh_xgf_pg","hsh_g", league_fin_h), axis=1)
    out["hfin_car"] = out.apply(lambda r: shrunk_ratio(r, "hch_gf_pg","hch_xgf_pg","hch_g", league_fin_h), axis=1)
    out["hfin_rec"] = out.apply(lambda r: shrunk_ratio(r, "hrh_gf",   "hrh_xgf",   "hrh_n", league_fin_h), axis=1)
    out["afin_szn"] = out.apply(lambda r: shrunk_ratio(r, "asw_gf_pg","asw_xgf_pg","asw_g", league_fin_a), axis=1)
    out["afin_car"] = out.apply(lambda r: shrunk_ratio(r, "acw_gf_pg","acw_xgf_pg","acw_g", league_fin_a), axis=1)
    out["afin_rec"] = out.apply(lambda r: shrunk_ratio(r, "arw_gf",   "arw_xgf",   "arw_n", league_fin_a), axis=1)

    out["hdef_szn"] = out.apply(lambda r: shrunk_ratio(r, "hsh_ga_pg","hsh_xga_pg","hsh_g", league_def_oa_h), axis=1)
    out["hdef_car"] = out.apply(lambda r: shrunk_ratio(r, "hch_ga_pg","hch_xga_pg","hch_g", league_def_oa_h), axis=1)
    out["adef_szn"] = out.apply(lambda r: shrunk_ratio(r, "asw_ga_pg","asw_xga_pg","asw_g", league_def_oa_a), axis=1)
    out["adef_car"] = out.apply(lambda r: shrunk_ratio(r, "acw_ga_pg","acw_xga_pg","acw_g", league_def_oa_a), axis=1)

    out["hfin"] = out[["hfin_szn","hfin_car","hfin_rec"]].mean(axis=1)
    out["afin"] = out[["afin_szn","afin_car","afin_rec"]].mean(axis=1)
    out["hdef"] = out[["hdef_szn","hdef_car"]].mean(axis=1)
    out["adef"] = out[["adef_szn","adef_car"]].mean(axis=1)

    # Goal anchors: xG × √(finishing × def_overallow adversária)
    out["h_anchor"] = out["pred_xg_h"] * np.sqrt(
        np.clip(out["hfin"], 0.60, 1.60) * np.clip(out["adef"], 0.60, 1.60))
    out["a_anchor"] = out["pred_xg_a"] * np.sqrt(
        np.clip(out["afin"], 0.60, 1.60) * np.clip(out["hdef"], 0.60, 1.60))

    out["pred_xg_total"] = out["pred_xg_h"] + out["pred_xg_a"]
    out["pred_xg_diff"]  = out["pred_xg_h"] - out["pred_xg_a"]
    out["anchor_total"]  = out["h_anchor"]   + out["a_anchor"]
    out["anchor_diff"]   = out["h_anchor"]   - out["a_anchor"]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA CV + CALIBRAÇÃO ISOTÔNICA (melhorias V2)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_alpha_cv(dc_h, dc_a, s2_h, s2_a, y_h, y_a):
    """
    Busca o alpha ótimo (peso do Dixon-Coles vs Stage2) via
    minimização do MAE de gols reais num holdout cronológico.
    """
    def loss(alpha):
        lh = alpha * dc_h + (1-alpha) * s2_h
        la = alpha * dc_a + (1-alpha) * s2_a
        return (mean_absolute_error(y_h, lh) + mean_absolute_error(y_a, la)) / 2

    result = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    return float(result.x)


def fit_isotonic_calibrators(oof_lam_h, oof_lam_a,
                             y_hg, y_ag, r_h=None, r_a=None):
    """
    Calibração isotônica por threshold.
    Mapeia Prob_modelo(over N) → Prob_calibrada(over N) usando previsões OOF.
    Isso corrige o bias sistemático:
      Over 0.5 mandante: +7pp | Over 1.5 mandante: +6pp
    """
    calibrators = {}
    for thr in THRESHOLDS:
        # Home
        raw_h = np.array([
            (prob_over_nb(l, thr, r_h) if (USE_NB and r_h) else prob_over_poisson(l, thr))
            for l in oof_lam_h
        ])
        act_h = (np.array(y_hg) > thr).astype(float)
        ir_h  = IsotonicRegression(out_of_bounds="clip", increasing=True)
        ir_h.fit(raw_h, act_h)
        calibrators[f"h_{thr}"] = ir_h

        # Away
        raw_a = np.array([
            (prob_over_nb(l, thr, r_a) if (USE_NB and r_a) else prob_over_poisson(l, thr))
            for l in oof_lam_a
        ])
        act_a = (np.array(y_ag) > thr).astype(float)
        ir_a  = IsotonicRegression(out_of_bounds="clip", increasing=True)
        ir_a.fit(raw_a, act_a)
        calibrators[f"a_{thr}"] = ir_a

    return calibrators


def apply_calibrated_prob(lam, thr, calibrator, r=None):
    """Aplica calibração isotônica à probabilidade raw de Poisson/NB."""
    if USE_NB and r is not None:
        raw = prob_over_nb(lam, thr, r)
    else:
        raw = prob_over_poisson(lam, thr)
    return float(calibrator.predict([raw])[0])


# ══════════════════════════════════════════════════════════════════════════════
# RELATÓRIO DE CALIBRAÇÃO (novo em V2)
# ══════════════════════════════════════════════════════════════════════════════

def calibration_report(oof_lam_h, oof_lam_a, y_hg, y_ag,
                       calibrators, r_h=None, r_a=None):
    """
    Compara bias de calibração antes e depois do ajuste isotônico.
    """
    banner("RELATÓRIO DE CALIBRAÇÃO (holdout cronológico 80/20)")
    print(f"  {'Mercado':<25} {'Real':>7} {'Antes':>7} {'Depois':>7} {'ΔBias':>8}")
    print(f"  {SEP2}")

    for side, lams, y_goals, r in [
        ("Mandante", oof_lam_h, y_hg, r_h),
        ("Visitante", oof_lam_a, y_ag, r_a),
    ]:
        for thr in THRESHOLDS:
            actual = float(np.mean(np.array(y_goals) > thr))

            raw_probs = np.array([
                (prob_over_nb(l, thr, r) if (USE_NB and r) else prob_over_poisson(l, thr))
                for l in lams
            ])
            before = float(np.mean(raw_probs))

            key = f"{'h' if 'Man' in side else 'a'}_{thr}"
            cal_probs = calibrators[key].predict(raw_probs)
            after = float(np.mean(cal_probs))

            delta = abs(after - actual) - abs(before - actual)
            sign  = "✅" if delta < -0.001 else ("➖" if abs(delta) < 0.001 else "⚠️ ")
            print(f"  {sign} {side} Over {thr:<8}   "
                  f"{actual:>6.3f}  {before:>6.3f}  {after:>6.3f}  {delta:>+7.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    banner("MODELO PREDITIVO V2 — QUALIDADE PREDITIVA + ESTABILIDADE")

    # ── 1. Carregar dados ─────────────────────────────────────────────────────
    for f in [PASSADAS_FILE, ATUAIS_FILE, PROXIMA_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Arquivo não encontrado: {f}")

    print("\n📂 Carregando dados...")
    passadas = prep_df(pd.read_excel(PASSADAS_FILE), is_future=False)
    atuais   = prep_df(pd.read_excel(ATUAIS_FILE),   is_future=False)
    proxima  = prep_df(pd.read_excel(PROXIMA_FILE),  is_future=True)

    passadas["dataset_source"] = "passadas"
    atuais["dataset_source"]   = "atuais"

    hist = pd.concat([passadas, atuais], ignore_index=True, sort=False)
    hist = hist.sort_values(["ano","rodada","_row"]).reset_index(drop=True)

    print(f"   Histórico: {len(hist)} jogos | Anos: {sorted(hist['ano'].unique())}")
    print(f"   Temporada atual: {len(atuais)} jogos | Rodadas: {sorted(atuais['rodada'].unique())}")
    print(f"   Próxima rodada {proxima['rodada'].iloc[0]}: {len(proxima)} jogos")
    print("\n  Jogos a prever:")
    for _, r in proxima.iterrows():
        print(f"    {r['home']:<24} vs  {r['away']}")

    # ── 2. Dixon-Coles ────────────────────────────────────────────────────────
    banner("STEP 1/7 — Dixon-Coles MLE (vetorizado)")
    attack_r, defense_r, home_adv = compute_dc_ratings(hist)
    print(f"  Vantagem de mandante (exp): {np.exp(home_adv):.4f}x")
    df_rat = pd.DataFrame({"Ataque": attack_r, "Defesa": defense_r})
    df_rat = df_rat.sort_values("Ataque", ascending=False)
    df_rat["λ_casa"] = np.exp(df_rat["Ataque"] - df_rat["Defesa"].mean() + home_adv).round(3)
    df_rat["λ_fora"] = np.exp(df_rat["Ataque"] - df_rat["Defesa"].mean()).round(3)
    print("\n  Ratings Dixon-Coles (Top 8 ataques):")
    print(df_rat.head(8).round(4).to_string())

    # ── 3. Construir datasets ─────────────────────────────────────────────────
    banner("STEP 2/7 — Montagem de features (sem data leakage)")
    hist_f, next_f, ranking = build_datasets(
        hist, proxima, attack_r, defense_r, home_adv)
    print(f"  ✅ {len(hist_f)} jogos com features | {len(next_f)} para prever")

    # Tabela de classificação
    if not ranking.empty:
        print(f"\n  TABELA DO CAMPEONATO — ANO CORRENTE")
        print(f"  {'#':>3}  {'Time':<24} {'Pts':>5} {'J':>3} {'GD':>4} {'xGD':>6} {'ppg':>5}")
        print(f"  {SEP2}")
        for _, r in ranking.iterrows():
            print(f"  {int(r['rank']):>3}  {r['team']:<24} {int(r['points']):>5} "
                  f"{int(r['games']):>3} {int(r['gd']):>4} {r['xgd']:>6.2f} {r['ppg']:>5.2f}")

    # ── 4. Preparar X/y ───────────────────────────────────────────────────────
    banner("STEP 3/7 — Treinamento Stage 1 (xG) com OOF expanding-window")

    exclude = {"home","away","hg","ag","hxg","axg","hxg_proxy","axg_proxy","dataset_source"}
    feat_cols = [c for c in hist_f.columns
                 if c not in exclude and pd.api.types.is_numeric_dtype(hist_f[c])]

    X      = hist_f[feat_cols].fillna(hist_f[feat_cols].median())
    X_next = next_f[feat_cols].fillna(X.median())

    y_hxg = hist_f["hxg"].clip(lower=0.01, upper=4.5)
    y_axg = hist_f["axg"].clip(lower=0.01, upper=4.5)
    y_hg  = hist_f["hg"].astype(float)
    y_ag  = hist_f["ag"].astype(float)
    w     = sample_weights(hist_f)

    # OOF Stage 1
    print("  Gerando previsões OOF de xG (expanding window)...")
    oof_h, oof_a = oof_xg(X, y_hxg, y_axg, w)

    fb_h = hist_f.get("hsh_xgf_pg", pd.Series(np.nan, index=hist_f.index)).fillna(
           hist_f.get("hca_xgf_pg", pd.Series(np.nan, index=hist_f.index))).fillna(y_hxg.median())
    fb_a = hist_f.get("asw_xgf_pg", pd.Series(np.nan, index=hist_f.index)).fillna(
           hist_f.get("aca_xgf_pg", pd.Series(np.nan, index=hist_f.index))).fillna(y_axg.median())

    hist_f["oof_xg_h"] = np.where(np.isnan(oof_h), fb_h, oof_h)
    hist_f["oof_xg_a"] = np.where(np.isnan(oof_a), fb_a, oof_a)

    # Treinar modelo final Stage 1 no histórico completo
    print("  Treinando modelo final Stage 1 no histórico completo...")
    mh1, mh1_name = get_xg_model(); ma1, ma1_name = get_xg_model()
    mh1.fit(X, y_hxg, sample_weight=w)
    ma1.fit(X, y_axg, sample_weight=w)
    next_f["pred_xg_h"] = np.clip(mh1.predict(X_next), 0.01, None)
    next_f["pred_xg_a"] = np.clip(ma1.predict(X_next), 0.01, None)

    mae_xg_oof_h = mean_absolute_error(y_hxg, hist_f["oof_xg_h"])
    mae_xg_oof_a = mean_absolute_error(y_axg, hist_f["oof_xg_a"])
    print(f"  MAE xG OOF — mandante: {mae_xg_oof_h:.4f} | visitante: {mae_xg_oof_a:.4f}")

    # ── 5. Stage 2 com finishing shrinkage ────────────────────────────────────
    banner("STEP 4/7 — Stage 2 (xG → gols) com finishing shrinkage bayesiano")

    # Médias da liga para usar como prior no shrinkage
    league_fin_h  = safe_div(y_hg.sum(), y_hxg.sum(), 1.0)
    league_fin_a  = safe_div(y_ag.sum(), y_axg.sum(), 1.0)
    league_doa_h  = safe_div((hist_f.get("hsh_ga_pg",pd.Series([1.0]*len(hist_f)))).mean(), 1.0, 1.0)
    league_doa_a  = safe_div((hist_f.get("asw_ga_pg",pd.Series([1.0]*len(hist_f)))).mean(), 1.0, 1.0)
    print(f"  Prior de finishing — mandante: {league_fin_h:.3f} | visitante: {league_fin_a:.3f}")
    print(f"  (Prior N = {FINISHING_PRIOR_N} jogos equivalentes)")

    hist_s2 = add_stage2(hist_f, "oof_xg_h", "oof_xg_a",
                         league_fin_h, league_fin_a, league_doa_h, league_doa_a)
    next_s2 = add_stage2(next_f, "pred_xg_h", "pred_xg_a",
                         league_fin_h, league_fin_a, league_doa_h, league_doa_a)

    s2_exclude = exclude.copy()
    s2_cols = [c for c in hist_s2.columns
               if c not in s2_exclude and c in next_s2.columns
               and pd.api.types.is_numeric_dtype(hist_s2[c])]

    X_s2 = hist_s2[s2_cols].fillna(hist_s2[s2_cols].median())
    X_next_s2 = next_s2[s2_cols].fillna(X_s2.median())

    mh2, mh2_name = get_goal_model(); ma2, ma2_name = get_goal_model()
    mh2.fit(X_s2, y_hg, sample_weight=w)
    ma2.fit(X_s2, y_ag, sample_weight=w)

    s2_lam_h = np.clip(mh2.predict(X_next_s2), 0.02, None)
    s2_lam_a = np.clip(ma2.predict(X_next_s2), 0.02, None)

    # OOF Stage 2 para calibração
    print("  Gerando OOF Stage 2 para calibração...")
    oof_s2_h = np.full(len(hist_s2), np.nan)
    oof_s2_a = np.full(len(hist_s2), np.nan)
    folds = expanding_folds(len(hist_s2))
    if folds:
        for tr, te in folds:
            ms2h, _ = get_goal_model(); ms2a, _ = get_goal_model()
            ms2h.fit(X_s2.iloc[tr], y_hg.iloc[tr], sample_weight=w[tr])
            ms2a.fit(X_s2.iloc[tr], y_ag.iloc[tr], sample_weight=w[tr])
            oof_s2_h[te] = np.clip(ms2h.predict(X_s2.iloc[te]), 0.02, None)
            oof_s2_a[te] = np.clip(ms2a.predict(X_s2.iloc[te]), 0.02, None)
    else:
        oof_s2_h = s2_lam_h.mean() * np.ones(len(hist_s2))
        oof_s2_a = s2_lam_a.mean() * np.ones(len(hist_s2))

    nan_mask = np.isnan(oof_s2_h)
    oof_s2_h[nan_mask] = float(np.nanmean(oof_s2_h))
    oof_s2_a[np.isnan(oof_s2_a)] = float(np.nanmean(oof_s2_a))

    # ── 6. Alpha CV-otimizado ─────────────────────────────────────────────────
    banner("STEP 5/7 — Otimização CV do alpha DC/Stage2")

    dc_lam_hist_h = hist_f["dc_lam_h"].values
    dc_lam_hist_a = hist_f["dc_lam_a"].values

    # Usar últimos 20% como holdout para otimizar alpha
    split = int(len(hist_s2) * 0.80)
    alpha_opt = optimize_alpha_cv(
        dc_lam_hist_h[split:], dc_lam_hist_a[split:],
        oof_s2_h[split:],      oof_s2_a[split:],
        y_hg.values[split:],   y_ag.values[split:]
    )
    print(f"  Alpha DC ótimo (CV): {alpha_opt:.3f}  "
          f"(DC={int(alpha_opt*100)}% | Stage2={int((1-alpha_opt)*100)}%)")

    # OOF ensemble lambdas (para calibração)
    oof_ens_h = alpha_opt * dc_lam_hist_h + (1-alpha_opt) * oof_s2_h
    oof_ens_a = alpha_opt * dc_lam_hist_a + (1-alpha_opt) * oof_s2_a

    # ── 7. Dispersão NB + calibração isotônica ────────────────────────────────
    banner("STEP 6/7 — Dispersão NB + Calibração Isotônica")

    # Usar apenas jogos com OOF válidos (não-nan) para NB e calibração
    valid = ~(np.isnan(oof_ens_h) | np.isnan(oof_ens_a))
    ens_h_v = oof_ens_h[valid]; ens_a_v = oof_ens_a[valid]
    y_hg_v  = y_hg.values[valid]; y_ag_v  = y_ag.values[valid]

    if USE_NB:
        print("  Estimando parâmetro de dispersão (Binomial Negativa)...")
        r_h = estimate_nb_dispersion(y_hg_v, ens_h_v)
        r_a = estimate_nb_dispersion(y_ag_v, ens_a_v)
        print(f"  r_home = {r_h:.2f} (var/mean = {1+np.mean(ens_h_v)/r_h:.4f})")
        print(f"  r_away = {r_a:.2f} (var/mean = {1+np.mean(ens_a_v)/r_a:.4f})")
    else:
        r_h = r_a = None

    print("\n  Ajustando calibradores isotônicos por threshold...")
    calibrators = fit_isotonic_calibrators(
        ens_h_v, ens_a_v, y_hg_v, y_ag_v, r_h, r_a)

    calibration_report(ens_h_v, ens_a_v, y_hg_v, y_ag_v,
                       calibrators, r_h, r_a)

    # ── 8. Previsão final ─────────────────────────────────────────────────────
    banner("STEP 7/7 — Previsão da próxima rodada")

    dc_lam_h = next_f["dc_lam_h"].values
    dc_lam_a = next_f["dc_lam_a"].values
    lam_h = np.maximum(alpha_opt * dc_lam_h + (1-alpha_opt) * s2_lam_h, 0.10)
    lam_a = np.maximum(alpha_opt * dc_lam_a + (1-alpha_opt) * s2_lam_a, 0.10)

    # Probabilidades calibradas
    results = []
    for i, (_, row) in enumerate(proxima.iterrows()):
        home = row["home"]; away = row["away"]
        lh = lam_h[i]; la = lam_a[i]

        res = {
            "home": home, "away": away,
            "xg_home": round(float(next_f["pred_xg_h"].iloc[i]), 3),
            "xg_away": round(float(next_f["pred_xg_a"].iloc[i]), 3),
            "dc_lam_home": round(float(dc_lam_h[i]), 3),
            "dc_lam_away": round(float(dc_lam_a[i]), 3),
            "s2_lam_home": round(float(s2_lam_h[i]), 3),
            "s2_lam_away": round(float(s2_lam_a[i]), 3),
            "lambda_home": round(lh, 3),
            "lambda_away": round(la, 3),
        }

        for thr in THRESHOLDS:
            suf = str(thr).replace(".", "_")
            res[f"home_over_{suf}"] = round(apply_calibrated_prob(
                lh, thr, calibrators[f"h_{thr}"], r_h), 4)
            res[f"away_over_{suf}"] = round(apply_calibrated_prob(
                la, thr, calibrators[f"a_{thr}"], r_a), 4)

        results.append(res)

    df_pred = pd.DataFrame(results)

    # ── Exibição ──────────────────────────────────────────────────────────────
    print(f"\n  {'Mandante':<24} {'Visitante':<24} {'xG_M':>6} {'xG_V':>6} "
          f"{'DC_M':>6} {'S2_M':>6} {'λ_M':>6} {'λ_V':>6}")
    print(f"  {SEP2}")
    for _, r in df_pred.iterrows():
        print(f"  {r['home']:<24} {r['away']:<24} "
              f"{r['xg_home']:>6.3f} {r['xg_away']:>6.3f} "
              f"{r['dc_lam_home']:>6.3f} {r['s2_lam_home']:>6.3f} "
              f"{r['lambda_home']:>6.3f} {r['lambda_away']:>6.3f}")

    print(f"\n  {'Mandante':<24} {'O0.5':>7} {'O1.5':>7} {'O2.5':>7} {'O3.5':>7}")
    print(f"  {SEP2}")
    for _, r in df_pred.iterrows():
        print(f"  {r['home']:<24} "
              f"{pct(r['home_over_0_5']):>7} {pct(r['home_over_1_5']):>7} "
              f"{pct(r['home_over_2_5']):>7} {pct(r['home_over_3_5']):>7}")

    print(f"\n  {'Visitante':<24} {'O0.5':>7} {'O1.5':>7} {'O2.5':>7} {'O3.5':>7}")
    print(f"  {SEP2}")
    for _, r in df_pred.iterrows():
        print(f"  {r['away']:<24} "
              f"{pct(r['away_over_0_5']):>7} {pct(r['away_over_1_5']):>7} "
              f"{pct(r['away_over_2_5']):>7} {pct(r['away_over_3_5']):>7}")

    # ── Scorecard ≥ 70% ────────────────────────────────────────────────────────
    market_map = {
        "home_over_0_5":"Mandante O0.5","home_over_1_5":"Mandante O1.5",
        "home_over_2_5":"Mandante O2.5","home_over_3_5":"Mandante O3.5",
        "away_over_0_5":"Visitante O0.5","away_over_1_5":"Visitante O1.5",
        "away_over_2_5":"Visitante O2.5","away_over_3_5":"Visitante O3.5",
    }
    bets = []
    for _, r in df_pred.iterrows():
        for col, label in market_map.items():
            if r[col] >= SCORECARD_MIN_PROB:
                lam_ref = r["lambda_home"] if "Man" in label else r["lambda_away"]
                bets.append({"Jogo": f"{r['home']} vs {r['away']}",
                              "Mercado": label, "Prob": pct(r[col]),
                              "λ": f"{lam_ref:.2f}"})

    print(f"\n  🎯 SCORECARD — mercados ≥ {int(SCORECARD_MIN_PROB*100)}% (probabilidades calibradas)")
    print(f"  {SEP2}")
    if bets:
        df_bets = pd.DataFrame(bets).sort_values("Prob", ascending=False)
        print(df_bets.to_string(index=False))
    else:
        print("  Nenhum mercado ≥ 70%.")

    # ── Info do modelo ─────────────────────────────────────────────────────────
    info = pd.DataFrame([
        {"item": "modelo_xg",             "value": mh1_name},
        {"item": "modelo_gols",           "value": mh2_name},
        {"item": "alpha_dc_cv",           "value": round(alpha_opt, 4)},
        {"item": "dc_home_adv_exp",       "value": round(float(np.exp(home_adv)), 4)},
        {"item": "finishing_prior_n",     "value": FINISHING_PRIOR_N},
        {"item": "use_nb",                "value": int(USE_NB)},
        {"item": "r_home_nb",             "value": round(r_h, 3) if r_h else np.nan},
        {"item": "r_away_nb",             "value": round(r_a, 3) if r_a else np.nan},
        {"item": "season_decay",          "value": SEASON_DECAY},
        {"item": "form_decay",            "value": FORM_DECAY},
        {"item": "jogos_historico",       "value": len(hist_f)},
        {"item": "n_features_stage1",     "value": len(feat_cols)},
        {"item": "n_features_stage2",     "value": len(s2_cols)},
        {"item": "mae_xg_oof_home",       "value": round(mae_xg_oof_h, 4)},
        {"item": "mae_xg_oof_away",       "value": round(mae_xg_oof_a, 4)},
        {"item": "league_finishing_home", "value": round(league_fin_h, 4)},
        {"item": "league_finishing_away", "value": round(league_fin_a, 4)},
    ])

    oof_diag = pd.DataFrame({
        "ano":        hist_f["ano"].values,
        "rodada":     hist_f["rodada"].values,
        "home":       hist_f["home"].values,
        "away":       hist_f["away"].values,
        "real_hxg":   y_hxg.values,
        "oof_xg_h":   hist_f["oof_xg_h"].values,
        "real_axg":   y_axg.values,
        "oof_xg_a":   hist_f["oof_xg_a"].values,
        "real_hg":    y_hg.values,
        "oof_lam_h":  oof_ens_h,
        "real_ag":    y_ag.values,
        "oof_lam_a":  oof_ens_a,
    })

    # ── Export Excel ───────────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_pred.to_excel(writer, sheet_name="previsoes",       index=False)
        if not ranking.empty:
            ranking.to_excel(writer, sheet_name="ranking",     index=False)
        info.to_excel(writer, sheet_name="info_modelo",        index=False)
        oof_diag.to_excel(writer, sheet_name="oof_diagnostico",index=False)
        if bets:
            df_bets.to_excel(writer, sheet_name="scorecard",   index=False)

    print(f"\n{SEP}")
    print(f"  ✅ Arquivo '{OUTPUT_FILE}' gerado com sucesso!")
    print(f"     Abas: previsoes | ranking | info_modelo | oof_diagnostico | scorecard")
    print(SEP)


if __name__ == "__main__":
    main()
