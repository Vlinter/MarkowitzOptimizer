# utils.py
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ---- Filtres de warnings (optionnel, mais propre) ----
def setup_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Values in x were outside bounds during a minimize step, clipping to bounds",
        category=RuntimeWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="DataFrame.applymap has been deprecated. Use DataFrame.map instead.",
        category=FutureWarning
    )
    # Ignorer le FutureWarning de yfinance
    warnings.filterwarnings(
        "ignore",
        message="YF.download() has changed argument auto_adjust default to True",
        category=FutureWarning
    )


# ---- helper rerun (compat old/new streamlit) ----
def _safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ---- Helpers Fréquence et Périodes ----
def infer_frequency(index: pd.DatetimeIndex) -> str:
    idx = index.sort_values().unique()
    if len(idx) < 3: return "monthly"
    deltas = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    med = np.median(deltas)
    if med <= 2:    return "daily"
    if med <= 10:   return "weekly"
    if med <= 40:   return "monthly"
    if med <= 120:  return "quarterly"
    return "yearly"

def ann_factor(freq: str) -> int:
    return {"daily":252,"weekly":52,"monthly":12,"quarterly":4,"yearly":1}[freq.lower()]

def months_to_periods(months: int, freq: str) -> int:
    f = freq.lower()
    if f == "monthly":   return max(1, int(round(months)))
    if f == "quarterly": return max(1, int(round(months/3)))
    if f == "yearly":    return max(1, int(round(months/12)))
    if f == "weekly":    return max(1, int(round(months * 52/12)))
    if f == "daily":     return max(1, int(round(months * 252/12)))
    return max(1, months)

def n_for_rebalance_choice(choice: str, data_freq: str, n_custom: int) -> int:
    if choice == "Auto (fréquence des données)": return 1
    if choice == "Mensuel":   return months_to_periods(1, data_freq)
    if choice == "Trimestriel": return months_to_periods(3, data_freq)
    if choice == "Annuel":      return months_to_periods(12, data_freq)
    if choice == "Tous les N périodes": return max(1, int(n_custom))
    return 1

# ---- Helpers Traitement de Données ----
def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")

def build_availability_from_union(prices_union: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in prices_union.columns:
        s = prices_union[col]
        first = s.first_valid_index(); last  = s.last_valid_index()
        rows.append({"Ticker": str(col).upper(), "First": pd.to_datetime(first) if first is not None else pd.NaT,
                                                 "Last":  pd.to_datetime(last)  if last  is not None else pd.NaT})
    df = pd.DataFrame(rows)
    return df.sort_values("First") if not df.empty else df

# ---- Helpers Maths (Covariance, Corrélation) ----
def near_psd_clip(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w, V = np.linalg.eigh(cov); w_clipped = np.clip(w, eps, None)
    return (V @ np.diag(w_clipped) @ V.T)

def ridge_regularize(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    n = cov.shape[0]; tau = np.trace(cov) / max(1, n)
    return cov + ridge * float(tau) * np.eye(n)

def cluster_order_from_corr(corr: pd.DataFrame) -> list:
    dist = (1 - corr).clip(lower=0) / 2.0
    linked = linkage(squareform(dist.values, checks=False), method="average")
    order = leaves_list(linked)
    return list(corr.index[order])

def top_corr_pairs(corr: pd.DataFrame, k: int = 3):
    M = corr.copy(); np.fill_diagonal(M.values, np.nan)
    pairs = []; cols = M.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            c = M.iloc[i, j]
            if pd.notna(c): pairs.append((cols[i], cols[j], float(c)))
    if not pairs: return [], []
    pairs_sorted_pos = sorted(pairs, key=lambda x: x[2], reverse=True)
    pairs_sorted_neg = sorted(pairs, key=lambda x: x[2])
    return pairs_sorted_pos[:k], pairs_sorted_neg[:k]

# ---- Helpers Validation (Poids) ----
def validate_weight_bounds(min_w: float, max_w: float, n: int, tol: float = 1e-12):
    problems = []; hints = []
    if min_w < 0: problems.append("Le poids minimum est négatif."); hints.append("Fixe MIN_W ≥ 0.")
    if max_w <= 0: problems.append("Le poids maximum est nul ou négatif."); hints.append("Fixe MAX_W > 0.")
    if min_w > max_w: problems.append("MIN_W est supérieur à MAX_W."); hints.append("Assure-toi que MIN_W ≤ MAX_W.")
    if min_w * n > 1 + tol:
        problems.append(f"MIN_W×n = {min_w:.2%}×{n} = {min_w*n:.2%} > 100%.")
        hints.append(f"Baisse MIN_W ≤ {1/n:.2%} ou réduis le nombre d’actifs contraints.")
    if max_w * n < 1 - tol:
        problems.append(f"MAX_W×n = {max_w:.2%}×{n} = {max_w*n:.2%} < 100%.")
        hints.append(f"Augmente MAX_W ≥ {1/n:.2%} ou retire des actifs.")
    return problems, hints

def warn_tight_bounds(min_w: float, max_w: float, n: int, tol: float = 1e-12):
    slack_min = 1 - min_w * n; slack_max = max_w * n - 1
    msgs = []
    if slack_min >= 0 and slack_min < 0.02 + tol: msgs.append(f"MIN_W×n proche de 100% (marge {slack_min:.2%}).")
    if slack_max >= 0 and slack_max < 0.02 + tol: msgs.append(f"MAX_W×n proche de 100% (marge {slack_max:.2%}).")
    if msgs: st.warning("Bornes très serrées : " + " ".join(msgs))