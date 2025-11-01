# =========================================================
# MARKOWITZ PORTFOLIO OPTIMIZER — v7.6 (Partie 1/2)
#
# MODIFIÉ (v7.6) :
# - La fonction 'run_backtest' retourne maintenant un
#   'target_weights_log' (historique des allocations cibles)
#   pour l'affichage dans l'UI.
# =========================================================
import warnings
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


import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Optionnel: Ledoit–Wolf (scikit-learn)
try:
    from sklearn.covariance import LedoitWolf
    HAS_LW = True
except Exception:
    HAS_LW = False

# Yahoo Finance
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ---- CSS Personnalisé pour arrondir les coins ----
st.markdown("""
<style>
/* Arrondir les coins des conteneurs bordés */
[data-testid="stVerticalBlock"] > [style*="border: 1px solid"] {
    border-radius: 10px;
}
/* Arrondir les coins des expanders */
[data-testid="stExpander"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
# ---- Fin du CSS ----


# ---- helper rerun (compat old/new streamlit) ----
def _safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ---- Plotly helper ----
def st_plotly_chart(fig: go.Figure) -> None:
    fig.update_layout(autosize=True, margin=dict(l=40, r=20, t=40, b=40))
    plotly_config = {"displaylogo": False, "scrollZoom": True, "responsive": True}
    st.plotly_chart(fig, config=plotly_config, theme=None, use_container_width=True)

# ============================ Utils (Maths) ============================
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

def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")

def near_psd_clip(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w, V = np.linalg.eigh(cov); w_clipped = np.clip(w, eps, None)
    return (V @ np.diag(w_clipped) @ V.T)

def ridge_regularize(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    n = cov.shape[0]; tau = np.trace(cov) / max(1, n)
    return cov + ridge * float(tau) * np.eye(n)

def portfolio_perf(w, mu, cov):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol

def feasible_start(n: int, min_w: float, max_w: float) -> np.ndarray:
    if min_w < 0 or max_w <= 0 or min_w > max_w: raise ValueError("Bornes incohérentes.")
    w = np.full(n, min_w, dtype=float); R = 1.0 - n * min_w
    if R < -1e-12: raise ValueError("Bornes infaisables : min_w * n > 1.")
    if R <= 1e-12: return w
    cap = np.maximum(0.0, max_w - w)
    while R > 1e-12:
        idx = np.where(cap > 1e-12)[0]
        if idx.size == 0: break
        inc = R / idx.size
        add = np.minimum(cap[idx], inc)
        w[idx] += add; cap[idx] -= add; R -= float(add.sum())
    np.clip(w, min_w, max_w, out=w)
    s = w.sum()
    if abs(s - 1.0) > 1e-10:
        w /= s; np.clip(w, min_w, max_w, out=w)
    return w

def _solve_slsqp(objective, n, bounds, cons, w0=None, maxiter=2000, ftol=1e-10):
    if w0 is None: w0 = feasible_start(n, bounds[0][0], bounds[0][1])
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})
    if res.success: return res.x
    w0b = np.ones(n)/n
    res2 = minimize(objective, w0b, method='SLSQP', bounds=bounds, constraints=cons,
                    options={'maxiter': maxiter*2, 'ftol': ftol*10, 'disp': False})
    if res2.success: return res2.x
    try:
        w_fallback = feasible_start(n, bounds[0][0], bounds[0][1])
        return w_fallback
    except ValueError:
        return np.ones(n) / n

def max_sharpe_weights(mu, cov, min_w, max_w):
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = feasible_start(n, min_w, max_w)
    def obj(w):
        ret, vol = portfolio_perf(w, mu, cov)
        return 1e6 if vol <= 1e-12 else - (ret) / vol
    return _solve_slsqp(obj, n, bounds, cons, w0=w0)

def min_var_weights(mu, cov, min_w, max_w):
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = feasible_start(n, min_w, max_w)
    return _solve_slsqp(lambda w: float(w @ cov @ w), n, bounds, cons, w0=w0)

def max_return_weights(mu, cov, min_w, max_w):
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = feasible_start(n, min_w, max_w)
    return _solve_slsqp(lambda w: - float(w @ mu), n, bounds, cons, w0=w0)

def risk_parity_weights(cov, min_w, max_w):
    n = cov.shape[0]
    bounds = [(min_w, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = feasible_start(n, min_w, max_w)
    def obj_rp(w, cov):
        port_var = float(w @ cov @ w)
        if port_var <= 1e-12:
            return 0.0
        rc_pct = (w * (cov @ w)) / port_var
        return np.std(rc_pct)
    return _solve_slsqp(lambda w: obj_rp(w, cov), n, bounds, cons, w0=w0)

def _min_var_for_target_return(mu, cov, r_target, min_w, max_w, w_start=None):
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w, rt=r_target: float(w @ mu) - rt}
    ]
    if w_start is None:
        w_start = feasible_start(n, min_w, max_w)
        w_start = np.clip(w_start + 0.01 * (mu / (np.linalg.norm(mu)+1e-12)), min_w, max_w)
        w_start = w_start / w_start.sum()
    def obj(w): return float(w @ cov @ w)
    return _solve_slsqp(obj, n, bounds, cons, w0=w_start, maxiter=3000, ftol=1e-12)

def efficient_frontier(mu, cov, min_w, max_w, npts=160, eps_keep=1e-10):
    try:
        w_mv = min_var_weights(mu, cov, min_w, max_w); r_mv, _ = portfolio_perf(w_mv, mu, cov)
        w_mr = max_return_weights(mu, cov, min_w, max_w); r_mr, _ = portfolio_perf(w_mr, mu, cov)
    except Exception:
        return np.empty((0, 2))
    r_grid = np.linspace(r_mv, r_mr, npts)
    pts = []; w_prev = None
    for rt in r_grid:
        try:
            w_t = _min_var_for_target_return(mu, cov, rt, min_w, max_w, w_start=w_prev)
            r_t, v_t = portfolio_perf(w_t, mu, cov)
            pts.append((r_t, v_t)); w_prev = w_t
        except Exception:
            continue
    if not pts: return np.empty((0, 2))
    pts = np.array(pts); order = np.argsort(pts[:, 1]); pts = pts[order]
    cleaned = [pts[0]]
    for r, v in pts[1:]:
        if r >= cleaned[-1][0] - eps_keep:
            cleaned.append([r, v])
    return np.array(cleaned)

def risk_contrib(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    tot_var = float(w @ cov @ w)
    if tot_var <= 0: return np.zeros_like(w)
    mrc = cov @ w
    return (w * mrc) / tot_var

def cluster_order_from_corr(corr: pd.DataFrame) -> list:
    dist = (1 - corr).clip(lower=0) / 2.0
    linked = linkage(squareform(dist.values, checks=False), method="average")
    order = leaves_list(linked)
    return list(corr.index[order])

def sample_bounded_simplex(n: int, N: int, min_w: float, max_w: float, rng: np.random.Generator) -> np.ndarray:
    base = np.full(n, min_w, dtype=float)
    cap = np.full(n, max_w - min_w, dtype=float)
    Rtot = 1.0 - n * min_w
    if Rtot < -1e-12: raise ValueError("Bornes infaisables : min_w * n > 1.")
    if Rtot <= 1e-12: return np.tile(base, (N, 1))
    alpha = np.where(cap > 0, cap, 1e-12)
    W = np.empty((N, n), dtype=float)
    for k in range(N):
        p = rng.dirichlet(alpha)
        add = np.minimum(p * Rtot, cap)
        R = Rtot - float(add.sum())
        free = cap - add
        while R > 1e-12 and (free > 1e-12).any():
            q = rng.dirichlet(np.where(free > 1e-12, free, 1e-12))
            inc = np.minimum(free, q * R)
            add += inc; R -= float(inc.sum()); free = cap - add
        W[k, :] = base + add
    W = np.clip(W, min_w, max_w); s = W.sum(axis=1, keepdims=True); W = W / s
    return W

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

# === MOTEUR DE BACKTEST (v7.6) ===
@st.cache_data(show_spinner="Exécution du backtest...")
def run_backtest(
    returns: pd.DataFrame, 
    _strategy_func: callable, # Ignoré par le cache
    w_static: np.ndarray, 
    rebalance_freq_n: int, 
    lookback_window_n: int, 
    min_w: float, 
    max_w: float, 
    fee_ter_ann_pct: float, 
    fee_txn_pct: float, 
    k: int
) -> tuple[pd.Series, pd.DataFrame, list]: # <--- MODIFIÉ v7.6
    """
    Moteur de backtest unifié.
    - Gère les allocations Statiques (w_static) ou Dynamiques (_strategy_func).
    - Gère Buy & Hold (rebalance_freq_n=None) ou Rebalancement (rebalance_freq_n=N).
    - Gère les frais (TER et Transaction).
    - RETOURNE : wealth, weights_history (dérive journalière), target_weights_log
    """
    n = returns.shape[1]
    
    # --- Initialisation ---
    if w_static is not None:
        w_target = w_static / w_static.sum()
    else:
        # Fallback pour le premier jour si dynamique
        w_target = feasible_start(n, min_w, max_w) 
    
    w_cur = w_target.copy()
    values = np.empty(len(returns), dtype=float)
    weights_history = []
    target_weights_log = [] # <--- AJOUT v7.6
    port_val = 1.0
    
    # Log des poids initiaux <--- AJOUT v7.6
    target_weights_log.append({
        "Date": returns.index[0],
        "Weights": w_target
    })

    # --- Frais ---
    fee_ter_periodic = (fee_ter_ann_pct / 100.0) / k
    fee_txn = fee_txn_pct / 100.0

    # --- Boucle principale ---
    for t in range(len(returns)):
        
        is_rebalance_day = (rebalance_freq_n is not None) and (t % rebalance_freq_n == 0) and (t > 0)
        
        if is_rebalance_day:
            
            # --- 1a. Déterminer les poids cibles (w_target) ---
            if _strategy_func is not None:
                # Mode Dynamique : On recalcule les poids
                start_idx = max(0, t - lookback_window_n)
                end_idx = t 
                
                if (end_idx - start_idx) < (n * 2): 
                    pass 
                else:
                    returns_slice = returns.iloc[start_idx:end_idx]
                    mu_slice = returns_slice.mean().values * k
                    cov_raw_slice = returns_slice.cov().values * k
                    cov_slice = near_psd_clip(ridge_regularize(cov_raw_slice, ridge=1e-5))
                    try:
                        w_target = _strategy_func(mu_slice, cov_slice, min_w, max_w)
                    except Exception as e:
                        pass 
            
            else:
                # Mode Statique : w_target ne change jamais
                pass 

            # Log des nouveaux poids cibles <--- AJOUT v7.6
            target_weights_log.append({
                "Date": returns.index[t],
                "Weights": w_target
            })

            # --- 1b. Appliquer les Frais de Transaction ---
            turnover = np.sum(np.abs(w_target - w_cur)) / 2.0 
            txn_cost = turnover * fee_txn
            port_val *= (1.0 - txn_cost) 
            
            # --- 1c. Mettre à jour les poids ---
            w_cur = w_target.copy()

        # --- 2. Enregistrer les poids du jour (AVANT calcul du rendement) ---
        weights_history.append(w_cur.copy())
        
        # --- 3. Calculer la performance de la période (Fin de période) ---
        r_vec = returns.iloc[t].values.astype(float)
        rp_gross = float(w_cur @ r_vec)
        rp_net = rp_gross - fee_ter_periodic
        port_val *= (1 + rp_net)
        values[t] = port_val
        
        # --- 4. Mettre à jour w_cur pour le drift (pour la prochaine itération) ---
        if t < len(returns) - 1:
            gross_ret_components = w_cur * (1 + r_vec)
            denom = gross_ret_components.sum()
            
            if denom > 0 and np.isfinite(denom):
                w_cur = gross_ret_components / denom
            else:
                w_cur = w_target.copy() if (rebalance_freq_n is not None) else w_cur

    # --- 5. Finalisation ---
    wealth = pd.Series(values, index=returns.index)
    wealth.iloc[0] = 1.0 
    weights_df = pd.DataFrame(weights_history, index=returns.index, columns=returns.columns)
    
    return wealth, weights_df, target_weights_log # <--- MODIFIÉ v7.6

# === FIN DU MOTEUR DE BACKTEST ===


def perf_metrics(wealth: pd.Series, freq_k: int):
    ret = wealth.pct_change().dropna()
    total_return = wealth.iloc[-1] - 1.0
    years = len(ret) / freq_k if freq_k > 0 else np.nan
    cagr = wealth.iloc[-1] ** (1/years) - 1 if years and years > 0 else np.nan
    vol_ann = ret.std(ddof=1) * np.sqrt(freq_k)
    sharpe = (ret.mean() * freq_k) / vol_ann if vol_ann > 0 else np.nan
    negative_returns = ret[ret < 0].dropna()
    downside_std = negative_returns.std(ddof=1) * np.sqrt(freq_k)
    sortino = (ret.mean() * freq_k) / downside_std if downside_std > 0 else np.nan
    cummax = wealth.cummax(); dd = wealth / cummax - 1.0
    max_dd = dd.min(); worst = ret.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {"Total Return": total_return, "CAGR": cagr, "Vol ann.": vol_ann,
            "Sharpe": sharpe, "Max Drawdown": max_dd, "Worst period": worst,
            "Sortino": sortino, "Calmar": calmar}, dd

def calculate_relative_metrics(port_returns: pd.Series, bench_returns: pd.Series, freq_k: int) -> dict:
    df = pd.DataFrame({'port': port_returns, 'bench': bench_returns}).dropna()
    if len(df) < 2:
        return {"Beta": np.nan, "Alpha (ann.)": np.nan, "Tracking Error": np.nan, "Information Ratio": np.nan}
    cov_matrix = df.cov() * freq_k
    covariance = cov_matrix.iloc[0, 1]
    bench_var = df['bench'].var() * freq_k
    beta = covariance / bench_var if bench_var > 0 else np.nan
    years = len(df) / freq_k
    port_cagr = (1 + df['port']).prod()**(1/years) - 1 if years > 0 else np.nan
    bench_cagr = (1 + df['bench']).prod()**(1/years) - 1 if years > 0 else np.nan
    alpha = port_cagr - (beta * bench_cagr) if pd.notna(port_cagr) and pd.notna(bench_cagr) and pd.notna(beta) else np.nan
    diff_ret = df['port'] - df['bench']
    tracking_error = diff_ret.std(ddof=1) * np.sqrt(freq_k)
    information_ratio = (port_cagr - bench_cagr) / tracking_error if tracking_error > 0 else np.nan
    return {"Beta": beta, "Alpha (ann.)": alpha, "Tracking Error": tracking_error, "Information Ratio": information_ratio}

def calculate_market_regime_metrics(port_returns_monthly: pd.Series, bench_returns_monthly: pd.Series) -> dict:
    """Calcule Batting Average, Up-Capture et Down-Capture Ratios (basé sur les rendements mensuels)."""
    df = pd.DataFrame({'port': port_returns_monthly, 'bench': bench_returns_monthly}).dropna()
    if df.empty or len(df) < 2:
        return {"Batting Average": np.nan, "Up-Capture": np.nan, "Down-Capture": np.nan}
    batting_avg = (df['port'] > df['bench']).mean()
    bench_up_months_ret = df[df['bench'] > 0]
    bench_down_months_ret = df[df['bench'] < 0]
    port_up_mean = bench_up_months_ret['port'].mean()
    bench_up_mean = bench_up_months_ret['bench'].mean()
    port_down_mean = bench_down_months_ret['port'].mean()
    bench_down_mean = bench_down_months_ret['bench'].mean()
    up_capture = (port_up_mean / bench_up_mean) if bench_up_mean > 0 else np.nan
    down_capture = (port_down_mean / bench_down_mean) if bench_down_mean < 0 else np.nan 
    return {
        "Batting Average": batting_avg,
        "Up-Capture": up_capture * 100.0, 
        "Down-Capture": down_capture * 100.0
    }

def calculate_var_cvar(returns: pd.Series, percentile: float = 0.95) -> dict:
    """Calcule la Value-at-Risk (VaR) et la Conditional VaR (CVaR) historiques."""
    if returns.empty:
        return {"VaR": np.nan, "CVaR": np.nan}
    var = returns.quantile(1.0 - percentile)
    cvar = returns[returns <= var].mean()
    return {"VaR": var, "CVaR": cvar}

def create_monthly_heatmap_df(monthly_returns: pd.Series) -> pd.DataFrame:
    """Prépare le DataFrame pour la heatmap des rendements mensuels."""
    if monthly_returns.empty:
        return pd.DataFrame()
    df = pd.DataFrame(monthly_returns)
    df.columns = ['Return']
    df['Year'] = df.index.year
    df['Month'] = df.index.strftime('%b')
    heatmap_df = df.pivot(index='Year', columns='Month', values='Return')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    available_cols = [m for m in month_order if m in heatmap_df.columns]
    return heatmap_df[available_cols]

def find_drawdowns(wealth: pd.Series, top_n: int = 5) -> pd.DataFrame:
    drawdowns = []
    peak_date = wealth.index[0]
    peak_value = wealth.iloc[0]
    in_drawdown = False
    trough_date = peak_date
    trough_value = peak_value
    for t, v in wealth.iloc[1:].items():
        if v > peak_value:
            if in_drawdown:
                recovery_date = t
                max_dd = (trough_value / peak_value) - 1.0
                p_to_t_days = (trough_date - peak_date).days
                total_duration_days = (recovery_date - peak_date).days
                drawdowns.append({
                    "Peak Date": peak_date.date(),
                    "Trough Date": trough_date.date(),
                    "Recovery Date": recovery_date.date(),
                    "Max Drawdown": max_dd,
                    "Peak-to-Trough (Days)": p_to_t_days,
                    "Total Duration (Days)": total_duration_days
                })
            peak_date = t
            peak_value = v
            trough_date = t
            trough_value = v
            in_drawdown = False
        elif v < trough_value:
            in_drawdown = True
            trough_date = t
            trough_value = v
    if in_drawdown and trough_value < peak_value:
        max_dd = (trough_value / peak_value) - 1.0
        p_to_t_days = (trough_date - peak_date).days
        drawdowns.append({
            "Peak Date": peak_date.date(),
            "Trough Date": trough_date.date(),
            "Recovery Date": pd.NaT,
            "Max Drawdown": max_dd,
            "Peak-to-Trough (Days)": p_to_t_days,
            "Total Duration (Days)": np.nan
        })
    if not drawdowns:
        return pd.DataFrame(columns=[
            "Peak Date", "Trough Date", "Recovery Date",
            "Max Drawdown", "Peak-to-Trough (Days)", "Total Duration (Days)"
        ])
    df = pd.DataFrame(drawdowns)
    df = df.sort_values(by="Max Drawdown", ascending=True)
    return df.head(top_n)

def n_for_rebalance_choice(choice: str, data_freq: str, n_custom: int) -> int:
    if choice == "Auto (fréquence des données)": return 1
    if choice == "Mensuel":   return months_to_periods(1, data_freq)
    if choice == "Trimestriel": return months_to_periods(3, data_freq)
    if choice == "Annuel":      return months_to_periods(12, data_freq)
    if choice == "Tous les N périodes": return max(1, int(n_custom))
    return 1

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

def build_availability_from_union(prices_union: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in prices_union.columns:
        s = prices_union[col]
        first = s.first_valid_index(); last  = s.last_valid_index()
        rows.append({"Ticker": str(col).upper(), "First": pd.to_datetime(first) if first is not None else pd.NaT,
                                                 "Last":  pd.to_datetime(last)  if last  is not None else pd.NaT})
    df = pd.DataFrame(rows)
    return df.sort_values("First") if not df.empty else df

# ============================ Cache I/O ============================
@st.cache_data(show_spinner="Chargement du fichier Excel...")
def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index().dropna(axis=1, how="all")
    df.columns = [str(c).upper() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices_to_returns(prices)

@st.cache_data(show_spinner="Téléchargement des données Yahoo Finance...")
def fetch_yahoo_prices(
    tickers: list[str],
    interval: str = "1d",
    auto_adjust: bool = True,
):
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    tickers_up = [t.upper() for t in tickers]
    df = yf.download(
        tickers=tickers_up,
        period="max",
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    rows = []; union_cols = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers_up:
            if (t, "Close") in df.columns:
                s_union = df[(t, "Close")].rename(t)
                first = s_union.first_valid_index(); last  = s_union.last_valid_index()
                if first is not None and last is not None:
                    rows.append({"Ticker": t, "First": pd.to_datetime(first), "Last": pd.to_datetime(last)})
                    union_cols[t] = s_union
    else:
        if "Close" in df.columns and len(tickers_up) == 1:
            t = tickers_up[0]; s_union = df["Close"].rename(t)
            first = s_union.first_valid_index(); last  = s_union.last_valid_index()
            if first is not None and last is not None:
                rows.append({"Ticker": t, "First": pd.to_datetime(first), "Last": pd.to_datetime(last)})
                union_cols[t] = s_union
        elif "Close" in df.columns and len(tickers_up) > 1:
            for t in tickers_up:
                if t in df.columns: 
                    s_union = df[t].rename(t)
                    first = s_union.first_valid_index(); last = s_union.last_valid_index()
                    if first is not None and last is not None:
                        rows.append({"Ticker": t, "First": pd.to_datetime(first), "Last": pd.to_datetime(last)})
                        union_cols[t] = s_union
    availability_df = pd.DataFrame(rows).sort_values("First") if rows else pd.DataFrame(columns=["Ticker","First","Last"])
    if union_cols:
        prices_union = pd.concat(union_cols.values(), axis=1); prices_union.columns = list(union_cols.keys())
        prices_intersection = prices_union.dropna(how="any").sort_index()
    else:
        prices_intersection = pd.DataFrame()
    return prices_intersection, availability_df

@st.cache_data(show_spinner="Chargement du benchmark...")
def fetch_benchmark_prices(ticker: str, start_date, end_date, interval: str) -> pd.Series:
    """Charge les prix de clôture pour un seul ticker de benchmark."""
    if not ticker:
        return pd.Series(dtype=float)
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        st.error(f"Impossible de télécharger les données du benchmark {ticker}.")
        return pd.Series(dtype=float)
    s = df['Close']
    s.name = ticker 
    return s

# ============================ UI PRINCIPALE ============================
st.title("Optimisation de portefeuille (Markowitz)")
st.markdown("Analysez la frontière efficiente et backtestez des portefeuilles selon la théorie de Markowitz.")
st.divider()

with st.container(border=True):
    st.markdown("#### 1. Source des données")
    data_src = st.radio(
        "Choisir une source :",
        ["Fichier Excel", "Yahoo Finance"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
        key="data_src_radio",
        help="Chargez vos propres prix depuis Excel ou téléchargez-les depuis Yahoo Finance."
    )

    prices_all = None
    availability = None
    yf_interval = "1d"
    
    if data_src == "Fichier Excel":
        uploaded_file = st.file_uploader(
            "Fichier Excel (.xlsx/.xls)", 
            type=["xlsx", "xls"], 
            key="excel_uploader",
            help="Le fichier doit avoir une colonne de dates en premier, puis une colonne de prix par actif."
        )
        if uploaded_file is not None:
            prices_union = load_excel(uploaded_file)
            availability = build_availability_from_union(prices_union)
            prices_all = prices_union.dropna(how="any")
            if not prices_all.empty:
                yf_interval = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}.get(infer_frequency(prices_all.index), "1d")

    else:
        if not HAS_YF:
            st.error("yfinance n’est pas installé : `pip install yfinance`")
        col1, col2 = st.columns([2, 1])
        with col1:
            tick_input = st.text_input(
                "Tickers Yahoo (séparés par des virgules)",
                value="QQQ, VGK, VWO, GLD, SLV, BTC-USD",
                placeholder="Ex: QQQ, VGK, VWO, GLD, SLV, BTC-USD",
                key="yf_tickers_input",
                help="Entrez les symboles boursiers tels que reconnus par Yahoo Finance (ex: 'AAPL', 'BTC-USD', '^GSPC')."
            )
        with col2:
            yf_interval = st.selectbox(
                "Intervalle", 
                options=["1d", "1wk", "1mo"], 
                index=0, 
                key="yf_interval_select",
                help="Fréquence des données à télécharger (Journalier, Hebdomadaire, Mensuel)."
            )
            yf_auto_adjust = st.checkbox(
                "Ajuster dividendes/splits", 
                value=True, 
                key="yf_auto_adjust",
                help="Ajuste automatiquement les prix historiques pour les dividendes et les fractionnements d'actions (recommandé)."
            )
        yf_tickers = [t.strip().upper() for t in tick_input.split(",") if t.strip()]
        if st.button("📥 Charger depuis Yahoo (période max)", key="yf_download_btn"):
            if len(yf_tickers) < 2:
                st.error("Renseigne au moins deux tickers.")
            elif HAS_YF:
                prices_all, availability = fetch_yahoo_prices(
                    tickers=yf_tickers, interval=yf_interval, auto_adjust=yf_auto_adjust
                )
                if prices_all is None or prices_all.empty:
                    st.error("Aucune donnée renvoyée par Yahoo Finance (tickers/intervalle).")
                else:
                    st.session_state["yf_prices_all"] = prices_all
                    st.session_state["yf_availability"] = availability
        if "yf_prices_all" in st.session_state and st.session_state["yf_prices_all"] is not None:
            current_tickers = st.session_state["yf_prices_all"].columns.tolist()
            st.caption(f"Tickers actuellement en mémoire : {', '.join(current_tickers)}")
        if prices_all is None:
            prices_all = st.session_state.get("yf_prices_all", None)
            availability = st.session_state.get("yf_availability", None)

if prices_all is None or prices_all.empty:
    st.info("Charge des données (Excel ou Yahoo) ci-dessus pour continuer.")
    st.stop()

prices_all.columns = [str(c).upper() for c in prices_all.columns]
all_tickers = prices_all.columns.tolist()
if len(all_tickers) < 2:
    st.error("Il faut au moins deux actifs.")
    st.stop()

if "excluded" not in st.session_state:
    st.session_state["excluded"] = []

if availability is not None and not availability.empty:
    availability["Ticker"] = availability["Ticker"].astype(str).str.upper()
    start_def = pd.to_datetime(availability["First"]).max().date()
    end_def    = pd.to_datetime(availability["Last"]).min().date()
else:
    start_def = prices_all.index.min().date()
    end_def    = prices_all.index.max().date()

with st.container(border=True):
    st.markdown("#### 2. Paramètres de l'analyse")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Périmètre temporel et Fréquence", expanded=True):
            c1a, c1b = st.columns(2)
            with c1a:
                start_date = st.date_input(
                    "Date de début", 
                    value=start_def, 
                    min_value=start_def, 
                    max_value=end_def, 
                    key="start_date",
                    help="Première date à inclure dans l'analyse. Basée sur la date commune la plus récente de tous les actifs."
                )
            with c1b:
                end_date   = st.date_input(
                    "Date de fin", 
                    value=end_def,
                    min_value=start_def, 
                    max_value=end_def, 
                    key="end_date",
                    help="Dernière date à inclure dans l'analyse."
                )
            
            temp_index = prices_all.loc[(prices_all.index.date >= start_date) & (prices_all.index.date <= end_date)].index
            auto_freq = infer_frequency(temp_index) if not temp_index.empty else "daily"
            freq_options = ["daily", "weekly", "monthly", "quarterly", "yearly"]
            try:
                freq_index = freq_options.index(auto_freq)
            except ValueError:
                freq_index = 0
            freq = st.selectbox(
                "Fréquence des données (détection auto modifiable)",
                options=freq_options,
                index=freq_index,
                help="Fréquence d'échantillonnage des données (daily, weekly, etc.). Affecte le facteur d'annualisation (k).",
                key="freq_select"
            )
            k = ann_factor(freq)
            st.caption(f"Fréquence sélectionnée : {freq} | Annualisation : ×{k}")
    with c2:
        with st.expander("Paramètres d'optimisation & Monte Carlo", expanded=True):
            c2a, c2b = st.columns(2)
            with c2a:
                min_w = st.slider(
                    "Poids minimum par actif", 0.0, 1.0, 0.00, 0.01, 
                    key="min_w_slider",
                    help="Contrainte de poids minimum (ex: 0.05 pour 5%) pour chaque actif dans l'optimisation. Doit être 'long-only' (≥ 0)."
                )
                n_mc  = st.number_input(
                    "Portefeuilles Monte Carlo", 
                    value=1000, 
                    step=500, 
                    min_value=100, 
                    key="mc_input",
                    help="Nombre de portefeuilles aléatoires à générer pour visualiser l'univers des possibles (le 'nuage')."
                )
            with c2b:
                max_w = st.slider(
                    "Poids maximum par actif", 0.0, 1.0, 1.00, 0.01, 
                    key="max_w_slider",
                    help="Contrainte de poids maximum (ex: 0.25 pour 25%) pour chaque actif."
                )
                seed = 42
                use_lw = st.checkbox(
                    "Utiliser Ledoit–Wolf (si dispo)", 
                    value=True, 
                    disabled=not HAS_LW, 
                    key="use_lw_checkbox",
                    help="Utilise l'estimateur de covariance 'Ledoit-Wolf' (shrinkage) au lieu de la covariance standard. Plus robuste pour peu d'observations."
                )

if start_date > end_date:
    st.error("La date de début doit être antérieure ou égale à la date de fin.")
    st.stop()

mask = (prices_all.index.date >= start_date) & (prices_all.index.date <= end_date)
prices = prices_all.loc[mask].copy()
excluded = [t for t in st.session_state["excluded"] if t in prices.columns]
if excluded:
    prices = prices.drop(columns=excluded, errors="ignore")

tickers = [str(c).upper() for c in prices.columns]
prices.columns = tickers

if prices.shape[0] < 3:
    st.error("Trop peu d’observations après filtrage. Élargissez la période.")
    st.stop()
if len(tickers) < 2:
    st.error("Il faut au moins deux actifs après exclusions.")
    st.stop()

st.caption(f"Période utilisée : {prices.index.min().date()} → {prices.index.max().date()} | Observations : {len(prices)}")
if excluded:
    st.caption(f"Actifs exclus actuellement : {', '.join(excluded)}")

problems, hints = validate_weight_bounds(min_w, max_w, len(tickers))
if problems:
    st.error("**Bornes de poids infaisables.**\n\n" + "• " + "\n• ".join(problems) +
            ("\n\n**Comment corriger :**\n" + "• " + "\n• ".join(hints) if hints else ""))
    st.stop()
else:
    warn_tight_bounds(min_w, max_w, len(tickers))


# =========================================================
# MARKOWITZ PORTFOLIO OPTIMIZER — v7.9 (Partie 2/2)
#
# MODIFIÉ (v7.9) :
# - CORRECTION (Erreur) : Remplacement de l'ancienne
#   méthode pandas '.union_many()' (qui causait un crash)
#   par une boucle '.union()' moderne et compatible.
# - Alignement des métriques relatives (v7.8) conservé.
# - Graphe Annuel switchable (v7.8) conservé.
# =========================================================

# ====================================================================
# ========== SECTION DES CALCULS CACHÉS (PERFORMANCE) ==========
# ====================================================================
@st.cache_data(show_spinner="Calcul des statistiques (Rendements, Covariance)...")
def get_stats(prices: pd.DataFrame, k: int, use_lw: bool):
    # Cette fonction calcule les stats sur TOUTE la période
    # Elle est utilisée pour les onglets 'Optimisation' et 'Graphiques'
    returns = compute_returns(prices)
    mu = returns.mean().values * k
    if use_lw and HAS_LW:
        lw = LedoitWolf().fit(returns.values); cov_raw = lw.covariance_ * k
    else:
        cov_raw = returns.cov().values * k
    cov = near_psd_clip(cov_raw); cov = ridge_regularize(cov, ridge=1e-6)
    return returns, mu, cov

@st.cache_data(show_spinner="Calcul des portefeuilles optimaux (statiques)...")
def run_static_optimization(mu, cov, min_w, max_w):
    # Calcule les portefeuilles optimaux sur TOUTE la période
    # pour les onglets 'Optimisation' et 'Graphiques'
    n = len(mu)
    try:
        w_ms = max_sharpe_weights(mu, cov, min_w, max_w)
    except Exception: w_ms = feasible_start(n, min_w, max_w)
    try:
        w_mv = min_var_weights(mu, cov, min_w, max_w)
    except Exception: w_mv = feasible_start(n, min_w, max_w)
    try:
        w_mr = max_return_weights(mu, cov, min_w, max_w)
    except Exception: w_mr = feasible_start(n, min_w, max_w)
    try:
        w_rp = risk_parity_weights(cov, min_w, max_w)
    except Exception: w_rp = feasible_start(n, min_w, max_w)
    return w_ms, w_mv, w_mr, w_rp

@st.cache_data(show_spinner="Simulation Monte Carlo...")
def run_monte_carlo(mu, cov, min_w, max_w, n_mc, seed):
    n = len(mu)
    rng = np.random.default_rng(int(seed))
    W = sample_bounded_simplex(n=n, N=int(n_mc), min_w=min_w, max_w=max_w, rng=rng)
    rets_mc = W @ mu
    vols_mc = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))
    return rets_mc, vols_mc

@st.cache_data(show_spinner="Calcul de la frontière efficiente...")
def run_efficient_frontier(mu, cov, min_w, max_w):
    return efficient_frontier(mu, cov, min_w, max_w, npts=160)

try:
    # 'returns' est maintenant la source de vérité pour les rendements
    returns, mu_static, cov_static = get_stats(prices, k, use_lw)
    
    # Optimisations statiques (basées sur toute la période)
    w_ms_static, w_mv_static, w_mr_static, w_rp_static = run_static_optimization(mu_static, cov_static, min_w, max_w)
    
    # Monte Carlo et Frontière (basés sur toute la période)
    rets_mc, vols_mc = run_monte_carlo(mu_static, cov_static, min_w, max_w, n_mc, seed)
    front = run_efficient_frontier(mu_static, cov_static, min_w, max_w)

except Exception as e:
    st.error(f"Erreur irrécupérable lors de l'optimisation statique : {e}")
    st.error("Essayez d'ajuster les bornes (min_w, max_w) ou la période.")
    st.stop()

st.divider()
st.markdown("#### 3. Analyse")

# ====================================================================
# ========== SECTION DE NAVIGATION (AVEC ÉTAT) ==========
# ====================================================================

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Données" # onglet par défaut

selected_tab = st.radio(
    "Navigation principale",
    ["📊 Données", "⚙️ Optimisation", "📈 Graphiques", "🔗 Corrélation", "⏱️ Backtest"],
    key="active_tab", 
    horizontal=True,
    label_visibility="collapsed"
)


# ============================ Onglet Données ============================
if selected_tab == "📊 Données":
    st.subheader("Tableau de prix utilisé (après filtre de dates et exclusions)")
    st.dataframe(prices, width='stretch')
    if availability is not None and not availability.empty:
        start_common = pd.to_datetime(availability["First"]).max()
        end_common   = pd.to_datetime(availability["Last"]).min()
        limiters = availability.loc[availability["First"] == start_common, "Ticker"].astype(str).str.upper().tolist()
        limiters_str = ", ".join(limiters)
        st.info(
            f"La période par défaut **commence le {start_common.date()}** car "
            f"**{limiters_str}** n’a aucun historique avant cette date. "
            f"La fin par défaut (**{end_common.date()}**) correspond à la plus récente date commune."
        )
        with st.expander("Disponibilité par ticker (première / dernière date)", expanded=False):
            avail_fmt = availability.copy()
            avail_fmt["Ticker"] = avail_fmt["Ticker"].astype(str).str.upper()
            avail_fmt["First"] = pd.to_datetime(avail_fmt["First"]).dt.date
            avail_fmt["Last"]  = pd.to_datetime(avail_fmt["Last"]).dt.date
            st.dataframe(avail_fmt, hide_index=True, width='stretch')

    st.markdown("---")
    st.subheader("Aperçu graphique des prix")
    col_left, col_right = st.columns([1,1])
    with col_left:
        rebase = st.checkbox(
            "Rebase (base=100)", 
            value=True, 
            key="data_plot_rebase",
            help="Affiche l'évolution de tous les actifs à partir d'une base commune de 100, au lieu de leurs prix réels."
        )
    with col_right:
        log_scale_prices = st.checkbox(
            "Échelle log", 
            value=True, 
            key="data_plot_log",
            help="Utilise une échelle logarithmique sur l'axe Y, utile pour visualiser les variations en pourcentage sur de longues périodes."
        )

    plot_df = prices.copy()
    if rebase:
        first_prices = plot_df.iloc[0]
        safe_first_prices = first_prices.replace(0, np.nan)
        plot_df = (plot_df / safe_first_prices) * 100.0
        plot_df = plot_df.fillna(100.0)

    fig_data = go.Figure()
    for col in plot_df.columns:
        fig_data.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df[col].values,
            mode="lines",
            name=col,
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{'Index' if rebase else 'Prix'} {col}: %{{y:.2f}}<extra></extra>"
        ))
    fig_data.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title=("Index (base=100)" if rebase else "Prix"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig_data.update_xaxes(rangeslider_visible=True)
    if log_scale_prices:
        fig_data.update_yaxes(type="log")
    st_plotly_chart(fig_data)

    st.markdown("---")
    st.subheader("Inclure / Exclure des actifs")
    current_all = prices_all.columns.tolist()
    excluded_all = st.session_state["excluded"]
    available_all = [t for t in current_all if t not in excluded_all]

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Exclure des actifs**")
        to_exclude = st.multiselect(
            "Sélectionne pour exclure", 
            options=available_all, 
            key="to_exclude_list",
            help="Les actifs sélectionnés ici seront retirés de l'analyse (optimisation et backtest) lors du prochain re-calcul."
        )
        if st.button("Exclure", type="primary", key="btn_exclure"):
            new_excluded = sorted(list(set([s.upper() for s in (excluded_all + to_exclude)])))
            st.session_state["excluded"] = new_excluded
            st.success(f"Exclus : {', '.join(to_exclude)}"); _safe_rerun()
    with colB:
        st.markdown("**Réintégrer des actifs**")
        to_include = st.multiselect(
            "Sélectionne pour réintégrer", 
            options=excluded_all, 
            key="to_include_list",
            help="Réintègre des actifs précédemment exclus."
        )
        if st.button("Réintégrer", key="btn_reintegrer"):
            new_excluded = [t for t in excluded_all if t not in to_include]
            st.session_state["excluded"] = new_excluded
            st.success(f"Réintégrés : {', '.join(to_include)}"); _safe_rerun()
    st.caption(f"Actifs totaux : {len(current_all)} | Exclus : {len(st.session_state['excluded'])} | Utilisés : {len(tickers)}")
    
    st.markdown("---")
    st.subheader("Analyse individuelle des actifs")
    ticker_to_analyze = st.selectbox(
        "Choisir un actif à analyser", 
        options=tickers, 
        index=0, 
        label_visibility="collapsed", 
        key="asset_to_analyze",
        help="Sélectionnez un seul actif pour voir ses métriques de performance individuelles (basées sur la période sélectionnée)."
    )
    if ticker_to_analyze:
        idx = tickers.index(ticker_to_analyze)
        ret_ann = mu_static[idx]
        vol_ann = np.sqrt(cov_static[idx, idx])
        shp = (ret_ann / vol_ann) if vol_ann > 0 else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Rendement Ann.", 
            f"{ret_ann*100:.2f}%",
            help="Rendement annualisé moyen de cet actif sur la période."
        )
        c2.metric(
            "Volatilité Ann.", 
            f"{vol_ann*100:.2f}%",
            help="Volatilité (écart-type des rendements) annualisée de cet actif. Mesure du risque."
        )
        c3.metric(
            "Ratio de Sharpe", 
            f"{shp:.2f}" if pd.notna(shp) else "N/A",
            help="Rendement annualisé divisé par la volatilité annualisée. Mesure du rendement ajusté au risque (plus c'est élevé, mieux c'est)."
        )
        st.markdown(f"**Distribution des rendements (fréquence: {freq}) pour {ticker_to_analyze}**")
        fig_hist_ret = go.Figure()
        fig_hist_ret.add_trace(go.Histogram(
            x=returns[ticker_to_analyze], 
            name="Rendements", 
            nbinsx=40, 
            histnorm='probability density', 
            marker_color='cornflowerblue'
        ))
        fig_hist_ret.update_layout(
            template="plotly_dark", 
            yaxis_title="Densité", 
            xaxis_title=f"Rendement ({freq})", 
            barmode="overlay"
        )
        fig_hist_ret.update_traces(opacity=0.75)
        fig_hist_ret.update_xaxes(tickformat=".1%")
        st_plotly_chart(fig_hist_ret)

# ============================ Onglet Optimisation ============================
elif selected_tab == "⚙️ Optimisation":
    st.info("Cette page montre les résultats d'une optimisation **statique**, calculée sur **l'ensemble** de la période sélectionnée (section 2).")
    
    vol = np.sqrt(np.diag(cov_static)); shp = np.where(vol > 0, (mu_static)/vol, np.nan)
    st.subheader("Rendements, volatilités et Sharpe (annualisés)")
    st.caption("Statistiques individuelles des actifs, calculées sur la période sélectionnée (section 2).")
    df_metrics = pd.DataFrame({"Return_ann": mu_static, "Vol_ann": vol, "Sharpe_ann": shp}, index=tickers)
    df_metrics_fmt = df_metrics.copy()
    for c in ["Return_ann", "Vol_ann"]:
        df_metrics_fmt[c] = (df_metrics_fmt[c]*100).map(lambda v: f"{v:.2f}%")
    df_metrics_fmt["Sharpe_ann"] = df_metrics_fmt["Sharpe_ann"].map(lambda v: f"{v:.2f}")
    st.dataframe(df_metrics_fmt, width='stretch')

    def metrics_from_w(name, w):
        ret, vol_ = portfolio_perf(w, mu_static, cov_static); sharpe = (ret)/vol_ if vol_ > 0 else 0
        return {"Portefeuille": name, "Return": ret, "Vol": vol_, "Sharpe": sharpe}

    df_res = pd.DataFrame([
        metrics_from_w("Max Sharpe", w_ms_static),
        metrics_from_w("Min Variance", w_mv_static),
        metrics_from_w("Max Return", w_mr_static),
        metrics_from_w("Risk Parity", w_rp_static)
    ]).set_index("Portefeuille")

    st.subheader("Résultats de l’optimisation")
    st.caption("Portefeuilles optimaux calculés selon la théorie de Markowitz, en respectant les contraintes de poids (section 2).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        r = df_res.loc["Max Sharpe"]
        st.metric(
            label="🏆 Max Sharpe", 
            value=f"{r['Sharpe']:.2f}", 
            delta=f"Ret: {r['Return']:.1%} | Vol: {r['Vol']:.1%}", 
            delta_color="off",
            help="Portefeuille qui maximise le Ratio de Sharpe (rendement ajusté au risque) sur la frontière efficiente."
        )
    with c2:
        r = df_res.loc["Min Variance"]
        st.metric(
            label="🛡️ Min Variance", 
            value=f"{r['Vol']:.1%}", 
            delta=f"Ret: {r['Return']:.1%} | Sharpe: {r['Sharpe']:.2f}", 
            delta_color="off",
            help="Portefeuille qui minimise la volatilité (risque) totale, quel que soit le rendement."
        )
    with c3:
        r = df_res.loc["Risk Parity"]
        st.metric(
            label="⚖️ Risk Parity", 
            value=f"{r['Sharpe']:.2f}", 
            delta=f"Ret: {r['Return']:.1%} | Vol: {r['Vol']:.1%}", 
            delta_color="off",
            help="Portefeuille qui tente d'égaliser la contribution au risque de chaque actif (non basé sur la frontière efficiente)."
        )
    with c4:
        r = df_res.loc["Max Return"]
        st.metric(
            label="🚀 Max Return", 
            value=f"{r['Return']:.1%}", 
            delta=f"Vol: {r['Vol']:.1%} | Sharpe: {r['Sharpe']:.2f}", 
            delta_color="off",
            help="Portefeuille qui maximise le rendement, quel que soit le risque (généralement 100% sur l'actif le plus performant)."
        )
    st.markdown("---")

    st.subheader("Contributions au risque (somme = 100%)")
    st.caption("Pour chaque portefeuille, montre quelle part du risque total (en %) provient de chaque actif. Pour Risk Parity, ces parts devraient être quasi-égales.")
    df_rc = pd.DataFrame({
        "Max Sharpe": risk_contrib(w_ms_static, cov_static),
        "Min Variance": risk_contrib(w_mv_static, cov_static),
        "Risk Parity": risk_contrib(w_rp_static, cov_static),
        "Max Return": risk_contrib(w_mr_static, cov_static)
    }, index=tickers)
    st.dataframe((df_rc*100).round(2).astype(str) + " %", width='stretch')

# ============================ Onglet Graphiques ============================
elif selected_tab == "📈 Graphiques":
    st.info("Cette page montre les résultats d'une optimisation **statique**, calculée sur **l'ensemble** de la période sélectionnée (section 2).")
    
    n = len(tickers)
    st.subheader("Nuage de portefeuilles, frontière efficiente (bornée) et portefeuilles optimaux")
    st.caption("Visualisation de l'optimisation. Chaque point est un portefeuille.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols_mc, y=rets_mc, mode="markers",
        marker=dict(size=4, opacity=0.35, color="#0A84FF"), 
        name="Portefeuilles aléatoires (bornés)",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
    ))
    if front.size > 0:
        fig.add_trace(go.Scatter(
            x=front[:,1], y=front[:,0], mode="lines",
            name="Frontière efficiente",
            line=dict(width=3, color="white"), 
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
        ))
    for name, w, sym, c in [
            ("Max Sharpe", w_ms_static, "star", "red"), 
            ("Min Variance", w_mv_static, "circle", "green"), 
            ("Risk Parity", w_rp_static, "diamond", "purple"),
            ("Max Return", w_mr_static, "square", "cyan")
        ]:
        r, v = portfolio_perf(w, mu_static, cov_static)
        fig.add_trace(go.Scatter(
            x=[v], y=[r], mode="markers", name=name,
            marker=dict(size=12, symbol=sym, color=c, line=dict(width=1, color="black")),
            hovertemplate=f"{name}<br>Vol: %{{x:.2%}}<br>Ret: %{{y:.2%}}<extra></extra>"
        ))
    fig.update_layout(
        xaxis_title="Volatilité (ann.)", yaxis_title="Rendement (ann.)",
        template="plotly_dark", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest"
    )
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
    st_plotly_chart(fig)

    def pie_series(weights: np.ndarray, labels: list, threshold: float = 0.02, eps: float = 1e-6) -> pd.Series:
        s = pd.Series(weights, index=labels); s = s[s > eps]
        if s.empty: return pd.Series([1.0], index=["N/A"])
        s = s.sort_values(ascending=False)
        if threshold and threshold > 0:
            majors = s[s >= threshold]; others = s[s < threshold].sum()
            if others > eps: majors.loc["Others"] = others
            s = majors
        s = s / s.sum(); return s

    s_ms = pie_series(w_ms_static, tickers, threshold=0.02)
    s_mv = pie_series(w_mv_static, tickers, threshold=0.02)
    s_mr = pie_series(w_mr_static, tickers, threshold=0.02)
    s_rp = pie_series(w_rp_static, tickers, threshold=0.02)

    pies = make_subplots(rows=1, cols=4, specs=[[{"type": "domain"}]*4],
                         subplot_titles=("Max Sharpe", "Min Variance", "Risk Parity", "Max Return"))
    def add_pie(fig_, r, c, s: pd.Series):
        if len(s) == 1:
            fig_.add_trace(go.Pie(labels=[s.index[0]], values=[1.0], hole=0.35,
                                  sort=False, textinfo="label+percent", textposition="inside", showlegend=False), r, c)
        else:
            fig_.add_trace(go.Pie(labels=s.index.tolist(), values=s.values.tolist(), hole=0.35,
                                  sort=False, textinfo="percent+label"), r, c)
    add_pie(pies, 1, 1, s_ms); add_pie(pies, 1, 2, s_mv); add_pie(pies, 1, 3, s_rp); add_pie(pies, 1, 4, s_mr)
    pies.update_layout(template="plotly_dark", showlegend=False)
    st_plotly_chart(pies)

    st.subheader("Poids par portefeuille (%)")
    st.caption("Allocation détaillée (en pourcentage) pour chaque portefeuille optimal calculé.")
    weights_df = pd.DataFrame({
        "Max Sharpe": w_ms_static, 
        "Min Variance": w_mv_static, 
        "Risk Parity": w_rp_static,
        "Max Return": w_mr_static
    }, index=tickers)
    st.dataframe((weights_df * 100).round(2).astype(str) + " %", width='stretch')

# ============================ Onglet Corrélation ============================
elif selected_tab == "🔗 Corrélation":
    st.subheader("Paramètres de corrélation")
    c1, c2, c3 = st.columns(3)
    with c1:
        corr_type = st.selectbox(
            "Type de corrélation", 
            ["Pearson", "Spearman"], 
            index=0, 
            key="corr_type_select",
            help="Méthode de calcul : 'Pearson' (linéaire) ou 'Spearman' (basée sur les rangs, non linéaire)."
        )
    with c2:
        months_win = st.number_input(
            "Fenêtre (mois) pour la matrice/rolling", 
            value=24, 
            step=1, 
            min_value=3, 
            max_value=240, 
            key="months_win_input",
            help="Nombre de mois à utiliser pour calculer la matrice de corrélation et la corrélation glissante."
        )
    with c3:
        freq_used = freq
    win = months_to_periods(int(months_win), freq_used)
    st.caption(f"Fenêtre utilisée : {win} périodes ({months_win} mois, fréquence {freq_used})")

    corr_data = returns.iloc[-win:] if len(returns) >= win else returns
    corr = corr_data.corr(method="spearman" if corr_type=="Spearman" else "pearson")
    ordered = cluster_order_from_corr(corr); corr_ord = corr.loc[ordered, ordered]

    st.subheader("Matrice de corrélation (ordre clusterisé)")
    st.caption("Matrice de corrélation (-1 à +1) calculée sur la fenêtre spécifiée. Les actifs sont regroupés par similarité (clustering).")
    heat = go.Figure(data=go.Heatmap(
        z=corr_ord.values, x=corr_ord.columns, y=corr_ord.index,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0, colorbar=dict(title="Corr")
    ))
    heat.update_layout(template="plotly_dark", height=600, xaxis_showgrid=False, yaxis_showgrid=False)
    st_plotly_chart(heat)

    top_pos, top_neg = top_corr_pairs(corr, k=3)
    st.subheader("Top corrélations (fenêtre courante)")
    colp, coln = st.columns(2)
    with colp:
        st.markdown("Top 3 positives")
        if top_pos:
            dfp = pd.DataFrame([{"Actif A": a, "Actif B": b, "Corrélation": f"{c:.2f}"} for a,b,c in top_pos])
            st.table(dfp)
        else:
            st.write("Aucune paire.")
    with coln:
        st.markdown("Top 3 négatives")
        if top_neg:
            dfn = pd.DataFrame([{"Actif A": a, "Actif B": b, "Corrélation": f"{c:.2f}"} for a,b,c in top_neg])
            st.table(dfn)
        else:
            st.write("Aucune paire.")

    st.subheader("Corrélation roulante (deux actifs)")
    st.caption("Évolution de la corrélation entre deux actifs dans le temps, calculée sur la fenêtre glissante spécifiée.")
    c1, c2 = st.columns(2)
    with c1:
        asset_a = st.selectbox("Actif A", tickers, index=0, key="corr_a")
    with c2:
        asset_b = st.selectbox("Actif B", tickers, index=min(1, len(tickers)-1), key="corr_b")
    if asset_a != asset_b:
        s1, s2 = returns[asset_a], returns[asset_b]
        if corr_type == "Spearman":
            s1r, s2r = s1.rank(method="average"), s2.rank(method="average")
            roll_corr = s1r.rolling(win).corr(s2r)
        else:
            roll_corr = s1.rolling(win).corr(s2)
        figc = go.Figure()
        figc.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", annotation_text="Corrélation Zéro", annotation_position="top right")
        figc.add_trace(go.Scatter(
            x=roll_corr.index, y=np.where(roll_corr < 0, roll_corr, 0),
            mode='lines', line=dict(width=0), fill='tozeroy',
            fillcolor='rgba(255, 128, 128, 0.5)', name='Corrélation Négative', showlegend=False
        ))
        figc.add_trace(go.Scatter(
            x=roll_corr.index, y=np.where(roll_corr > 0, roll_corr, 0),
            mode='lines', line=dict(width=0), fill='tozeroy',
            fillcolor='rgba(144, 238, 144, 0.5)', name='Corrélation Positive', showlegend=False
        ))
        figc.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr.values, mode='lines',
            line=dict(color='deepskyblue', width=2),
            name=f"Corrélation ({asset_a}, {asset_b})",
            hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>Corrélation: %{{y:.2f}}<extra></extra>"
        ))
        figc.update_layout(
            template="plotly_dark", 
            yaxis_title="Coefficient de Corrélation", xaxis_title="Date",
            title=f"Corrélation Glissante ({win} Périodes) : {asset_a} vs. {asset_b}",
            yaxis_range=[-1, 1], legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
            hovermode="x unified"
        )
        st_plotly_chart(figc)
    else:
        st.info("Sélectionnez deux actifs distincts.")

# ============================ Onglet Backtest ============================
elif selected_tab == "⏱️ Backtest":
    
    st.subheader("Configuration du Backtest")
    n = len(tickers)
    
    # --- UI BLOC 1: Stratégie d'allocation ---
    with st.container(border=True):
        st.markdown("##### 1. Stratégie d'allocation")
        
        c1, c2 = st.columns(2)
        with c1:
            alloc_mode = st.radio(
                "Mode d'allocation",
                ["Statique", "Dynamique (Roulante)"],
                index=1,
                horizontal=True,
                key="bt_alloc_mode",
                help="**Statique** : Utilise une seule allocation (calculée sur toute la période) et la conserve. **Dynamique (Roulante)** : Recalcule l'allocation périodiquement en utilisant uniquement les données passées (plus réaliste)."
            )
        with c2:
            strategy_choice = st.selectbox(
                "Stratégie (Allocation cible)",
                ["Manuel", "Égal-pondéré (1/N)", "Max Sharpe", "Min Variance", "Risk Parity"],
                index=2, # Max Sharpe par défaut
                key="bt_strategy_choice",
                help="La stratégie utilisée pour déterminer les poids cibles. 'Manuel' utilise l'éditeur ci-dessous. Les autres sont des optimisations."
            )
        
        # --- Options pour le mode Dynamique ---
        lookback_months = None
        lookback_n = None
        if alloc_mode == "Dynamique (Roulante)":
            lookback_months = st.slider(
                "Fenêtre de calcul (Lookback)",
                min_value=12,
                max_value=120,
                value=36,
                step=6,
                key="bt_lookback_months",
                help="Nombre de mois de données passées à utiliser pour recalculer l'allocation à chaque rebalancement (ex: 36 mois)."
            )
            lookback_n = months_to_periods(lookback_months, freq)
            if lookback_n < len(tickers) * 2:
                st.warning(f"La fenêtre de {lookback_n} périodes ({lookback_months} mois) est très courte pour {len(tickers)} actifs. L'optimisation pourrait être instable.")
            st.caption(f"Fenêtre de calcul : {lookback_months} mois = {lookback_n} périodes ({freq}).")
        
        # --- Éditeur pour le mode Manuel ---
        w_manual = None
        if strategy_choice == "Manuel":
            st.info("Modifiez les poids ci-dessous. La somme sera normalisée à 100%.")
            df_edit = pd.DataFrame({"Ticker": tickers, "Poids_%": (np.ones(n) / n * 100)})
            edited = st.data_editor(
                df_edit,
                num_rows="fixed",
                column_config={"Poids_%": st.column_config.NumberColumn("Poids (%)", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f")},
                width='stretch',
                key="weights_editor"
            )
            w_user_pct = np.asarray(edited["Poids_%"].values, dtype=float)
            total_pct = float(np.nansum(w_user_pct))
            
            if not np.isfinite(total_pct) or total_pct <= 0:
                st.error("La somme des poids est nulle ou invalide."); st.stop()
            if total_pct > 100.0 + 1e-9:
                st.error(f"La somme des poids ({total_pct:.2f}%) dépasse 100%."); st.stop()
            
            if total_pct < 100.0 - 1e-9:
                st.caption(f"Somme actuelle {total_pct:.2f}%. Normalisation à 100% appliquée.")
            w_manual = np.clip(w_user_pct, 0, None) / total_pct
            
            if alloc_mode == "Dynamique (Roulante)":
                st.warning("Le mode 'Manuel' est toujours 'Statique'. Passage en mode Statique.")
                alloc_mode = "Statique"

    # --- DÉTERMINATION DE LA STRATÉGIE (AVANT BLOC 2) ---
    _strategy_func = None
    w_static = None
    
    if strategy_choice == "Manuel":
        w_static = w_manual
    elif strategy_choice == "Égal-pondéré (1/N)":
        w_static = np.ones(n) / n
    else:
        strategy_map = {
            "Max Sharpe": max_sharpe_weights,
            "Min Variance": min_var_weights,
            "Risk Parity": risk_parity_weights
        }
        if alloc_mode == "Statique":
            static_weights_map = {
                "Max Sharpe": w_ms_static,
                "Min Variance": w_mv_static,
                "Risk Parity": w_rp_static
            }
            w_static = static_weights_map[strategy_choice]
        else:
            _strategy_func = strategy_map[strategy_choice]
            if lookback_n is None:
                st.error("Erreur : Mode Dynamique sélectionné mais 'lookback_n' non défini.")
                st.stop()

    # --- UI BLOC 2: Rebalancement et Frais (v7.5) ---
    with st.container(border=True):
        st.markdown("##### 2. Rebalancement et Frais")
        c1, c2, c3 = st.columns(3)
        
        with c2:
            fee_ter_ann_pct = st.number_input(
                "Frais de gestion ann. (TER) (%)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key="bt_fee_ter",
                help="Frais de gestion annuels (Total Expense Ratio), ex: 0.25 pour 0.25%. Déduits de la performance au pro-rata de la période."
            )
        with c3:
            fee_txn_pct = st.number_input(
                "Frais de transaction (%)",
                min_value=0.0,
                max_value=10.0,
                value=0.10,
                step=0.01,
                format="%.2f",
                key="bt_fee_txn",
                help="Coût (en %) appliqué à chaque transaction (achat ou vente) lors du rebalancement. Ex: 0.10 pour 0.10%."
            )

        with c1:
            reb_opts_scan = ["Mensuel", "Trimestriel", "Annuel"]
            all_reb_opts = ["Jamais (Buy & Hold)", "Mensuel", "Trimestriel", "Annuel", "Tous les N périodes"]
            
            # Utilisation de st.cache_data pour la fonction de scan
            @st.cache_data(show_spinner="Scan des fréquences de rebalancement...")
            def _get_best_reb_freq(
                returns_hash, strategy_choice_key, alloc_mode_key, lookback_n_key, 
                min_w_key, max_w_key, fee_ter_key, fee_txn_key, k_key
            ):
                # Helper function pour le scan (ne peut pas être cachée directement)
                def _score_for(choice_str):
                    n_trial = n_for_rebalance_choice(choice_str, freq, n_custom=1)
                    # Note : On recalcule w_static / _strategy_func ici
                    w_static_scan = None
                    _strategy_func_scan = None
                    if strategy_choice_key == "Manuel": w_static_scan = w_manual
                    elif strategy_choice_key == "Égal-pondéré (1/N)": w_static_scan = np.ones(n) / n
                    else:
                        strategy_map_scan = { "Max Sharpe": max_sharpe_weights, "Min Variance": min_var_weights, "Risk Parity": risk_parity_weights }
                        if alloc_mode_key == "Statique":
                            static_weights_map_scan = { "Max Sharpe": w_ms_static, "Min Variance": w_mv_static, "Risk Parity": w_rp_static }
                            w_static_scan = static_weights_map_scan[strategy_choice_key]
                        else:
                            _strategy_func_scan = strategy_map_scan[strategy_choice_key]
                    
                    w_trial, _, _ = run_backtest(
                        returns=returns,
                        _strategy_func=_strategy_func_scan,
                        w_static=w_static_scan,
                        rebalance_freq_n=n_trial,
                        lookback_window_n=lookback_n_key,
                        min_w=min_w_key,
                        max_w=max_w_key,
                        fee_ter_ann_pct=fee_ter_key,
                        fee_txn_pct=fee_txn_key,
                        k=k_key
                    )
                    m_trial, _ = perf_metrics(w_trial, freq_k=k_key)
                    s = m_trial.get("Sharpe", np.nan)
                    return s if pd.notna(s) else -np.inf

                scores = {opt: _score_for(opt) for opt in reb_opts_scan}
                best_opt = max(scores, key=scores.get) if scores else "Trimestriel"
                return best_opt

            returns_hash = pd.util.hash_pandas_object(returns).sum()
            
            best_opt = _get_best_reb_freq(
                returns_hash, strategy_choice, alloc_mode, lookback_n, 
                min_w, max_w, fee_ter_ann_pct, fee_txn_pct, k
            )
            
            if strategy_choice in ["Manuel", "Égal-pondéré (1/N)"] and alloc_mode == "Statique":
                default_index = 0 # "Jamais (Buy & Hold)"
                best_opt_display = "Jamais (Buy & Hold)"
            else:
                default_index = all_reb_opts.index(best_opt)
                best_opt_display = best_opt

            reb_choice = st.selectbox(
                "Fréquence de rebalancement",
                all_reb_opts,
                index=default_index, 
                key="bt_reb_choice",
                help="**Jamais (Buy & Hold)** : L'allocation dérive avec le marché. **Autre** : L'allocation est réinitialisée aux poids cibles à la fréquence choisie."
            )
            
            if default_index > 0: 
                st.caption(f"Défaut auto-sélectionné : **{best_opt_display}** (Sharpe le plus élevé).")
                st.warning("Attention : Le 'meilleur' rebalancement passé (calculé sur tout l'historique) n'est pas garanti d'être le meilleur à l'avenir (risque de sur-optimisation).", icon="⚠️")
            
            n_custom = 1
            if reb_choice == "Tous les N périodes":
                n_custom = st.number_input(
                    f"N Périodes ({freq})", value=1, min_value=1, step=1, key="bt_reb_n_custom"
                )

    # --- Logique de préparation du backtest ---
    if reb_choice == "Jamais (Buy & Hold)":
        n_reb = None 
    elif reb_choice == "Tous les N périodes":
        n_reb = max(1, int(n_custom))
    else:
        freq_map_n = {"Mensuel": "monthly", "Trimestriel": "quarterly", "Annuel": "yearly"}
        n_reb = months_to_periods(
            {"monthly": 1, "quarterly": 3, "yearly": 12}[freq_map_n[reb_choice]], 
            freq
        )
    
    if alloc_mode == "Dynamique (Roulante)" and n_reb is None:
        st.error("Le mode 'Dynamique' est incompatible avec 'Jamais (Buy & Hold)'.")
        st.info("Le rebalancement est nécessaire pour appliquer les nouveaux poids. Veuillez choisir une fréquence (ex: Annuel).")
        st.stop()


    # --- Exécution du Moteur de Backtest ---
    try:
        wealth, weights_history, target_weights_log = run_backtest(
            returns=returns,
            _strategy_func=_strategy_func, 
            w_static=w_static,
            rebalance_freq_n=n_reb,
            lookback_window_n=lookback_n,
            min_w=min_w,
            max_w=max_w,
            fee_ter_ann_pct=fee_ter_ann_pct,
            fee_txn_pct=fee_txn_pct,
            k=k
        )
    except Exception as e:
        st.error(f"Erreur lors de l'exécution du backtest : {e}")
        st.exception(e) 
        st.stop()

    metrics, dd = perf_metrics(wealth, freq_k=k)
    ret_series = wealth.pct_change().dropna()
    
    st.divider()
    st.subheader("Métriques de performance du backtest")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("CAGR (Net de frais)", f"{metrics['CAGR']*100:.2f}%" if pd.notna(metrics["CAGR"]) else "N/A", help="Compound Annual Growth Rate : Le taux de croissance annuel composé (net de frais de gestion et de transaction).")
    kpi2.metric("Volatilité Ann.", f"{metrics['Vol ann.']*100:.2f}%" if pd.notna(metrics["Vol ann."]) else "N/A", help="Volatilité (écart-type) annualisée des rendements du portefeuille. C'est la mesure standard du risque.")
    kpi3.metric("Ratio de Sharpe (Net)", f"{metrics['Sharpe']:.2f}" if pd.notna(metrics["Sharpe"]) else "N/A", help="CAGR (Net) divisé par la Volatilité Annuelle. Mesure le rendement ajusté au risque. Plus il est élevé, mieux c'est.")
    kpi4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%" if pd.notna(metrics["Max Drawdown"]) else "N/A", delta_color="inverse", help="La plus grande perte (en pourcentage) depuis un pic jusqu'à un creux subséquent durant la période de backtest.")
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    kpi5.metric("Ratio de Sortino (Net)", f"{metrics['Sortino']:.2f}" if pd.notna(metrics["Sortino"]) else "N/A", help="Similaire au Sharpe, mais ne pénalise que la 'mauvaise' volatilité (rendements négatifs) au lieu de la volatilité totale.")
    kpi6.metric("Ratio de Calmar (Net)", f"{metrics['Calmar']:.2f}" if pd.notna(metrics["Calmar"]) else "N/A", help="CAGR (Net) divisé par la valeur absolue du Max Drawdown. Mesure le rendement par unité de risque de 'drawdown'.")
    kpi7.metric("Total Return (Net)", f"{metrics['Total Return']*100:.2f}%", help="Le gain total (net de frais) du portefeuille sur l'ensemble de la période.")
    kpi8.metric("Worst Period", f"{metrics['Worst period']*100:.2f}%" if pd.notna(metrics["Worst period"]) else "N/A", delta_color="inverse", help=f"La pire performance sur une seule période (ici, la pire fréquence : '{freq}').")

    # --- Benchmark section ---
    bench_returns = None
    bench_wealth = None
    bench_dd = None
    st.divider()
    st.subheader("Comparaison Benchmark")
    bench_c1, bench_c2 = st.columns([1,1])
    with bench_c1:
        bench_ticker = st.text_input("Ticker Benchmark", value="^GSPC", key="benchmark_ticker_input", help="Ticker Yahoo Finance (ex: ^GSPC pour S&P 500, 'IEUR.AS' pour un ETF) à utiliser comme référence.")
    with bench_c2:
        st.markdown("<br/>", unsafe_allow_html=True)
        run_bench = st.checkbox("Lancer la comparaison au benchmark", value=True, key="run_bench_checkbox")
    bench_annual_ret = None
    bench_monthly_ret = None 
    regime_metrics = {} 
    if run_bench and bench_ticker:
        if not HAS_YF:
            st.error("yfinance n’est pas installé : `pip install yfinance` pour activer la comparaison au benchmark.")
        else:
            freq_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo", "quarterly": "3mo", "yearly": "1y"}
            yf_interval_bench = freq_map.get(freq, "1d")
            bench_prices = fetch_benchmark_prices(bench_ticker.upper(), prices.index.min(), prices.index.max(), yf_interval_bench)
            if not bench_prices.empty:
                aligned_bench_prices = bench_prices.reindex(returns.index).ffill()
                bench_returns_aligned = aligned_bench_prices.pct_change().dropna()
                if isinstance(bench_returns_aligned, pd.DataFrame): bench_returns_aligned = bench_returns_aligned.squeeze("columns")
                port_returns_common, bench_returns_common = ret_series.align(bench_returns_aligned, join="inner")
                if isinstance(port_returns_common, pd.DataFrame): port_returns_common = port_returns_common.squeeze("columns")
                if isinstance(bench_returns_common, pd.DataFrame): bench_returns_common = bench_returns_common.squeeze("columns")
                if (port_returns_common is not None) and (bench_returns_common is not None) and (not port_returns_common.empty) and (not bench_returns_common.empty):
                    bench_returns = bench_returns_common
                    st.markdown("**Métriques relatives**")
                    rel_metrics = calculate_relative_metrics(port_returns_common, bench_returns_common, k)
                    
                    # --- DÉBUT MODIFICATION v7.8 : Alignement des métriques ---
                    rkpi1, rkpi2, rkpi3, rkpi4 = st.columns(4)
                    rkpi1.metric("Beta", f"{rel_metrics['Beta']:.2f}" if pd.notna(rel_metrics["Beta"]) else "N/A", help="Mesure de la volatilité du portefeuille par rapport au benchmark. Beta > 1 = plus volatil. Beta < 1 = moins volatil.")
                    rkpi2.metric("Alpha (ann.)", f"{rel_metrics['Alpha (ann.)']*100:.2f}%" if pd.notna(rel_metrics["Alpha (ann.)"]) else "N/A", help="Surperformance (ou sous-performance) annualisée du portefeuille par rapport au rendement attendu (basé sur le Beta).")
                    rkpi3.metric("Tracking Error", f"{rel_metrics['Tracking Error']*100:.2f}%" if pd.notna(rel_metrics["Tracking Error"]) else "N/A", help="Volatilité (écart-type) de la *différence* de rendement entre le portefeuille et le benchmark.")
                    rkpi4.metric("Information Ratio", f"{rel_metrics['Information Ratio']:.2f}" if pd.notna(rel_metrics["Information Ratio"]) else "N/A", help="Surperformance (Portefeuille - Benchmark) divisée par la Tracking Error. Mesure la constance de la surperformance.")
                    
                    bench_wealth = (1 + bench_returns).cumprod()
                    if not bench_wealth.empty:
                        bench_wealth.iloc[0] = 1.0
                        bw_aligned = bench_wealth.reindex(dd.index)
                        bench_dd = bw_aligned / bw_aligned.cummax() - 1.0
                        bench_annual_ret = (1 + bench_returns).resample("YE").prod() - 1
                        bench_annual_ret.index = bench_annual_ret.index.year
                        bench_annual_ret.name = f"Benchmark ({bench_ticker})"
                        bench_monthly_ret = (1 + bench_returns).resample("ME").prod() - 1
                        monthly_ret_for_regime = (1 + port_returns_common).resample("ME").prod() - 1
                        regime_metrics = calculate_market_regime_metrics(monthly_ret_for_regime, bench_monthly_ret)
                        
                        rkpi5, rkpi6, rkpi7, _ = st.columns(4) # Alignement sur 4 colonnes
                        rkpi5.metric("Batting Average", f"{regime_metrics.get('Batting Average', np.nan)*100:.1f}%", help="Pourcentage de périodes (mois) où le portefeuille a surperformé le benchmark.")
                        rkpi6.metric("Up-Capture Ratio", f"{regime_metrics.get('Up-Capture', np.nan):.1f}%", help="Pourcentage de la performance du benchmark capturée par le portefeuille pendant les mois de *hausse* du benchmark. >100% = surperformance en hausse.")
                        rkpi7.metric("Down-Capture Ratio", f"{regime_metrics.get('Down-Capture', np.nan):.1f}%", help="Pourcentage de la performance du benchmark subie par le portefeuille pendant les mois de *baisse* du benchmark. <100% = protection en baisse.")
                    # --- FIN MODIFICATION v7.8 ---
                    else: 
                        st.warning("Impossible d'aligner benchmark et portefeuille (pas de dates communes suffisantes).")

    st.subheader("Statistiques des rendements mensuels")
    monthly_ret = (1 + ret_series).resample("ME").prod() - 1
    risk_metrics = calculate_var_cvar(monthly_ret, percentile=0.95)
    pv_df = pd.DataFrame({
        "Moyenne Arithm. (ann)": [f"{((1+monthly_ret.mean())**12 - 1)*100:.2f}%"] if len(monthly_ret)>0 else ["N/A"],
        "Moyenne Géo. (ann)":  [f"{(((1+monthly_ret).prod())**(12/len(monthly_ret)) - 1)*100:.2f}%"] if len(monthly_ret)>0 else ["N/A"],
        "Std (ann)":                 [f"{monthly_ret.std(ddof=1)*np.sqrt(12)*100:.2f}%"] if len(monthly_ret)>0 else ["N/A"],
        "Skewness": [f"{monthly_ret.skew():.2f}"] if len(monthly_ret)>0 else ["N/A"],
        "Kurtosis (Excess)": [f"{monthly_ret.kurtosis():.2f}"] if len(monthly_ret)>0 else ["N/A"],
        "VaR (95%)": [f"{risk_metrics['VaR']*100:.2f}%"] if pd.notna(risk_metrics['VaR']) else ["N/A"],
        "CVaR (95%)": [f"{risk_metrics['CVaR']*100:.2f}%"] if pd.notna(risk_metrics['CVaR']) else ["N/A"]
    })
    st.dataframe(pv_df, width='stretch')
    st.caption("Skewness : Asymétrie de la distribution (0=symétrique). | Kurtosis : Aplatissement de la distribution (>0=queues épaisses). | VaR (Value-at-Risk) 95% : Perte mensuelle maximale attendue 95% du temps (pire que 1 mois sur 20). | CVaR (Conditional VaR) 95% : Perte mensuelle moyenne attendue lors des 5% pires mois (si la VaR est dépassée).")

    st.subheader("Courbes de backtest")
    log_scale = st.checkbox("Échelle log (Wealth)", value=True, key="wealth_log_checkbox", help="Utilise une échelle logarithmique pour le graphique 'Valeur du portefeuille', utile pour comparer les taux de croissance.")
    wealth_plot = wealth.where(wealth > 0, np.nan) if log_scale else wealth
    figw = go.Figure()
    figw.add_trace(go.Scatter(x=wealth_plot.index, y=wealth_plot.values, mode="lines", name="Portefeuille", line=dict(color="#0A84FF")))
    if bench_wealth is not None and not bench_wealth.empty:
        figw.add_trace(go.Scatter(x=bench_wealth.index, y=bench_wealth.values, mode="lines", name=f"Benchmark ({bench_ticker})", line=dict(color='gray', dash='dash')))
    figw.update_layout(template="plotly_dark", yaxis_title="Valeur du portefeuille (Nette de frais)", xaxis_title="Date", legend=dict(x=0.01, y=0.99))
    if log_scale: figw.update_yaxes(type="log")
    st_plotly_chart(figw)
    
    st.divider()
    figd = go.Figure()
    figd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown Portefeuille", fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='crimson')))
    if bench_dd is not None and not bench_dd.dropna().empty:
        figd.add_trace(go.Scatter(
            x=bench_dd.index, y=bench_dd.values, mode="lines",
            name=f"Drawdown Benchmark ({bench_ticker})",
            line=dict(color='steelblue', dash='dot'),
            fill='tozeroy', fillcolor='rgba(70,130,180,0.15)'
        )) # Correction v7.8 (parenthèse en trop retirée)
    figd.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    figd.update_layout(template="plotly_dark", yaxis_title="Drawdown", xaxis_title="Date", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), margin=dict(t=80))
    figd.update_yaxes(tickformat=".0%")
    st_plotly_chart(figd)

    st.divider()
    st.subheader("Rendements Annuels")
    
    # --- DÉBUT MODIFICATION v7.8 : Radio button pour la vue ---
    view_mode_annual = st.radio(
        "Changer la vue du graphique :",
        ["Portefeuille vs. Benchmark", "Actifs Individuels"],
        horizontal=True,
        label_visibility="collapsed",
        key="annual_view_mode"
    )

    try:
        fig_annual = go.Figure()
        
        if view_mode_annual == "Portefeuille vs. Benchmark":
            # --- VUE 1 : Portefeuille vs. Benchmark ---
            port_annual_ret = (1 + ret_series).resample("YE").prod() - 1
            port_annual_ret.index = port_annual_ret.index.year
            port_annual_ret.name = "Portefeuille (Net)"
            
            all_series_to_plot = {port_annual_ret.name: port_annual_ret}
            if bench_annual_ret is not None and not bench_annual_ret.empty:
                all_series_to_plot[bench_annual_ret.name] = bench_annual_ret

            all_indices = [s.index for s in all_series_to_plot.values()]
            
            # --- DÉBUT CORRECTION v7.9 : Remplacement de .union_many() ---
            all_years = pd.Index([])
            if all_indices:
                all_years = all_indices[0] # Commence par le premier
                if len(all_indices) > 1:
                    for idx in all_indices[1:]: # Union avec les suivants
                        all_years = all_years.union(idx)
            all_years = all_years.sort_values() # Trie à la fin
            # --- FIN CORRECTION v7.9 ---

            if not all_years.empty:
                for name, series in all_series_to_plot.items():
                    vals_aligned = series.reindex(all_years)
                    color = "#0A84FF" if name == "Portefeuille (Net)" else "gray"
                    fig_annual.add_trace(go.Bar(
                        x=all_years, y=vals_aligned.values, name=name,
                        marker_color=color, 
                        hovertemplate="Année: %{x}<br>Rendement: %{y:.2%}<extra></extra>"
                    ))
            fig_annual.update_layout(title_text="Rendements Annuels (Portefeuille vs. Benchmark)")
            
        else:
            # --- VUE 2 : Actifs Individuels ---
            assets_annual_ret_df = (1 + returns).resample("YE").prod() - 1
            assets_annual_ret_df.index = assets_annual_ret_df.index.year
            all_years = assets_annual_ret_df.index
            
            for asset in assets_annual_ret_df.columns:
                fig_annual.add_trace(go.Bar(
                    x=all_years, 
                    y=assets_annual_ret_df[asset], 
                    name=asset,
                    hovertemplate="Année: %{x}<br>Rendement: %{y:.2%}<extra></extra>"
                ))
            fig_annual.update_layout(title_text="Rendements Annuels (Actifs Individuels)")

        # Paramètres communs aux deux graphiques
        fig_annual.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        fig_annual.update_layout(
            template="plotly_dark", barmode='group', yaxis_title="Rendement Annuel",
            xaxis_title="Année", yaxis_tickformat=".0%",
            xaxis=dict(tickmode='linear', dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st_plotly_chart(fig_annual)

    except Exception as e_annual:
        st.warning(f"Impossible de générer le graphique des rendements annuels : {e_annual}")
    # --- FIN MODIFICATION v7.8 ---
    
    st.divider()
    st.subheader("Calendrier des rendements mensuels (Heatmap)")
    st.caption("Visualisation des rendements mensuels (nets de frais) pour chaque mois et chaque année. Vert = positif, Rouge = négatif.")
    heatmap_df = create_monthly_heatmap_df(monthly_ret)
    if not heatmap_df.empty:
        z_std = monthly_ret.std()
        z_min_bound = max(-z_std * 2.5, monthly_ret.min()) 
        z_max_bound = min(z_std * 2.5, monthly_ret.max()) 
        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale='RdYlGn', 
            zmid=0, 
            zmin=z_min_bound,
            zmax=z_max_bound,
            text=heatmap_df.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""),
            texttemplate="%{text}",
            textfont={"size":10, "color":"black"},
            hovertemplate="<b>%{y} - %{x}</b><br>Rendement: %{z:.2%}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Rend. %", tickformat=".1%")
        ))
        fig_heat.update_layout(template="plotly_dark", yaxis_title="Année", xaxis_title="Mois", xaxis=dict(showgrid=False, side="top"), yaxis=dict(tickmode='linear', showgrid=False))
        st_plotly_chart(fig_heat)
    else:
        st.info("Pas assez de données pour générer la heatmap mensuelle (nécessite au moins 1 mois).")
    
    st.subheader("Évolution de l'allocation du portefeuille")
    
    st.caption(
        "Ce graphique montre l'évolution de la valeur relative de chaque actif dans le portefeuille. "
        "En mode 'Buy & Hold', il montre la **dérive**. "
        "En mode 'Rebalancé' ou 'Dynamique', il montre les **réajustements périodiques** (les 'sauts') vers les poids cibles (visibles dans le tableau ci-dessous)."
    )
    
    fig_alloc = go.Figure()
    for ticker_alloc in weights_history.columns:
        fig_alloc.add_trace(go.Scatter(
            x=weights_history.index, 
            y=weights_history[ticker_alloc],
            mode='lines',
            stackgroup='one',
            name=ticker_alloc,
            hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{ticker_alloc}: %{{y:.1%}}<extra></extra>"
        ))
    fig_alloc.update_layout(template="plotly_dark", yaxis_title="Allocation (%)", xaxis_title="Date", hovermode="x unified", yaxis_tickformat=".0%", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st_plotly_chart(fig_alloc)
    
    if target_weights_log and len(target_weights_log) > 1:
        with st.expander("Historique des allocations cibles (rebalancements)", expanded=False):
            st.caption(
                "Ce tableau montre l'allocation cible exacte qui a été calculée et appliquée à chaque date de rebalancement. "
                "En mode 'Statique', ces poids sont toujours identiques. "
                "En mode 'Dynamique', ils changent à chaque rebalancement."
            )
            
            log_data = []
            for entry in target_weights_log:
                row = {"Date": entry["Date"].date()}
                for i, ticker in enumerate(tickers):
                    row[ticker] = f"{entry['Weights'][i]*100:.2f}%"
                log_data.append(row)
            
            log_df = pd.DataFrame(log_data).set_index("Date")
            st.dataframe(log_df, width='stretch')
    
    st.subheader("Analyse détaillée des Drawdowns")
    top_n_dd = st.number_input("Afficher le Top N des pires drawdowns", min_value=1, max_value=20, value=5, step=1, key="topn_dd_input", help="Classe les pires 'drawdowns' (pertes pic-à-creux) et affiche les N plus importants.")
    with st.spinner("Analyse des périodes de drawdown..."):
        dd_table = find_drawdowns(wealth, top_n=top_n_dd)
        if dd_table.empty:
            st.info("Aucun drawdown enregistré (performance toujours positive).")
        else:
            dd_table_fmt = dd_table.copy()
            dd_table_fmt["Max Drawdown"] = dd_table_fmt["Max Drawdown"].map(lambda x: f"{x*100:.2f}%")
            dd_table_fmt["Recovery Date"] = dd_table_fmt["Recovery Date"].fillna("En cours")
            dd_table_fmt["Total Duration (Days)"] = dd_table_fmt["Total Duration (Days)"].fillna("-")
            dd_table_fmt["Peak-to-Trough (Days)"] = dd_table_fmt["Peak-to-Trough (Days)"].astype(str)
            dd_table_fmt["Total Duration (Days)"] = dd_table_fmt["Total Duration (Days)"].astype(str)
            st.dataframe(dd_table_fmt, width='stretch', hide_index=True)
    
    st.subheader("Analyses additionnelles")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Volatilité glissante (annualisée)**")
        st.caption(f"Affiche la volatilité annualisée du portefeuille sur une fenêtre glissante de {k} périodes (environ 1 an de données {freq}).")
        roll_window = k 
        if len(ret_series) > roll_window:
            rolling_vol = ret_series.rolling(window=roll_window).std(ddof=1) * np.sqrt(k)
            fig_roll_vol = go.Figure()
            fig_roll_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode="lines", name="Volatilité glissante", line=dict(color="tomato")))
            fig_roll_vol.update_layout(template="plotly_dark", yaxis_title="Volatilité Ann.", xaxis_title="Date")
            fig_roll_vol.update_yaxes(tickformat=".0%")
            st_plotly_chart(fig_roll_vol)
        else:
            st.info(f"Pas assez de données pour une fenêtre glissante de {roll_window} périodes.")
    with c4:
        st.markdown("**Distribution des rendements mensuels**")
        st.caption("Histogramme montrant la distribution (fréquence) des rendements mensuels du portefeuille (nets de frais).")
        if len(monthly_ret) > 1:
            fig_hist_ret = go.Figure()
            fig_hist_ret.add_trace(go.Histogram(x=monthly_ret, name="Rendements Mensuels", nbinsx=30, histnorm='probability density', marker_color='cornflowerblue'))
            fig_hist_ret.update_layout(template="plotly_dark", yaxis_title="Densité", xaxis_title="Rendement Mensuel", barmode="overlay")
            fig_hist_ret.update_traces(opacity=0.75)
            fig_hist_ret.update_xaxes(tickformat=".1%")
            st_plotly_chart(fig_hist_ret)
        else:
            st.info("Pas assez de données pour l'histogramme mensuel.")
