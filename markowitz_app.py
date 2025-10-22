# =========================================================
# MARKOWITZ PORTFOLIO OPTIMIZER — fréquence auto + override manuel
# Ledoit–Wolf (si dispo), optimisation bornée, corrélations, backtest
# Frontière efficiente robuste (cible de rendement) + vues "Portfolio Visualizer"
# =========================================================
import warnings
warnings.filterwarnings(
    "ignore",
    message="Values in x were outside bounds during a minimize step, clipping to bounds",
    category=RuntimeWarning
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

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ---- helper rerun (compat old/new streamlit) ----
def _safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ============================ Sidebar ============================
st.sidebar.header("Paramètres")
uploaded_file = st.sidebar.file_uploader("Fichier Excel (.xlsx/.xls)", type=["xlsx", "xls"])

# RF toujours 0
RF = 0.0

# Méthodes d’estimation (defaults: Ledoit–Wolf + Moyenne simple)
cov_options = ["Échantillon", "Ledoit–Wolf"]
cov_default_index = (1 if HAS_LW else 0)
cov_method = st.sidebar.selectbox("Méthode de covariance", cov_options, index=cov_default_index)

# Bornes et Monte Carlo
min_w = st.sidebar.slider("Poids minimum par actif", 0.0, 1.0, 0.00, 0.01)
max_w = st.sidebar.slider("Poids maximum par actif", 0.0, 1.0, 1.00, 0.01)
n_mc  = st.sidebar.number_input("Nombre de portefeuilles Monte Carlo", value=1000, step=500)
seed  = st.sidebar.number_input("Graine aléatoire (Monte Carlo)", value=42, step=1)

# ============================ Utilitaires ============================
def infer_frequency(index: pd.DatetimeIndex) -> str:
    idx = index.sort_values().unique()
    if len(idx) < 3:
        return "monthly"
    deltas = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    med = np.median(deltas)
    if med <= 2:    return "daily"
    if med <= 10:   return "weekly"
    if med <= 40:   return "monthly"
    if med <= 120:  return "quarterly"
    return "yearly"

def ann_factor(freq: str) -> int:
    f = freq.lower()
    if f == "daily":     return 252
    if f == "weekly":    return 52
    if f == "monthly":   return 12
    if f == "quarterly": return 4
    if f == "yearly":    return 1
    raise ValueError("Fréquence non reconnue.")

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
    w, V = np.linalg.eigh(cov)
    w_clipped = np.clip(w, eps, None)
    return (V @ np.diag(w_clipped) @ V.T)

def ridge_regularize(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    n = cov.shape[0]
    tau = np.trace(cov) / max(1, n)
    return cov + ridge * float(tau) * np.eye(n)

def portfolio_perf(w, mu, cov):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol

def feasible_start(n: int, min_w: float, max_w: float) -> np.ndarray:
    if min_w < 0 or max_w <= 0 or min_w > max_w:
        raise ValueError("Bornes incohérentes pour les poids.")
    w = np.full(n, min_w, dtype=float)
    R = 1.0 - n * min_w
    if R < -1e-12:
        raise ValueError("Bornes infaisables : min_w * n > 1.")
    if R <= 1e-12:
        return w
    cap = np.maximum(0.0, max_w - w)
    while R > 1e-12:
        idx = np.where(cap > 1e-12)[0]
        if idx.size == 0: break
        inc = R / idx.size
        add = np.minimum(cap[idx], inc)
        w[idx] += add
        cap[idx] -= add
        R -= float(add.sum())
    np.clip(w, min_w, max_w, out=w)
    s = w.sum()
    if abs(s - 1.0) > 1e-10:
        w /= s
        np.clip(w, min_w, max_w, out=w)
    return w

def _solve_slsqp(objective, n, bounds, cons, w0=None, maxiter=2000, ftol=1e-10):
    if w0 is None:
        w0 = feasible_start(n, bounds[0][0], bounds[0][1])
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})
    if res.success:
        return res.x
    # second try from equal-weights
    w0b = np.ones(n)/n
    res2 = minimize(objective, w0b, method='SLSQP', bounds=bounds, constraints=cons,
                    options={'maxiter': maxiter*2, 'ftol': ftol*10, 'disp': False})
    if res2.success:
        return res2.x
    raise RuntimeError(f"Echec de l’optimisation : {res.message}")

def max_sharpe_weights(mu, cov, min_w, max_w):
    n = len(mu)
    bounds = [(min_w, max_w)] * n
    cons   = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0     = feasible_start(n, min_w, max_w)
    def obj(w):
        ret, vol = portfolio_perf(w, mu, cov)
        return 1e6 if vol <= 1e-12 else - (ret) / vol  # RF=0
    return _solve_slsqp(obj, n, bounds, cons, w0=w0)

def min_var_weights(mu, cov, min_w, max_w):
    n = len(mu)
    bounds = [(min_w, max_w)] * n
    cons   = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0     = feasible_start(n, min_w, max_w)
    return _solve_slsqp(lambda w: float(w @ cov @ w), n, bounds, cons, w0=w0)

def max_return_weights(mu, cov, min_w, max_w):
    n = len(mu)
    bounds = [(min_w, max_w)] * n
    cons   = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0     = feasible_start(n, min_w, max_w)
    return _solve_slsqp(lambda w: - float(w @ mu), n, bounds, cons, w0=w0)

# ---------- FRONTIÈRE EFFICIENTE (cible de rendement) ----------
def _min_var_for_target_return(mu, cov, r_target, min_w, max_w, w_start=None):
    n = len(mu)
    bounds = [(min_w, max_w)] * n
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
    w_mv = min_var_weights(mu, cov, min_w, max_w)
    r_mv, v_mv = portfolio_perf(w_mv, mu, cov)
    w_mr = max_return_weights(mu, cov, min_w, max_w)
    r_mr, _ = portfolio_perf(w_mr, mu, cov)

    r_grid = np.linspace(r_mv, r_mr, npts)
    pts = []
    w_prev = None
    for rt in r_grid:
        try:
            w_t = _min_var_for_target_return(mu, cov, rt, min_w, max_w, w_start=w_prev)
            r_t, v_t = portfolio_perf(w_t, mu, cov)
            pts.append((r_t, v_t))
            w_prev = w_t
        except Exception:
            continue

    if not pts:
        return np.empty((0, 2))
    pts = np.array(pts)
    order = np.argsort(pts[:, 1])
    pts = pts[order]
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
    if Rtot < -1e-12:
        raise ValueError("Bornes infaisables : min_w * n > 1.")
    if Rtot <= 1e-12:
        return np.tile(base, (N, 1))
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
            add += inc
            R -= float(inc.sum())
            free = cap - add
        W[k, :] = base + add
    W = np.clip(W, min_w, max_w)
    s = W.sum(axis=1, keepdims=True)
    W = W / s
    return W

def top_corr_pairs(corr: pd.DataFrame, k: int = 3):
    M = corr.copy()
    np.fill_diagonal(M.values, np.nan)
    pairs = []
    cols = M.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            c = M.iloc[i, j]
            if pd.notna(c):
                pairs.append((cols[i], cols[j], float(c)))
    if not pairs:
        return [], []
    pairs_sorted_pos = sorted(pairs, key=lambda x: x[2], reverse=True)
    pairs_sorted_neg = sorted(pairs, key=lambda x: x[2])
    return pairs_sorted_pos[:k], pairs_sorted_neg[:k]

# ---------- Backtest helpers ----------
def wealth_buy_hold(returns: pd.DataFrame, w0: np.ndarray) -> pd.Series:
    cum = (1 + returns).cumprod()
    wealth = (cum * w0).sum(axis=1)
    wealth.iloc[0] = 1.0
    return wealth

def wealth_rebalanced_every_n(returns: pd.DataFrame, w_target: np.ndarray, n_reb: int = 1) -> pd.Series:
    w_target = (w_target / w_target.sum()).astype(float)
    w_cur = w_target.copy()
    values = np.empty(len(returns), dtype=float)
    port_val = 1.0
    for t in range(len(returns)):
        r_vec = returns.iloc[t].values.astype(float)
        rp = float(w_cur @ r_vec)
        port_val *= (1 + rp)
        values[t] = port_val
        gross = w_cur * (1 + r_vec)
        denom = gross.sum()
        w_cur = gross / denom if denom > 0 else w_target.copy()
        if ((t + 1) % max(1, int(n_reb))) == 0:
            w_cur = w_target.copy()
    wealth = pd.Series(values, index=returns.index)
    wealth.iloc[0] = 1.0
    return wealth

def perf_metrics(wealth: pd.Series, freq_k: int):
    ret = wealth.pct_change().dropna()
    total_return = wealth.iloc[-1] - 1.0
    years = len(ret) / freq_k if freq_k > 0 else np.nan
    cagr = wealth.iloc[-1] ** (1/years) - 1 if years and years > 0 else np.nan
    vol_ann = ret.std(ddof=1) * np.sqrt(freq_k)
    sharpe = (ret.mean() * freq_k) / vol_ann if vol_ann > 0 else np.nan
    cummax = wealth.cummax()
    dd = wealth / cummax - 1.0
    max_dd = dd.min()
    worst = ret.min()
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Vol ann.": vol_ann,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Worst period": worst
    }, dd

def n_for_rebalance_choice(choice: str, data_freq: str, n_custom: int) -> int:
    if choice == "Auto (fréquence des données)":
        return 1
    if choice == "Mensuel":
        return months_to_periods(1, data_freq)
    if choice == "Trimestriel":
        return months_to_periods(3, data_freq)
    if choice == "Annuel":
        return months_to_periods(12, data_freq)
    if choice == "Tous les N périodes":
        return max(1, int(n_custom))
    return 1

def validate_weight_bounds(min_w: float, max_w: float, n: int, tol: float = 1e-12):
    problems = []
    hints = []
    if min_w < 0:
        problems.append("Le poids minimum est négatif.")
        hints.append("Fixe MIN_W ≥ 0.")
    if max_w <= 0:
        problems.append("Le poids maximum est nul ou négatif.")
        hints.append("Fixe MAX_W > 0.")
    if min_w > max_w:
        problems.append("MIN_W est supérieur à MAX_W.")
        hints.append("Assure-toi que MIN_W ≤ MAX_W.")
    if min_w * n > 1 + tol:
        problems.append(f"MIN_W×n = {min_w:.2%}×{n} = {min_w*n:.2%} > 100%.")
        hints.append(f"Baisse MIN_W ≤ {1/n:.2%} ou réduis le nombre d’actifs contraints.")
    if max_w * n < 1 - tol:
        problems.append(f"MAX_W×n = {max_w:.2%}×{n} = {max_w*n:.2%} < 100%.")
        hints.append(f"Augmente MAX_W ≥ {1/n:.2%} ou retire des actifs.")
    return problems, hints

def warn_tight_bounds(min_w: float, max_w: float, n: int, tol: float = 1e-12):
    slack_min = 1 - min_w * n
    slack_max = max_w * n - 1
    msgs = []
    if slack_min >= 0 and slack_min < 0.02 + tol:
        msgs.append(f"MIN_W×n proche de 100% (marge {slack_min:.2%}).")
    if slack_max >= 0 and slack_max < 0.02 + tol:
        msgs.append(f"MAX_W×n proche de 100% (marge {slack_max:.2%}).")
    if msgs:
        st.warning("Bornes très serrées : " + " ".join(msgs))

# ---- Top drawdowns helper (PV-like) ----
def top_drawdowns_table(wealth: pd.Series, k_annual: int, top_n: int = 10) -> pd.DataFrame:
    w = wealth.dropna().copy()
    cummax = w.cummax()
    dd = w / cummax - 1.0

    peaks, troughs, recovs, depths = [], [], [], []
    in_dd = False
    peak_val = None
    peak_date = None
    trough_val = None
    trough_date = None

    for t, val in w.items():
        if val >= (peak_val if peak_val is not None else -np.inf):
            if in_dd:
                recovs.append(t)
                peaks.append(peak_date)
                troughs.append(trough_date)
                depths.append((trough_val/peak_val) - 1.0)
                in_dd = False
            peak_val = val
            peak_date = t
            trough_val = val
            trough_date = t
        else:
            in_dd = True
            if val < trough_val:
                trough_val = val
                trough_date = t

    if in_dd and peak_val is not None:
        peaks.append(peak_date)
        troughs.append(trough_date)
        recovs.append(pd.NaT)
        depths.append((trough_val/peak_val) - 1.0)

    df = pd.DataFrame({
        "Peak": peaks,
        "Trough": troughs,
        "Recovery": recovs,
        "Depth (%)": [d*100 for d in depths],
    })
    df["Peak→Trough (days)"] = (pd.to_datetime(df["Trough"]) - pd.to_datetime(df["Peak"])).dt.days
    df["Trough→Recovery (days)"] = (pd.to_datetime(df["Recovery"]) - pd.to_datetime(df["Trough"])).dt.days
    df["Total Days"] = (pd.to_datetime(df["Recovery"]) - pd.to_datetime(df["Peak"])).dt.days

    df = df.sort_values("Depth (%)").head(top_n).reset_index(drop=True)
    return df

# ---- PV metrics helper ----
def pv_stats_from_returns(
    ret_native: pd.Series,
    monthly_ret: pd.Series,
    wealth: pd.Series,
    k_annual: int,
    alpha: float = 0.05
) -> pd.DataFrame:
    # --- Moyennes/vol mensuelles (style PV) ---
    m_arith_m = monthly_ret.mean()
    m_std_m   = monthly_ret.std(ddof=1)
    m_geo_m   = (1 + monthly_ret).prod()**(1/len(monthly_ret)) - 1 if len(monthly_ret) > 0 else np.nan

    # Annualisations depuis mensuel
    arith_ann = (1 + m_arith_m)**12 - 1 if pd.notna(m_arith_m) else np.nan
    geo_ann   = (1 + m_geo_m)**12 - 1 if pd.notna(m_geo_m) else np.nan
    std_ann   = m_std_m * np.sqrt(12) if pd.notna(m_std_m) else np.nan

    # Downside deviation (mensuelle) & Sortino (annualisé)
    downside = monthly_ret[monthly_ret < 0]
    downside_dev_m = np.sqrt(np.mean(np.square(downside))) if len(downside) else np.nan
    sortino = (m_arith_m*12) / (downside_dev_m*np.sqrt(12)) if (pd.notna(m_arith_m) and downside_dev_m and downside_dev_m>0) else np.nan

    # Sharpe annualisé (depuis la granularité native)
    vol_ann_native = ret_native.std(ddof=1) * np.sqrt(k_annual)
    sharpe_ann = (ret_native.mean() * k_annual) / vol_ann_native if (vol_ann_native and vol_ann_native>0) else np.nan

    # MaxDD et CALMAR = CAGR / |MaxDD|
    dd_series = wealth / wealth.cummax() - 1.0
    max_dd = dd_series.min()  # négatif
    # CAGR annualisé calculé proprement depuis la durée (en années) observée
    n_periods = ret_native.count()
    years = n_periods / k_annual if k_annual > 0 else np.nan
    cagr = (wealth.iloc[-1] ** (1/years) - 1) if (pd.notna(years) and years > 0) else np.nan
    calmar = (cagr / abs(max_dd)) if (pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0) else np.nan

    # Skew / Kurtosis (excess)
    skew  = ret_native.skew()
    ex_k  = ret_native.kurt()

    # VaR / CVaR (mensuels, pertes positives)
    if len(monthly_ret) > 0:
        hist_var = -np.percentile(monthly_ret.dropna(), alpha*100)
        mu_m, sd_m = monthly_ret.mean(), monthly_ret.std(ddof=1)
        z = -1.6448536269514729  # 5%
        anal_var = -(mu_m + z*sd_m)
        thr = np.percentile(monthly_ret.dropna(), alpha*100)
        cvar = -monthly_ret[monthly_ret <= thr].mean() if (monthly_ret[monthly_ret <= thr].size > 0) else np.nan
    else:
        hist_var = anal_var = cvar = np.nan

    # Positive periods + % et Gain/Loss ratio
    pos_n = int((ret_native > 0).sum())
    tot_n = int(ret_native.count())
    pos_pct = (pos_n / tot_n) if tot_n else np.nan
    pos_text = f"{pos_n} / {tot_n} ({pos_pct*100:.2f}%)" if tot_n else "N/A"

    avg_gain = ret_native[ret_native > 0].mean()
    avg_loss = -ret_native[ret_native < 0].mean() if (ret_native < 0).any() else np.nan
    gl_ratio = (avg_gain / avg_loss) if (avg_loss and avg_loss>0) else np.nan

    # SWR & PWR (approximations robustes)
    pwr = geo_ann
    if pd.notna(geo_ann) and pd.notna(std_ann):
        mu_log = np.log1p(geo_ann)
        swr = np.expm1(mu_log - 0.5 * (std_ann**2))
    else:
        swr = np.nan

    rows = [
        ("Arithmetic Mean (monthly)",     m_arith_m),
        ("Arithmetic Mean (annualized)",  arith_ann),
        ("Geometric Mean (monthly)",      m_geo_m),
        ("Geometric Mean (annualized)",   geo_ann),
        ("Standard Deviation (monthly)",  m_std_m),
        ("Standard Deviation (annualized)", std_ann),
        ("Downside Deviation (monthly)",  downside_dev_m),
        ("Maximum Drawdown",              max_dd),
        ("Sharpe Ratio",                  sharpe_ann),
        ("Sortino Ratio",                 sortino),
        ("Calmar Ratio",                  calmar),
        ("Skewness",                      skew),
        ("Excess Kurtosis",               ex_k),
        ("Historical Value-at-Risk (5%)", hist_var),
        ("Analytical Value-at-Risk (5%)", anal_var),
        ("Conditional Value-at-Risk (5%)", cvar),
        ("Safe Withdrawal Rate",          swr),
        ("Perpetual Withdrawal Rate",     pwr),
        ("Positive Periods",              pos_text),
        ("Gain/Loss Ratio",               gl_ratio),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")

    # Formatage pour affichage
    pct_like = [
        "Arithmetic Mean (monthly)", "Arithmetic Mean (annualized)",
        "Geometric Mean (monthly)", "Geometric Mean (annualized)",
        "Standard Deviation (monthly)", "Standard Deviation (annualized)",
        "Downside Deviation (monthly)", "Maximum Drawdown",
        "Historical Value-at-Risk (5%)", "Analytical Value-at-Risk (5%)",
        "Conditional Value-at-Risk (5%)",
        "Safe Withdrawal Rate", "Perpetual Withdrawal Rate",
    ]
    for m in pct_like:
        if m in df.index:
            v = df.loc[m, "Value"]
            df.loc[m, "Value"] = f"{float(v)*100:.2f}%" if pd.notna(v) else "N/A"

    for m in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Skewness", "Excess Kurtosis", "Gain/Loss Ratio"]:
        if m in df.index:
            v = df.loc[m, "Value"]
            df.loc[m, "Value"] = f"{float(v):.2f}" if pd.notna(v) else "N/A"

    # Positive Periods est déjà sous forme texte "79 / 117 (67.52%)" — ne pas reformater
    return df


# ============================ Cache ============================
@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index().dropna(axis=1, how="all")
    return df

@st.cache_data(show_spinner=False)
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices_to_returns(prices)

# ============================ App ============================
st.title("Optimisation de portefeuille (Markowitz)")

if not uploaded_file:
    st.info("Chargez un fichier Excel avec la date en première colonne et les tickers dans les colonnes suivantes.")
    st.stop()

prices_all = load_excel(uploaded_file)
all_tickers = prices_all.columns.tolist()
if prices_all.empty or len(all_tickers) < 2:
    st.error("Le fichier doit contenir au moins deux colonnes de prix (deux actifs).")
    st.stop()

# État persistant des actifs exclus
if "excluded" not in st.session_state:
    st.session_state["excluded"] = []

# Période
min_date = prices_all.index.min().date()
max_date = prices_all.index.max().date()
with st.expander("Périmètre temporel", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Date de début", value=min_date, min_value=min_date, max_value=max_date)
    with c2:
        end_date   = st.date_input("Date de fin", value=max_date, min_value=min_date, max_value=max_date)
    if start_date > end_date:
        st.error("La date de début doit être antérieure ou égale à la date de fin.")
        st.stop()

# Filtre dates + exclusions
mask = (prices_all.index.date >= start_date) & (prices_all.index.date <= end_date)
prices = prices_all.loc[mask].copy()
excluded = [t for t in st.session_state["excluded"] if t in prices.columns]
if excluded:
    prices = prices.drop(columns=excluded, errors="ignore")

tickers = prices.columns.tolist()
if prices.shape[0] < 3:
    st.error("Trop peu d’observations après filtrage. Élargissez la période.")
    st.stop()
if len(tickers) < 2:
    st.error("Il faut au moins deux actifs après exclusions. Réintègre des actifs dans l’onglet Données.")
    st.stop()

# --------- Détection auto + override manuel de la fréquence ---------
auto_freq = infer_frequency(prices.index)
freq_options = ["daily", "weekly", "monthly", "quarterly", "yearly"]
freq = st.selectbox(
    "Fréquence des données (détection automatique modifiable)",
    options=freq_options,
    index=freq_options.index(auto_freq),
    help="La fréquence détectée peut être corrigée ici si nécessaire."
)
k = ann_factor(freq)

st.caption(f"Fréquence sélectionnée : {freq} | Annualisation : ×{k}")
st.caption(f"Période utilisée : {prices.index.min().date()} → {prices.index.max().date()} | Observations : {len(prices)}")
if excluded:
    st.caption(f"Actifs exclus actuellement : {', '.join(excluded)}")

# Validation bornes
n = len(tickers)
problems, hints = validate_weight_bounds(min_w, max_w, n)
if problems:
    st.error("**Bornes de poids infaisables.**\n\n" + "• " + "\n• ".join(problems) +
             ("\n\n**Comment corriger :**\n" + "• " + "\n• ".join(hints) if hints else ""))
    st.stop()
else:
    warn_tight_bounds(min_w, max_w, n)

# ---------------- Tabs ----------------
tab_data, tab_opt, tab_fig, tab_corr, tab_bt = st.tabs(["Données", "Optimisation", "Graphiques", "Corrélation", "Backtest"])

# ============================ Onglet Données ============================
with tab_data:
    st.subheader("Tableau de prix utilisé (après filtre de dates et exclusions)")
    st.dataframe(prices, use_container_width=True)

    st.markdown("---")
    st.subheader("Inclure / Exclure des actifs")

    current_all = prices_all.columns.tolist()
    excluded_all = st.session_state["excluded"]
    available_all = [t for t in current_all if t not in excluded_all]

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Exclure des actifs**")
        to_exclude = st.multiselect("Sélectionne pour exclure", options=available_all, key="to_exclude_list")
        if st.button("Exclure", type="primary", use_container_width=True, key="btn_exclure"):
            new_excluded = sorted(list(set(excluded_all + to_exclude)))
            st.session_state["excluded"] = new_excluded
            st.success(f"Exclus : {', '.join(to_exclude)}")
            _safe_rerun()
    with colB:
        st.markdown("**Réintégrer des actifs**")
        to_include = st.multiselect("Sélectionne pour réintégrer", options=excluded_all, key="to_include_list")
        if st.button("Réintégrer", use_container_width=True, key="btn_reintegrer"):
            new_excluded = [t for t in excluded_all if t not in to_include]
            st.session_state["excluded"] = new_excluded
            st.success(f"Réintégrés : {', '.join(to_include)}")
            _safe_rerun()

    st.caption(
        f"Actifs totaux : {len(current_all)} | "
        f"Exclus : {len(st.session_state['excluded'])} | "
        f"Utilisés : {len(tickers)}"
    )

# ============================ Onglet Optimisation ============================
with tab_opt:
    returns = compute_returns(prices)
    mu = returns.mean().values * k

    if cov_method == "Ledoit–Wolf":
        if HAS_LW:
            lw = LedoitWolf().fit(returns.values); cov = lw.covariance_ * k
        else:
            st.warning("Ledoit–Wolf indisponible (scikit-learn manquant). Utilisation de la covariance d’échantillon.")
            cov = returns.cov().values * k
    else:
        cov = returns.cov().values * k

    cov = near_psd_clip(cov)
    cov = ridge_regularize(cov, ridge=1e-6)

    vol = np.sqrt(np.diag(cov))
    shp = np.where(vol > 0, (mu)/vol, np.nan)  # RF=0

    st.subheader("Rendements, volatilités et Sharpe (annualisés)")
    df_metrics = pd.DataFrame({"Return_ann": mu, "Vol_ann": vol, "Sharpe_ann": shp}, index=tickers)
    df_metrics_fmt = df_metrics.copy()
    for c in ["Return_ann", "Vol_ann"]:
        df_metrics_fmt[c] = (df_metrics_fmt[c]*100).map(lambda v: f"{v:.2f}%")
    df_metrics_fmt["Sharpe_ann"] = df_metrics_fmt["Sharpe_ann"].map(lambda v: f"{v:.2f}")
    st.dataframe(df_metrics_fmt, use_container_width=True)

    # Portefeuilles optimaux
    try:
        w_ms = max_sharpe_weights(mu, cov, min_w, max_w)
    except RuntimeError:
        cov = ridge_regularize(cov, ridge=1e-4); w_ms = max_sharpe_weights(mu, cov, min_w, max_w)
    try:
        w_mv = min_var_weights(mu, cov, min_w, max_w)
    except RuntimeError:
        cov = ridge_regularize(cov, ridge=1e-4); w_mv = min_var_weights(mu, cov, min_w, max_w)
    try:
        w_mr = max_return_weights(mu, cov, min_w, max_w)
    except RuntimeError:
        w_mr = feasible_start(len(tickers), min_w, max_w)

    def metrics_from_w(name, w):
        ret, vol_ = portfolio_perf(w, mu, cov)
        sharpe = (ret)/vol_
        return {"Portefeuille": name, "Return": ret, "Vol": vol_, "Sharpe": sharpe}

    df_res = pd.DataFrame([
        metrics_from_w("Max Sharpe", w_ms),
        metrics_from_w("Min Variance", w_mv),
        metrics_from_w("Max Return", w_mr)
    ]).set_index("Portefeuille")

    df_res_fmt = df_res.copy()
    for c in ["Return", "Vol"]:
        df_res_fmt[c] = (df_res_fmt[c]*100).map(lambda v: f"{v:.2f}%")
    df_res_fmt["Sharpe"] = df_res_fmt["Sharpe"].map(lambda v: f"{v:.2f}")

    st.subheader("Résultats de l’optimisation")
    st.dataframe(df_res_fmt, use_container_width=True)

    st.subheader("Contributions au risque (somme = 100%)")
    df_rc = pd.DataFrame({
        "Max Sharpe": risk_contrib(w_ms, cov),
        "Min Variance": risk_contrib(w_mv, cov),
        "Max Return": risk_contrib(w_mr, cov)
    }, index=tickers)
    st.dataframe((df_rc*100).round(2).astype(str) + " %", use_container_width=True)

# ============================ Onglet Graphiques ============================
with tab_fig:
    returns = compute_returns(prices)
    mu = returns.mean().values * k
    if cov_method == "Ledoit–Wolf":
        if HAS_LW: lw = LedoitWolf().fit(returns.values); cov = lw.covariance_ * k
        else:      cov = returns.cov().values * k
    else:
        cov = returns.cov().values * k
    cov = near_psd_clip(cov); cov = ridge_regularize(cov, ridge=1e-6)

    n = len(tickers)
    try:
        w_ms = max_sharpe_weights(mu, cov, min_w, max_w)
    except RuntimeError:
        cov = ridge_regularize(cov, ridge=1e-4); w_ms = max_sharpe_weights(mu, cov, min_w, max_w)
    try:
        w_mv = min_var_weights(mu, cov, min_w, max_w)
    except RuntimeError:
        cov = ridge_regularize(cov, ridge=1e-4); w_mv = min_var_weights(mu, cov, min_w, max_w)
    try:
        w_mr = max_return_weights(mu, cov, min_w, max_w)
    except RuntimeError:
        w_mr = feasible_start(n, min_w, max_w)

    st.subheader("Nuage de portefeuilles, frontière efficiente (bornée) et portefeuilles optimaux")
    rng = np.random.default_rng(int(seed))
    W = sample_bounded_simplex(n=n, N=int(n_mc), min_w=min_w, max_w=max_w, rng=rng)
    rets_mc = W @ mu
    vols_mc = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))

    front = efficient_frontier(mu, cov, min_w, max_w, npts=160)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols_mc, y=rets_mc, mode="markers",
        marker=dict(size=4, opacity=0.35),
        name="Portefeuilles aléatoires (bornés)",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
    ))
    if front.size > 0:
        fig.add_trace(go.Scatter(
            x=front[:,1], y=front[:,0], mode="lines",
            name="Frontière efficiente",
            line=dict(width=3),
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
        ))
    for name, w, sym in [("Max Sharpe", w_ms, "star"), ("Min Variance", w_mv, "circle"), ("Max Return", w_mr, "square")]:
        r, v = portfolio_perf(w, mu, cov)
        fig.add_trace(go.Scatter(
            x=[v], y=[r], mode="markers", name=name,
            marker=dict(size=12, symbol=sym),
            hovertemplate=f"{name}<br>Vol: %{{x:.2%}}<br>Ret: %{{y:.2%}}<extra></extra>"
        ))
    fig.update_layout(
        xaxis_title="Volatilité (ann.)",
        yaxis_title="Rendement (ann.)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest"
    )
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, key="fig_frontier")

    # Camemberts
    def pie_series(weights: np.ndarray, labels: list, threshold: float = 0.02, eps: float = 1e-6) -> pd.Series:
        s = pd.Series(weights, index=labels)
        s = s[s > eps]
        if s.empty: return pd.Series([1.0], index=["N/A"])
        s = s.sort_values(ascending=False)
        if threshold and threshold > 0:
            majors = s[s >= threshold]
            others = s[s < threshold].sum()
            if others > eps:
                majors.loc["Others"] = others
            s = majors
        s = s / s.sum()
        return s

    s_ms = pie_series(w_ms, tickers, threshold=0.02)
    s_mv = pie_series(w_mv, tickers, threshold=0.02)
    s_mr = pie_series(w_mr, tickers, threshold=0.02)

    pies = make_subplots(rows=1, cols=3, specs=[[{"type": "domain"}]*3],
                         subplot_titles=("Max Sharpe", "Min Variance", "Max Return"))

    def add_pie(fig_, r, c, s: pd.Series):
        if len(s) == 1:
            fig_.add_trace(go.Pie(labels=[s.index[0]], values=[1.0], hole=0.35,
                                  sort=False, textinfo="label+percent", textposition="inside", showlegend=False), r, c)
        else:
            fig_.add_trace(go.Pie(labels=s.index.tolist(), values=s.values.tolist(), hole=0.35,
                                  sort=False, textinfo="percent+label"), r, c)

    add_pie(pies, 1, 1, s_ms)
    add_pie(pies, 1, 2, s_mv)
    add_pie(pies, 1, 3, s_mr)
    pies.update_layout(template="plotly_white")
    st.plotly_chart(pies, use_container_width=True, key="fig_pies")

    st.subheader("Poids par portefeuille (%)")
    weights_df = pd.DataFrame({"Max Sharpe": w_ms, "Min Variance": w_mv, "Max Return": w_mr}, index=tickers)
    st.dataframe((weights_df * 100).round(2).astype(str) + " %", use_container_width=True)

# ============================ Onglet Corrélation ============================
with tab_corr:
    returns = compute_returns(prices)
    st.subheader("Paramètres de corrélation")
    c1, c2, c3 = st.columns(3)
    with c1:
        corr_type = st.selectbox("Type de corrélation", ["Pearson", "Spearman"], index=0)
    with c2:
        months_win = st.number_input("Fenêtre (mois) pour la matrice/rolling", value=24, step=1, min_value=3, max_value=240)
    with c3:
        freq_used = freq

    win = months_to_periods(int(months_win), freq_used)
    st.caption(f"Fenêtre utilisée : {win} périodes ({months_win} mois, fréquence {freq_used})")

    corr_data = returns.iloc[-win:] if len(returns) >= win else returns
    corr = corr_data.corr(method="spearman" if corr_type=="Spearman" else "pearson")
    ordered = cluster_order_from_corr(corr); corr_ord = corr.loc[ordered, ordered]

    st.subheader("Matrice de corrélation (ordre clusterisé)")
    heat = go.Figure(data=go.Heatmap(
        z=corr_ord.values, x=corr_ord.columns, y=corr_ord.index,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0, colorbar=dict(title="Corr")
    ))
    heat.update_layout(template="plotly_white", height=600)
    st.plotly_chart(heat, use_container_width=True, key="fig_corr_heat")

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
        figc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr.values, mode="lines",
                                  name=f"Corr({asset_a},{asset_b})"))
        figc.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        figc.update_layout(template="plotly_white", yaxis_title="Corrélation roulante", xaxis_title="Date")
        figc.update_yaxes(range=[-1, 1])
        st.plotly_chart(figc, use_container_width=True, key="fig_corr_roll")
    else:
        st.info("Sélectionnez deux actifs distincts.")

    st.subheader("Corrélation de paniers (égal-pondérés)")
    c1, c2 = st.columns(2)
    with c1:
        basket_a = st.multiselect("Panier A — sélectionner des actifs", tickers, default=[], key="basket_a")
    with c2:
        basket_b = st.multiselect("Panier B — sélectionner des actifs", tickers, default=[], key="basket_b")
    if len(basket_a) >= 1 and len(basket_b) >= 1:
        Ra = returns[basket_a].mean(axis=1)
        Rb = returns[basket_b].mean(axis=1)
        if corr_type == "Spearman":
            corr_baskets = Ra.iloc[-win:].rank().corr(Rb.iloc[-win:].rank()) if len(Ra) >= win else Ra.rank().corr(Rb.rank())
        else:
            corr_baskets = Ra.iloc[-win:].corr(Rb.iloc[-win:]) if len(Ra) >= win else Ra.corr(Rb)
        st.write(f"Corrélation panier A vs panier B (fenêtre courante) : {corr_baskets:.2f}")
        if len(returns) >= win:
            roll_corr_baskets = Ra.rolling(win).corr(Rb)
            figb = go.Figure()
            figb.add_trace(go.Scatter(x=roll_corr_baskets.index, y=roll_corr_baskets.values, mode="lines",
                                      name="Corr(A,B) rolling"))
            figb.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
            figb.update_layout(template="plotly_white", yaxis_title="Corrélation roulante paniers", xaxis_title="Date")
            figb.update_yaxes(range=[-1, 1])
            st.plotly_chart(figb, use_container_width=True, key="fig_corr_baskets")
    else:
        st.info("Choisissez au moins un actif dans chaque panier.")

# ============================ Onglet Backtest ============================
with tab_bt:
    st.subheader("Backtest à partir de poids choisis")

    returns = compute_returns(prices)
    n = len(tickers)

    src = st.radio("Source des poids", ["Manuel", "Égal-pondéré", "Max Sharpe", "Min Variance", "Max Return"],
                   horizontal=True, index=2)

    mu_bt = returns.mean().values * k
    if cov_method == "Ledoit–Wolf":
        if HAS_LW:
            lw = LedoitWolf().fit(returns.values); cov_bt = lw.covariance_ * k
        else:
            cov_bt = returns.cov().values * k
    else:
        cov_bt = returns.cov().values * k
    cov_bt = near_psd_clip(cov_bt); cov_bt = ridge_regularize(cov_bt, ridge=1e-6)

    if src == "Égal-pondéré":
        w_init = np.ones(n) / n
    elif src == "Max Sharpe":
        try:
            w_init = max_sharpe_weights(mu_bt, cov_bt, min_w, max_w)
        except RuntimeError:
            cov_bt = ridge_regularize(cov_bt, ridge=1e-4); w_init = max_sharpe_weights(mu_bt, cov_bt, min_w, max_w)
    elif src == "Min Variance":
        try:
            w_init = min_var_weights(mu_bt, cov_bt, min_w, max_w)
        except RuntimeError:
            cov_bt = ridge_regularize(cov_bt, ridge=1e-4); w_init = min_var_weights(mu_bt, cov_bt, min_w, max_w)
    elif src == "Max Return":
        try:
            w_init = max_return_weights(mu_bt, cov_bt, min_w, max_w)
        except RuntimeError:
            w_init = feasible_start(n, min_w, max_w)
    else:
        w_init = np.ones(n) / n

    # Éditeur de poids
    df_edit = pd.DataFrame({"Ticker": tickers, "Poids_%": (w_init * 100)})
    edited = st.data_editor(
        df_edit,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Poids_%": st.column_config.NumberColumn("Poids (%)", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f")
        }
    )
    w_user_pct = np.asarray(edited["Poids_%"].values, dtype=float)
    total_pct = float(np.nansum(w_user_pct))
    if not np.isfinite(total_pct):
        st.error("La somme des poids est invalide (NaN/inf). Corrige les entrées."); st.stop()
    if total_pct <= 0:
        st.error("La somme des poids est nulle. Ajuste les valeurs pour dépasser 0%."); st.stop()
    if total_pct > 100.0 + 1e-9:
        excess = total_pct - 100.0
        st.error(f"La somme des poids ({total_pct:.2f}%) dépasse 100% de {excess:.2f}%. Réduis les poids."); st.stop()
    if total_pct < 100.0 - 1e-9:
        st.info(f"La somme des poids est {total_pct:.2f}%. Ils seront normalisés à 100% pour le backtest.")
    w_user = np.clip(w_user_pct, 0, None) / total_pct
    st.caption(f"Somme des poids saisie : {total_pct:.2f}% → utilisée après normalisation : 100.00%")

    # Choix du mode + fréquence de rebalancement
    mode = st.radio("Mode de backtest", ["Buy & Hold (sans rebalancement)", "Rebalancement"],
                    horizontal=True, index=1)
    if mode == "Rebalancement":
        col_rb1, col_rb2 = st.columns([2,1])
        with col_rb1:
            reb_choice = st.selectbox(
                "Fréquence de rebalancement",
                ["Auto (fréquence des données)", "Mensuel", "Trimestriel", "Annuel", "Tous les N périodes"],
                index=0
            )
        with col_rb2:
            n_custom = st.number_input("N périodes (si 'Tous les N périodes')", value=1, min_value=1, step=1,
                                       disabled=(reb_choice != "Tous les N périodes"))
        n_reb = n_for_rebalance_choice(reb_choice, freq, n_custom)
        st.caption(f"Rebalancement tous les {n_reb} pas de temps.")
    else:
        n_reb = None

    wealth = wealth_buy_hold(returns, w_user) if mode.startswith("Buy") else wealth_rebalanced_every_n(returns, w_user, n_reb=n_reb)
    metrics, dd = perf_metrics(wealth, freq_k=k)

    # ====== Contrôles d'affichage type PV ======
    cA, cB, cC, cD = st.columns(4)
    with cA:
        log_scale = st.checkbox("Échelle log (Wealth)", value=False)
    with cB:
        roll_months = st.number_input("Rolling window (mois)", value=12, min_value=3, max_value=120, step=3)
    with cC:
        hist_bins = st.number_input("Histogram bins", value=50, min_value=10, max_value=200, step=10)
    with cD:
        show_drawdowns = st.checkbox("Montrer le tableau des top drawdowns", value=True)

    n_roll = months_to_periods(int(roll_months), freq)

    # Rendements à la granularité native
    ret = wealth.pct_change().dropna()

    # Rendements mensuels & annuels
    monthly_ret = (1 + ret).resample("ME").prod() - 1
    annual_ret  = (1 + monthly_ret).resample("YE").prod() - 1

    # ====== Métriques du backtest simple ======
    st.subheader("Métriques du backtest")
    mdf = pd.DataFrame({
        "Total Return": [f"{metrics['Total Return']*100:.2f}%"],
        "CAGR": [f"{metrics['CAGR']*100:.2f}%"] if pd.notna(metrics["CAGR"]) else ["N/A"],
        "Vol ann.": [f"{metrics['Vol ann.']*100:.2f}%"] if pd.notna(metrics["Vol ann."]) else ["N/A"],
        "Sharpe": [f"{metrics['Sharpe']:.2f}"] if pd.notna(metrics["Sharpe"]) else ["N/A"],
        "Max Drawdown": [f"{metrics['Max Drawdown']*100:.2f}%"] if pd.notna(metrics["Max Drawdown"]) else ["N/A"],
        "Worst period": [f"{metrics['Worst period']*100:.2f}%"] if pd.notna(metrics["Worst period"]) else ["N/A"]
    })
    st.dataframe(mdf, use_container_width=True)

    # ====== Risk & Return Metrics (style PV) ======
    st.subheader("Risk & Return Metrics")
    pv_df = pv_stats_from_returns(ret_native=ret, monthly_ret=monthly_ret, wealth=wealth, k_annual=k)
    st.dataframe(pv_df, use_container_width=True)

    # ====== Courbes principales (Wealth + Drawdown) ======
    st.subheader("Courbes de backtest")
    col1, col2 = st.columns(2)
    with col1:
        wealth_plot = wealth.where(wealth > 0, np.nan) if log_scale else wealth
        figw = go.Figure()
        figw.add_trace(go.Scatter(x=wealth_plot.index, y=wealth_plot.values, mode="lines",
                                  name="Wealth (base=1.0)"))
        figw.update_layout(template="plotly_white", yaxis_title="Valeur du portefeuille", xaxis_title="Date")
        if log_scale:
            figw.update_yaxes(type="log")
        st.plotly_chart(figw, use_container_width=True, key="fig_bt_wealth")
    with col2:
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
        figd.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        figd.update_layout(template="plotly_white", yaxis_title="Drawdown", xaxis_title="Date")
        figd.update_yaxes(tickformat=".0%")
        st.plotly_chart(figd, use_container_width=True, key="fig_bt_dd")

    # ====== Analyses complémentaires (style Portfolio Visualizer) ======
    st.subheader("Analyses complémentaires")

    # === Rendements annuels : vue combinée (switch) ===
    st.subheader("Rendements annuels — vue combinée")
    annual_view = st.radio(
        "Vue :", 
        ["Portefeuille", "Par actif"],
        horizontal=True,
        index=0,
        key="annual_returns_view"
    )

    # (re)calcule les jeux nécessaires
    # annual_ret (portefeuille) existe déjà plus haut, on le sécurise :
    annual_ret_port = annual_ret.copy()

    # rendements annuels par actif
    asset_ret = prices.pct_change().dropna()
    asset_ret_y = pd.DataFrame()
    if not asset_ret.empty:
        asset_ret_m = (1 + asset_ret).resample("ME").prod() - 1
        asset_ret_y = (1 + asset_ret_m).resample("YE").prod() - 1  # DataFrame [années x actifs]

    fig_annual_combo = go.Figure()

    if annual_view == "Portefeuille":
        if not annual_ret_port.empty:
            y = annual_ret_port.index.year.astype(int)
            fig_annual_combo.add_trace(go.Bar(x=y, y=annual_ret_port.values, name="Portefeuille"))
            fig_annual_combo.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
            fig_annual_combo.update_layout(
                template="plotly_white",
                xaxis_title="Année",
                yaxis_title="Rendement annuel",
                title="Rendements annuels du portefeuille"
            )
            fig_annual_combo.update_yaxes(tickformat=".0%")
        else:
            st.info("Pas assez d'historique pour le rendement annuel du portefeuille.")
    else:  # "Par actif"
        if not asset_ret_y.empty:
            years = asset_ret_y.index.year.astype(int)
            for col in asset_ret_y.columns:
                fig_annual_combo.add_trace(go.Bar(x=years, y=asset_ret_y[col].values, name=col))
            fig_annual_combo.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
            fig_annual_combo.update_layout(
                template="plotly_white",
                barmode="group",  # groupé par année
                xaxis_title="Year",
                yaxis_title="Return",
                title="Annual Returns of Portfolio Assets",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_annual_combo.update_yaxes(tickformat=".0%")
        else:
            st.info("Séries d’actifs insuffisantes pour les rendements annuels par actif.")

    st.plotly_chart(fig_annual_combo, use_container_width=True, key="fig_bt_annual_combo")


    # 2) Histogramme des rendements (granularité native)
    fig_hist = go.Figure()
    if not ret.empty:
        fig_hist.add_trace(go.Histogram(x=ret.values, nbinsx=int(hist_bins), name="Rendements"))
        fig_hist.add_vline(x=0.0, line_width=1, line_dash="dash", line_color="gray")
        fig_hist.add_vline(x=float(ret.mean()), line_width=1, line_dash="dot", line_color="black")
        fig_hist.update_layout(template="plotly_white", xaxis_title="Rendement par période",
                               yaxis_title="Fréquence", title="Distribution des rendements")
        fig_hist.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_hist, use_container_width=True, key="fig_bt_hist")

    # 3) Rolling CAGR / Vol / Sharpe
    colr1, colr2, colr3 = st.columns(3)
    rolling_cagr = pd.Series(np.nan, index=wealth.index)
    rolling_cagr = pd.Series(np.nan, index=wealth.index)
    if len(wealth) > n_roll:
        base = wealth.shift(n_roll).replace(0, np.nan)
        ratio = (wealth / base)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        rolling_cagr = ratio**(k / n_roll) - 1

    rolling_vol = ret.rolling(n_roll).std(ddof=1) * np.sqrt(k)
    rolling_mean = ret.rolling(n_roll).mean()
    rolling_sharpe = (rolling_mean * k) / rolling_vol.replace(0, np.nan)

    with colr1:
        fig_rcagr = go.Figure()
        fig_rcagr.add_trace(go.Scatter(x=rolling_cagr.index, y=rolling_cagr.values, mode="lines", name="Rolling CAGR"))
        fig_rcagr.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        fig_rcagr.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="CAGR (annualisé)",
                                title=f"Rolling CAGR — fenêtre {roll_months} mois")
        fig_rcagr.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_rcagr, use_container_width=True, key="fig_bt_roll_cagr")
    with colr2:
        fig_rvol = go.Figure()
        fig_rvol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode="lines", name="Rolling Vol"))
        fig_rvol.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Vol (annualisée)",
                               title=f"Rolling Vol — fenêtre {roll_months} mois")
        fig_rvol.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_rvol, use_container_width=True, key="fig_bt_roll_vol")
    with colr3:
        fig_rsh = go.Figure()
        fig_rsh.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode="lines", name="Rolling Sharpe"))
        fig_rsh.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        fig_rsh.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Sharpe",
                              title=f"Rolling Sharpe — fenêtre {roll_months} mois")
        st.plotly_chart(fig_rsh, use_container_width=True, key="fig_bt_roll_sharpe")

    # 4) Calendar heatmap (mensuel)
    st.subheader("Calendar Heatmap — rendements mensuels")
    if not monthly_ret.empty:
        cal = monthly_ret.to_frame("ret").copy()
        cal["Year"] = cal.index.year
        cal["Month"] = cal.index.month
        pivot = cal.pivot(index="Year", columns="Month", values="ret").sort_index()
        pivot = pivot.reindex(columns=range(1,13))

        # bornes symétriques robustes
        zabs = np.nanmax(np.abs(pivot.values))
        zabs = float(zabs) if np.isfinite(zabs) and zabs > 0 else 0.01

        heatmap = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"],
            y=pivot.index.astype(int),
            colorscale="RdBu",
            zmin=-zabs, zmax=zabs, zmid=0,
            colorbar=dict(title="Mensuel"),
            hovertemplate="Année %{y}<br>%{x}: %{z:.2%}<extra></extra>"
        ))
    
        heatmap.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Année")
        st.plotly_chart(heatmap, use_container_width=True, key="fig_bt_cal_heat")

    

    # 6) Tableaux (mensuel + top drawdowns)
    # 6) Top drawdowns uniquement
    if show_drawdowns:
        st.subheader("Top drawdowns")
        dd_table = top_drawdowns_table(wealth, k_annual=k, top_n=10)
        if not dd_table.empty:
            fmt = dd_table.copy()
            fmt["Depth (%)"] = fmt["Depth (%)"].map(lambda x: f"{x:.2f}%")
            st.dataframe(fmt, use_container_width=True)
        else:
            st.info("Pas de drawdown identifié.")

