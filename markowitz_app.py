# =========================================================
# MARKOWITZ PORTFOLIO OPTIMIZER — Version allégée (bornes robustes + onglet Données)
# Defaults: Ledoit–Wolf + moyenne simple, RF=0
# Frontière τ, MC borné, Corrélations + Baskets, Backtest cohérent
# Onglet "Données" : affichage du tableau + exclusion/réintégration d'actifs (persistant)
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

# ============================ Sidebar ============================
st.sidebar.header("Paramètres")
uploaded_file = st.sidebar.file_uploader("Fichier Excel (.xlsx/.xls)", type=["xlsx", "xls"])

# RF toujours 0
RF = 0.0

# Méthodes d'estimation (defaults: Ledoit–Wolf + Moyenne simple)
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
    if med <= 2:   return "daily"
    if med <= 10:  return "weekly"
    if med <= 40:  return "monthly"
    if med <= 120: return "quarterly"
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
    vol = float(np.sqrt(w @ cov @ w))
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
        if idx.size == 0:
            break
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

def _solve_slsqp(objective, n, bounds, cons, w0=None, maxiter=2000):
    if w0 is None:
        w0 = feasible_start(n, bounds[0][0], bounds[0][1])
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False})
    if res.success:
        return res.x
    # re-try avec autre w0 et tolérance plus souple
    w0b = np.ones(n)/n
    res2 = minimize(objective, w0b, method='SLSQP', bounds=bounds, constraints=cons,
                    options={'maxiter': maxiter*2, 'ftol': 1e-9, 'disp': False})
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
        return 1e6 if vol <= 1e-12 else - (ret - RF) / vol  # RF=0
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

# ---- Frontière par compromis risque/rendement (τ) ----
def frontier_by_tradeoff(mu, cov, min_w, max_w, npts=120, tau_max=None):
    n = len(mu)
    bounds = [(min_w, max_w)] * n
    cons   = [{'type':'eq','fun': lambda w: np.sum(w) - 1.0}]
    if tau_max is None:
        num = np.linalg.norm(mu)
        den = np.linalg.norm(cov)
        tau_max = 50.0 * (num / (den + 1e-12))
    taus = np.linspace(0.0, tau_max, npts)
    pts = []
    w0 = feasible_start(n, min_w, max_w)
    for t in taus:
        def obj(w, tau=t):
            return float(w @ cov @ w) - tau * float(w @ mu)
        try:
            w = _solve_slsqp(obj, n, bounds, cons, w0=w0)
            r, v = portfolio_perf(w, mu, cov)
            pts.append((r, v))
            w0 = w
        except Exception:
            continue
    if not pts:
        return np.empty((0,2))
    pts = np.array(pts)
    order = np.argsort(pts[:,0])
    return pts[order]

def risk_contrib(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    tot_var = float(w @ cov @ w)
    if tot_var <= 0:
        return np.zeros_like(w)
    mrc = cov @ w
    return (w * mrc) / tot_var  # somme = 1

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
    """Buy & Hold sans rebalancement, richesse totale (base 1.0)"""
    cum = (1 + returns).cumprod()
    wealth = (cum * w0).sum(axis=1)
    wealth.iloc[0] = 1.0
    return wealth

def wealth_rebalanced_every_n(returns: pd.DataFrame, w_target: np.ndarray, n_reb: int = 1) -> pd.Series:
    """Rebalancement tous les n_reb pas de temps (périodes des données)."""
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
        if denom > 0:
            w_cur = gross / denom
        else:
            w_cur = w_target.copy()
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

# ----------- Validation robuste des bornes -----------
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
        problems.append(f"La somme minimale imposée MIN_W×n = {min_w:.2%}×{n} = {min_w*n:.2%} dépasse 100%.")
        hints.append(f"Baisse MIN_W ≤ {1/n:.2%} (ou diminue le nombre d’actifs contraints).")
    if max_w * n < 1 - tol:
        problems.append(f"La somme maximale autorisée MAX_W×n = {max_w:.2%}×{n} = {max_w*n:.2%} est inférieure à 100%.")
        hints.append(f"Augmente MAX_W ≥ {1/n:.2%} (ou retire des actifs).")
    return problems, hints

def warn_tight_bounds(min_w: float, max_w: float, n: int, tol: float = 1e-12):
    slack_min = 1 - min_w * n
    slack_max = max_w * n - 1
    msgs = []
    if slack_min >= 0 and slack_min < 0.02 + tol:
        msgs.append(f"MIN_W×n est très proche de 100% (marge {slack_min:.2%}).")
    if slack_max >= 0 and slack_max < 0.02 + tol:
        msgs.append(f"MAX_W×n est très proche de 100% (marge {slack_max:.2%}).")
    if msgs:
        st.warning(
            "Bornes très serrées : " + " ".join(msgs) +
            " Cela peut rendre l’optimisation instable ou pousser vers des solutions aux bornes."
        )

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
    st.session_state["excluded"] = []  # liste des tickers exclus

detected = infer_frequency(prices_all.index)
min_date = prices_all.index.min().date()
max_date = prices_all.index.max().date()

# Sélection période
with st.expander("Périmètre temporel", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Date de début", value=min_date, min_value=min_date, max_value=max_date)
    with c2:
        end_date   = st.date_input("Date de fin", value=max_date, min_value=min_date, max_value=max_date)
    if start_date > end_date:
        st.error("La date de début doit être antérieure ou égale à la date de fin.")
        st.stop()

# Appliquer filtre temporel, puis exclusions
mask = (prices_all.index.date >= start_date) & (prices_all.index.date <= end_date)
prices = prices_all.loc[mask].copy()

# Appliquer exclusions persistantes
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

freq = infer_frequency(prices.index); k = ann_factor(freq)

st.caption(f"Fréquence détectée : {freq} | Annualisation : ×{k}")
st.caption(f"Période utilisée : {prices.index.min().date()} → {prices.index.max().date()} | Observations : {len(prices)}")
if excluded:
    st.caption(f"Actifs exclus actuellement : {', '.join(excluded)}")

# ---------- Validation des bornes AVANT toute optimisation ----------
n = len(tickers)
problems, hints = validate_weight_bounds(min_w, max_w, n)
if problems:
    st.error(
        "**Bornes de poids infaisables.**\n\n"
        + "• " + "\n• ".join(problems) +
        ("\n\n**Comment corriger :**\n" + "• " + "\n• ".join(hints) if hints else "")
    )
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

    # Listes disponibles/exclus
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
            st.rerun()

    with colB:
        st.markdown("**Réintégrer des actifs**")
        to_include = st.multiselect("Sélectionne pour réintégrer", options=excluded_all, key="to_include_list")
        if st.button("Réintégrer", use_container_width=True, key="btn_reintegrer"):
            new_excluded = [t for t in excluded_all if t not in to_include]
            st.session_state["excluded"] = new_excluded
            st.success(f"Réintégrés : {', '.join(to_include)}")
            st.rerun()

    st.caption(
        f"Actifs disponibles totaux : {len(current_all)} | "
        f"Actifs exclus : {len(st.session_state['excluded'])} | "
        f"Actifs utilisés actuellement : {len(tickers)}"
    )

# ============================ Onglet Optimisation ============================
with tab_opt:
    returns = compute_returns(prices)

    # mu = moyenne simple (annualisée)
    mu = returns.mean().values * k

    # cov
    if cov_method == "Ledoit–Wolf":
        if HAS_LW:
            lw = LedoitWolf().fit(returns.values); cov = lw.covariance_ * k
        else:
            st.warning("Ledoit–Wolf indisponible (scikit-learn manquant). Utilisation de la covariance d’échantillon.")
            cov = returns.cov().values * k
    else:
        cov = returns.cov().values * k

    # Robustesse num.
    cov = near_psd_clip(cov)
    cov = ridge_regularize(cov, ridge=1e-6)

    vol = np.sqrt(np.diag(cov))
    shp = np.where(vol > 0, (mu - RF)/vol, np.nan)  # RF=0

    st.subheader("Rendements, volatilités et Sharpe (annualisés)")
    df_metrics = pd.DataFrame({"Return_ann": mu, "Vol_ann": vol, "Sharpe_ann": shp}, index=tickers)
    df_metrics_fmt = df_metrics.copy()
    for c in ["Return_ann", "Vol_ann"]:
        df_metrics_fmt[c] = (df_metrics_fmt[c]*100).map(lambda v: f"{v:.2f}%")
    df_metrics_fmt["Sharpe_ann"] = df_metrics_fmt["Sharpe_ann"].map(lambda v: f"{v:.2f}")
    st.dataframe(df_metrics_fmt, use_container_width=True)

    # Portefeuilles optimaux (mêmes contraintes)
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

    # Résultats
    def metrics_from_w(name, w):
        ret, vol_ = portfolio_perf(w, mu, cov)
        sharpe = (ret - RF)/vol_
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

    # Contributions au risque
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

    # Monte-Carlo borné + Frontière (τ)
    st.subheader("Nuage de portefeuilles, frontière efficiente (bornée) et portefeuilles optimaux")
    rng = np.random.default_rng(int(seed))
    W = sample_bounded_simplex(n=n, N=int(n_mc), min_w=min_w, max_w=max_w, rng=rng)
    rets_mc = W @ mu
    vols_mc = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))

    front = frontier_by_tradeoff(mu, cov, min_w, max_w, npts=120)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols_mc, y=rets_mc, mode="markers",
        marker=dict(size=4, opacity=0.35),
        name="Portefeuilles aléatoires (bornés)",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
    ))
    if len(front) > 0:
        fig.add_trace(go.Scatter(
            x=front[:,1], y=front[:,0], mode="lines",
            name="Frontière efficiente (bornée, τ)",
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
    st.plotly_chart(fig, use_container_width=True)

    # Camemberts (robustes)
    def pie_series(weights: np.ndarray, labels: list, threshold: float = 0.02, eps: float = 1e-6) -> pd.Series:
        s = pd.Series(weights, index=labels)
        s = s[s > eps]
        if s.empty:
            return pd.Series([1.0], index=["N/A"])
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

    def add_pie(fig, r, c, s: pd.Series):
        if len(s) == 1:
            fig.add_trace(go.Pie(labels=[s.index[0]], values=[1.0], hole=0.35,
                                 sort=False, textinfo="label+percent", textposition="inside", showlegend=False), r, c)
        else:
            fig.add_trace(go.Pie(labels=s.index.tolist(), values=s.values.tolist(), hole=0.35,
                                 sort=False, textinfo="percent+label"), r, c)

    add_pie(pies, 1, 1, s_ms)
    add_pie(pies, 1, 2, s_mv)
    add_pie(pies, 1, 3, s_mr)
    pies.update_layout(template="plotly_white")
    st.plotly_chart(pies, use_container_width=True)

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
        freq_detected = infer_frequency(returns.index)

    win = months_to_periods(int(months_win), freq_detected)
    st.caption(f"Fenêtre utilisée : {win} périodes ({months_win} mois, fréquence {freq_detected})")

    # Matrice (fenêtre de fin)
    corr_data = returns.iloc[-win:] if len(returns) >= win else returns
    corr = corr_data.corr(method="spearman" if corr_type=="Spearman" else "pearson")
    ordered = cluster_order_from_corr(corr); corr_ord = corr.loc[ordered, ordered]

    st.subheader("Matrice de corrélation (ordre clusterisé)")
    heat = go.Figure(data=go.Heatmap(
        z=corr_ord.values, x=corr_ord.columns, y=corr_ord.index,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0, colorbar=dict(title="Corr")
    ))
    heat.update_layout(template="plotly_white", height=600)
    st.plotly_chart(heat, use_container_width=True)

    # Top 3 corrélations positives et négatives
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

    # Corrélation roulante entre 2 actifs au choix
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
        st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("Sélectionnez deux actifs distincts.")

    # Corrélation entre deux paniers (baskets)
    st.subheader("Corrélation de paniers (égal-pondérés)")
    c1, c2 = st.columns(2)
    with c1:
        basket_a = st.multiselect("Panier A — sélectionner des actifs", tickers, default=[])
    with c2:
        basket_b = st.multiselect("Panier B — sélectionner des actifs", tickers, default=[])

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
            st.plotly_chart(figb, use_container_width=True)
    else:
        st.info("Choisissez au moins un actif dans chaque panier pour calculer la corrélation des paniers.")

# ============================ Onglet Backtest ============================
with tab_bt:
    st.subheader("Backtest à partir de poids choisis")

    returns = compute_returns(prices)
    n = len(tickers)

    # Source des poids (défaut: Max Sharpe)
    src = st.radio("Source des poids", ["Manuel", "Égal-pondéré", "Max Sharpe", "Min Variance", "Max Return"],
                   horizontal=True, index=2)

    # mu/cov pour proposer des poids optimaux si besoin
    mu_bt = returns.mean().values * k
    if cov_method == "Ledoit–Wolf":
        if HAS_LW:
            lw = LedoitWolf().fit(returns.values); cov_bt = lw.covariance_ * k
        else:
            cov_bt = returns.cov().values * k
    else:
        cov_bt = returns.cov().values * k
    cov_bt = near_psd_clip(cov_bt); cov_bt = ridge_regularize(cov_bt, ridge=1e-6)

    # Poids par défaut selon la source
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
    else:  # Manuel
        w_init = np.ones(n) / n

    # --- Éditeur de poids en % (avec contrôle strict de la somme) ---
    df_edit = pd.DataFrame({"Ticker": tickers, "Poids_%": (w_init * 100)})
    edited = st.data_editor(
        df_edit,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Poids_%": st.column_config.NumberColumn("Poids (%)", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f")
        }
    )

    # Récupération et contrôles
    w_user_pct = np.asarray(edited["Poids_%"].values, dtype=float)
    total_pct = float(np.nansum(w_user_pct))

    if not np.isfinite(total_pct):
        st.error("La somme des poids est invalide (NaN/inf). Corrige les entrées.")
        st.stop()

    if total_pct <= 0:
        st.error("La somme des poids est nulle. Ajuste les valeurs pour dépasser 0%.")
        st.stop()

    if total_pct > 100.0 + 1e-9:
        excess = total_pct - 100.0
        st.error(f"La somme des poids ({total_pct:.2f}%) dépasse 100% de {excess:.2f}%. "
                 "Réduis les poids pour ne pas dépasser 100%.")
        st.stop()

    if total_pct < 100.0 - 1e-9:
        st.info(f"La somme des poids est {total_pct:.2f}%. Ils seront normalisés à 100% pour le backtest.")

    # Passage en proportions (somme = 1)
    w_user = np.clip(w_user_pct, 0, None) / total_pct
    st.caption(f"Somme des poids saisie : {total_pct:.2f}% → utilisée après normalisation : 100.00%")

    # Mode & fréquence de rebalancement (défaut: Rebalancement + Auto)
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
        st.caption(f"Rebalancement tous les {n_reb} pas de temps (périodes des données).")
    else:
        n_reb = None

    # Calcul des trajectoires
    if mode.startswith("Buy"):
        wealth = wealth_buy_hold(returns, w_user)
    else:
        wealth = wealth_rebalanced_every_n(returns, w_user, n_reb=n_reb)

    metrics, dd = perf_metrics(wealth, freq_k=k)

    # Affichage métriques
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

    # Graphiques
    st.subheader("Courbes de backtest")
    col1, col2 = st.columns(2)
    with col1:
        figw = go.Figure()
        figw.add_trace(go.Scatter(x=wealth.index, y=wealth.values, mode="lines", name="Wealth (base=1.0)"))
        figw.update_layout(template="plotly_white", yaxis_title="Valeur du portefeuille", xaxis_title="Date")
        st.plotly_chart(figw, use_container_width=True)
    with col2:
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
        figd.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        figd.update_layout(template="plotly_white", yaxis_title="Drawdown", xaxis_title="Date")
        figd.update_yaxes(tickformat=".0%")
        st.plotly_chart(figd, use_container_width=True)

    st.caption(
        "Notes : "
        "1) 'Auto' = rebalancement à chaque période des données (quotidienne/hebdo/mensuelle...). "
        "2) 'Tous les N périodes' applique un rebalancement toutes N lignes de données. "
        "3) Les métriques sont calculées à la fréquence détectée puis annualisées (×k pour le rendement moyen, ×√k pour la volatilité)."
    )
