# =========================================================
# MARKOWITZ PORTFOLIO OPTIMIZER — version épurée & pro
# - Sources: Excel OU Yahoo Finance (yfinance, period=max)
# - Tickers normalisés en MAJUSCULES
# - Intersection auto + message "ticker limitant" (date commune)
# - Visualisation prix (rebase/log), téléchargements
# - Optimisation Markowitz (bornée), corrélations, backtest
# - Plotly propre: width='stretch', pas de kwargs dépréciés
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

# Yahoo Finance
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ---- helper rerun (compat old/new streamlit) ----
def _safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ---- Plotly helper (corrige le warning kwargs dépréciés) ----
def st_plotly_chart(fig: go.Figure) -> None:
    # Mise en page propre & compacte
    fig.update_layout(autosize=True, margin=dict(l=40, r=20, t=40, b=40))

    # Toute la config dans "config" (plus de kwargs implicites)
    plotly_config = {
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        # Ajoute/retire ce dont tu as besoin (modeBarButtons, toImageButtonOptions, etc.)
    }

    # >>> LE POINT-CLÉ <<<
    # 1) theme=None => empêche Streamlit d'injecter sa config Plotly via kwargs (source du warning)
    # 2) width='stretch' => remplace use_container_width
    st.plotly_chart(fig, config=plotly_config, width='stretch', theme=None)



# ============================ Utils ============================
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
    raise RuntimeError(f"Echec de l’optimisation : {res.message}")

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
    w_mv = min_var_weights(mu, cov, min_w, max_w); r_mv, _ = portfolio_perf(w_mv, mu, cov)
    w_mr = max_return_weights(mu, cov, min_w, max_w); r_mr, _ = portfolio_perf(w_mr, mu, cov)
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
        port_val *= (1 + rp); values[t] = port_val
        gross = w_cur * (1 + r_vec); denom = gross.sum()
        w_cur = gross / denom if denom > 0 else w_target.copy()
        if ((t + 1) % max(1, int(n_reb))) == 0:
            w_cur = w_target.copy()
    wealth = pd.Series(values, index=returns.index); wealth.iloc[0] = 1.0
    return wealth

def perf_metrics(wealth: pd.Series, freq_k: int):
    ret = wealth.pct_change().dropna()
    total_return = wealth.iloc[-1] - 1.0
    years = len(ret) / freq_k if freq_k > 0 else np.nan
    cagr = wealth.iloc[-1] ** (1/years) - 1 if years and years > 0 else np.nan
    vol_ann = ret.std(ddof=1) * np.sqrt(freq_k)
    sharpe = (ret.mean() * freq_k) / vol_ann if vol_ann > 0 else np.nan
    cummax = wealth.cummax(); dd = wealth / cummax - 1.0
    max_dd = dd.min(); worst = ret.min()
    return {"Total Return": total_return, "CAGR": cagr, "Vol ann.": vol_ann,
            "Sharpe": sharpe, "Max Drawdown": max_dd, "Worst period": worst}, dd

def n_for_rebalance_choice(choice: str, data_freq: str, n_custom: int) -> int:
    if choice == "Auto (fréquence des données)": return 1
    if choice == "Mensuel":   return months_to_periods(1, data_freq)
    if choice == "Trimestriel": return months_to_periods(3, data_freq)
    if choice == "Annuel":    return months_to_periods(12, data_freq)
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
@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index().dropna(axis=1, how="all")
    # Colonnes (tickers) en MAJ
    df.columns = [str(c).upper() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices_to_returns(prices)

@st.cache_data(show_spinner=True)
def fetch_yahoo_prices(
    tickers: list[str],
    interval: str = "1d",
    auto_adjust: bool = True,
):
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    # MAJUSCULES
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
        if "Close" in df.columns:
            t = tickers_up[0]; s_union = df["Close"].rename(t)
            first = s_union.first_valid_index(); last  = s_union.last_valid_index()
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

# ============================ UI PRINCIPALE ============================
st.title("Optimisation de portefeuille (Markowitz)")

with st.container():
    st.markdown("## Source des données")
    data_src = st.radio("Choisir une source :", ["Fichier Excel", "Yahoo Finance"], horizontal=True)

    prices_all = None
    availability = None

    if data_src == "Fichier Excel":
        uploaded_file = st.file_uploader("Fichier Excel (.xlsx/.xls)", type=["xlsx", "xls"])
        if uploaded_file is not None:
            prices_union = load_excel(uploaded_file)
            availability = build_availability_from_union(prices_union)
            prices_all = prices_union.dropna(how="any")
    else:
        if not HAS_YF:
            st.error("yfinance n’est pas installé : `pip install yfinance`")
        col1, col2 = st.columns([2, 1])
        with col1:
            tick_input = st.text_input(
                "Tickers Yahoo (séparés par des virgules)",
                value="",
                placeholder="Ex: ^GSPC, GC=F, BTC-USD"
            )
        with col2:
            yf_interval = st.selectbox("Intervalle", options=["1d", "1wk", "1mo"], index=0)
            yf_auto_adjust = st.checkbox("Ajuster dividendes/splits", value=True)

        # Normalise en MAJ
        yf_tickers = [t.strip().upper() for t in tick_input.split(",") if t.strip()]
        if st.button("📥 Charger depuis Yahoo (période max)"):
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
        # Persistance si déjà chargées
        if prices_all is None:
            prices_all = st.session_state.get("yf_prices_all", None)
            availability = st.session_state.get("yf_availability", None)

# Si pas de données -> stop
if prices_all is None or prices_all.empty:
    st.info("Charge des données (Excel ou Yahoo) ci-dessus pour continuer.")
    st.stop()

# S'assure que les colonnes sont en MAJ
prices_all.columns = [str(c).upper() for c in prices_all.columns]

all_tickers = prices_all.columns.tolist()
if len(all_tickers) < 2:
    st.error("Il faut au moins deux actifs.")
    st.stop()

# État persistant des actifs exclus
if "excluded" not in st.session_state:
    st.session_state["excluded"] = []

# ========== Périmètre temporel par défaut (intersection max) ==========
if availability is not None and not availability.empty:
    # Normalise MAJ pour l'affichage
    availability["Ticker"] = availability["Ticker"].astype(str).str.upper()
    start_def = pd.to_datetime(availability["First"]).max().date()
    end_def   = pd.to_datetime(availability["Last"]).min().date()
else:
    start_def = prices_all.index.min().date()
    end_def   = prices_all.index.max().date()

with st.expander("Périmètre temporel", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Date de début", value=start_def, min_value=start_def, max_value=end_def)
    with c2:
        end_date   = st.date_input("Date de fin", value=end_def,   min_value=start_def, max_value=end_def)
    if start_date > end_date:
        st.error("La date de début doit être antérieure ou égale à la date de fin.")
        st.stop()

# Filtre dates + exclusions
mask = (prices_all.index.date >= start_date) & (prices_all.index.date <= end_date)
prices = prices_all.loc[mask].copy()
excluded = [t for t in st.session_state["excluded"] if t in prices.columns]
if excluded:
    prices = prices.drop(columns=excluded, errors="ignore")

tickers = [str(c).upper() for c in prices.columns]
prices.columns = tickers  # assure MAJ après filtre

if prices.shape[0] < 3:
    st.error("Trop peu d’observations après filtrage. Élargissez la période.")
    st.stop()
if len(tickers) < 2:
    st.error("Il faut au moins deux actifs après exclusions.")
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

# ============================ Sidebar (bornes & MC) ============================
st.sidebar.header("Paramètres d’optimisation")
min_w = st.sidebar.slider("Poids minimum par actif", 0.0, 1.0, 0.00, 0.01)
max_w = st.sidebar.slider("Poids maximum par actif", 0.0, 1.0, 1.00, 0.01)
n_mc  = st.sidebar.number_input("Nombre de portefeuilles Monte Carlo", value=1000, step=500)
seed  = st.sidebar.number_input("Graine aléatoire (Monte Carlo)", value=42, step=1)

problems, hints = validate_weight_bounds(min_w, max_w, len(tickers))
if problems:
    st.error("**Bornes de poids infaisables.**\n\n" + "• " + "\n• ".join(problems) +
             ("\n\n**Comment corriger :**\n" + "• " + "\n• ".join(hints) if hints else ""))
    st.stop()
else:
    warn_tight_bounds(min_w, max_w, len(tickers))

# ---------------- Tabs ----------------
tab_data, tab_opt, tab_fig, tab_corr, tab_bt = st.tabs(["Données", "Optimisation", "Graphiques", "Corrélation", "Backtest"])

# ============================ Onglet Données ============================
with tab_data:
    st.subheader("Tableau de prix utilisé (après filtre de dates et exclusions)")
    st.dataframe(prices, width='stretch')

    # Téléchargements
    st.markdown("### Téléchargement des données")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    prices_csv = prices.to_csv(index=True).encode("utf-8")
    prices_all_csv = prices_all.to_csv(index=True).encode("utf-8")
    returns_filtered = compute_returns(prices)
    returns_csv = returns_filtered.to_csv(index=True).encode("utf-8")
    with col_dl1:
        st.download_button("📥 Prix filtrés (CSV)", data=prices_csv, file_name="prices_filtered.csv", mime="text/csv")
    with col_dl2:
        st.download_button("📥 Prix bruts source (CSV)", data=prices_all_csv, file_name="prices_raw_source.csv", mime="text/csv")
    with col_dl3:
        st.download_button("📥 Rendements filtrés (CSV)", data=returns_csv, file_name="returns_filtered.csv", mime="text/csv")

    # Explication de la date de début commune (ticker limitant)
    if availability is not None and not availability.empty:
        # MAJ déjà normalisé
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
        rebase = st.checkbox("Rebase (base=100)", value=True, key="data_plot_rebase")
    with col_right:
        log_scale_prices = st.checkbox("Échelle log", value=False, key="data_plot_log")

    plot_df = prices.copy()
    if rebase:
        plot_df = (plot_df / plot_df.iloc[0]) * 100.0

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
        template="plotly_white",
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
        to_exclude = st.multiselect("Sélectionne pour exclure", options=available_all, key="to_exclude_list")
        if st.button("Exclure", type="primary", key="btn_exclure"):
            new_excluded = sorted(list(set([s.upper() for s in (excluded_all + to_exclude)])))
            st.session_state["excluded"] = new_excluded
            st.success(f"Exclus : {', '.join(to_exclude)}"); _safe_rerun()
    with colB:
        st.markdown("**Réintégrer des actifs**")
        to_include = st.multiselect("Sélectionne pour réintégrer", options=excluded_all, key="to_include_list")
        if st.button("Réintégrer", key="btn_reintegrer"):
            new_excluded = [t for t in excluded_all if t not in to_include]
            st.session_state["excluded"] = new_excluded
            st.success(f"Réintégrés : {', '.join(to_include)}"); _safe_rerun()

    st.caption(f"Actifs totaux : {len(current_all)} | Exclus : {len(st.session_state['excluded'])} | Utilisés : {len(tickers)}")

# ============================ Onglet Optimisation ============================
with tab_opt:
    returns = compute_returns(prices)
    mu = returns.mean().values * k

    use_lw = HAS_LW and st.checkbox("Utiliser Ledoit–Wolf (si dispo)", value=True)
    if use_lw:
        lw = LedoitWolf().fit(returns.values); cov = lw.covariance_ * k
    else:
        cov = returns.cov().values * k

    cov = near_psd_clip(cov); cov = ridge_regularize(cov, ridge=1e-6)
    vol = np.sqrt(np.diag(cov)); shp = np.where(vol > 0, (mu)/vol, np.nan)

    st.subheader("Rendements, volatilités et Sharpe (annualisés)")
    df_metrics = pd.DataFrame({"Return_ann": mu, "Vol_ann": vol, "Sharpe_ann": shp}, index=tickers)
    df_metrics_fmt = df_metrics.copy()
    for c in ["Return_ann", "Vol_ann"]:
        df_metrics_fmt[c] = (df_metrics_fmt[c]*100).map(lambda v: f"{v:.2f}%")
    df_metrics_fmt["Sharpe_ann"] = df_metrics_fmt["Sharpe_ann"].map(lambda v: f"{v:.2f}")
    st.dataframe(df_metrics_fmt, width='stretch')

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
        ret, vol_ = portfolio_perf(w, mu, cov); sharpe = (ret)/vol_
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
    st.subheader("Résultats de l’optimisation"); st.dataframe(df_res_fmt, width='stretch')

    st.subheader("Contributions au risque (somme = 100%)")
    df_rc = pd.DataFrame({
        "Max Sharpe": risk_contrib(w_ms, cov),
        "Min Variance": risk_contrib(w_mv, cov),
        "Max Return": risk_contrib(w_mr, cov)
    }, index=tickers)
    st.dataframe((df_rc*100).round(2).astype(str) + " %", width='stretch')

# ============================ Onglet Graphiques ============================
with tab_fig:
    returns = compute_returns(prices)
    mu = returns.mean().values * k
    if HAS_LW:
        lw = LedoitWolf().fit(returns.values); cov = lw.covariance_ * k
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
        xaxis_title="Volatilité (ann.)", yaxis_title="Rendement (ann.)",
        template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest"
    )
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
    st_plotly_chart(fig)

    # Camemberts
    def pie_series(weights: np.ndarray, labels: list, threshold: float = 0.02, eps: float = 1e-6) -> pd.Series:
        s = pd.Series(weights, index=labels); s = s[s > eps]
        if s.empty: return pd.Series([1.0], index=["N/A"])
        s = s.sort_values(ascending=False)
        if threshold and threshold > 0:
            majors = s[s >= threshold]; others = s[s < threshold].sum()
            if others > eps: majors.loc["Others"] = others
            s = majors
        s = s / s.sum(); return s

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
    add_pie(pies, 1, 1, s_ms); add_pie(pies, 1, 2, s_mv); add_pie(pies, 1, 3, s_mr)
    pies.update_layout(template="plotly_white"); st_plotly_chart(pies)

    st.subheader("Poids par portefeuille (%)")
    weights_df = pd.DataFrame({"Max Sharpe": w_ms, "Min Variance": w_mv, "Max Return": w_mr}, index=tickers)
    st.dataframe((weights_df * 100).round(2).astype(str) + " %", width='stretch')

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
        st_plotly_chart(figc)
    else:
        st.info("Sélectionnez deux actifs distincts.")

# ============================ Onglet Backtest ============================
with tab_bt:
    st.subheader("Backtest à partir de poids choisis")
    returns = compute_returns(prices); n = len(tickers)
    src = st.radio("Source des poids", ["Manuel", "Égal-pondéré", "Max Sharpe", "Min Variance", "Max Return"],
                   horizontal=True, index=2)

    mu_bt = returns.mean().values * k
    if HAS_LW:
        lw = LedoitWolf().fit(returns.values); cov_bt = lw.covariance_ * k
    else:
        cov_bt = returns.cov().values * k
    cov_bt = near_psd_clip(cov_bt); cov_bt = ridge_regularize(cov_bt, ridge=1e-6)

    if src == "Égal-pondéré":
        w_init = np.ones(n) / n
    elif src == "Max Sharpe":
        try: w_init = max_sharpe_weights(mu_bt, cov_bt, min_w, max_w)
        except RuntimeError:
            cov_bt = ridge_regularize(cov_bt, ridge=1e-4); w_init = max_sharpe_weights(mu_bt, cov_bt, min_w, max_w)
    elif src == "Min Variance":
        try: w_init = min_var_weights(mu_bt, cov_bt, min_w, max_w)
        except RuntimeError:
            cov_bt = ridge_regularize(cov_bt, ridge=1e-4); w_init = min_var_weights(mu_bt, cov_bt, min_w, max_w)
    elif src == "Max Return":
        try: w_init = max_return_weights(mu_bt, cov_bt, min_w, max_w)
        except RuntimeError:
            w_init = feasible_start(n, min_w, max_w)
    else:
        w_init = np.ones(n) / n

    df_edit = pd.DataFrame({"Ticker": tickers, "Poids_%": (w_init * 100)})
    edited = st.data_editor(
        df_edit,
        num_rows="fixed",
        column_config={"Poids_%": st.column_config.NumberColumn("Poids (%)", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f")},
        width='stretch'
    )
    w_user_pct = np.asarray(edited["Poids_%"].values, dtype=float)
    total_pct = float(np.nansum(w_user_pct))
    if not np.isfinite(total_pct):
        st.error("La somme des poids est invalide (NaN/inf)."); st.stop()
    if total_pct <= 0:
        st.error("La somme des poids est nulle."); st.stop()
    if total_pct > 100.0 + 1e-9:
        st.error(f"La somme des poids ({total_pct:.2f}%) dépasse 100%."); st.stop()
    if total_pct < 100.0 - 1e-9:
        st.info(f"Somme actuelle {total_pct:.2f}%. Normalisation à 100% appliquée.")
    w_user = np.clip(w_user_pct, 0, None) / total_pct
    st.caption(f"Somme des poids saisie : {total_pct:.2f}% → utilisée après normalisation : 100.00%")

    mode = st.radio("Mode de backtest", ["Buy & Hold (sans rebalancement)", "Rebalancement"], horizontal=True, index=1)
    if mode == "Rebalancement":
        col_rb1, col_rb2 = st.columns([2,1])
        with col_rb1:
            reb_choice = st.selectbox("Fréquence de rebalancement",
                                      ["Auto (fréquence des données)", "Mensuel", "Trimestriel", "Annuel", "Tous les N périodes"], index=0)
        with col_rb2:
            n_custom = st.number_input("N périodes (si 'Tous les N périodes')", value=1, min_value=1, step=1,
                                       disabled=(reb_choice != "Tous les N périodes"))
        n_reb = n_for_rebalance_choice(reb_choice, freq, n_custom)
        st.caption(f"Rebalancement tous les {n_reb} pas de temps.")
    else:
        n_reb = None

    wealth = wealth_buy_hold(returns, w_user) if mode.startswith("Buy") else wealth_rebalanced_every_n(returns, w_user, n_reb=n_reb)
    metrics, dd = perf_metrics(wealth, freq_k=k)

    st.subheader("Métriques du backtest")
    mdf = pd.DataFrame({
        "Total Return": [f"{metrics['Total Return']*100:.2f}%"],
        "CAGR": [f"{metrics['CAGR']*100:.2f}%"] if pd.notna(metrics["CAGR"]) else ["N/A"],
        "Vol ann.": [f"{metrics['Vol ann.']*100:.2f}%"] if pd.notna(metrics["Vol ann."]) else ["N/A"],
        "Sharpe": [f"{metrics['Sharpe']:.2f}"] if pd.notna(metrics["Sharpe"]) else ["N/A"],
        "Max Drawdown": [f"{metrics['Max Drawdown']*100:.2f}%"] if pd.notna(metrics["Max Drawdown"]) else ["N/A"],
        "Worst period": [f"{metrics['Worst period']*100:.2f}%"] if pd.notna(metrics["Worst period"]) else ["N/A"]
    })
    st.dataframe(mdf, width='stretch')

    st.subheader("Risk & Return Metrics")
    ret = wealth.pct_change().dropna()
    monthly_ret = (1 + ret).resample("ME").prod() - 1
    pv_df = pd.DataFrame({
        "Arithmetic Mean (annualized)": [f"{((1+monthly_ret.mean())**12 - 1)*100:.2f}%"] if len(monthly_ret)>0 else ["N/A"],
        "Geometric Mean (annualized)":  [f"{(((1+monthly_ret).prod())**(12/len(monthly_ret)) - 1)*100:.2f}%"] if len(monthly_ret)>0 else ["N/A"],
        "Std (annualized)":             [f"{monthly_ret.std(ddof=1)*np.sqrt(12)*100:.2f}%"] if len(monthly_ret)>0 else ["N/A"],
        "Sharpe (native)":              [f"{metrics['Sharpe']:.2f}"] if pd.notna(metrics["Sharpe"]) else ["N/A"],
        "Max Drawdown":                 [f"{metrics['Max Drawdown']*100:.2f}%"] if pd.notna(metrics["Max Drawdown"]) else ["N/A"],
    })
    st.dataframe(pv_df, width='stretch')

    st.subheader("Courbes de backtest")
    col1, col2 = st.columns(2)
    with col1:
        log_scale = st.checkbox("Échelle log (Wealth)", value=False)
        wealth_plot = wealth.where(wealth > 0, np.nan) if log_scale else wealth
        figw = go.Figure()
        figw.add_trace(go.Scatter(x=wealth_plot.index, y=wealth_plot.values, mode="lines", name="Wealth (base=1.0)"))
        figw.update_layout(template="plotly_white", yaxis_title="Valeur du portefeuille", xaxis_title="Date")
        if log_scale: figw.update_yaxes(type="log")
        st_plotly_chart(figw)
    with col2:
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
        figd.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        figd.update_layout(template="plotly_white", yaxis_title="Drawdown", xaxis_title="Date")
        figd.update_yaxes(tickformat=".0%")
        st_plotly_chart(figd)
