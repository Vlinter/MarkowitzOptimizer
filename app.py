# app.py
# =========================================================
# MARKOWITZ PORTFOLIO OPTIMIZER — v9.2 (Simplifié)
# =========================================================
import numpy as np
import pandas as pd
import streamlit as st

# Imports depuis les nouveaux modules
from utils import (
    setup_warnings, _safe_rerun, infer_frequency, ann_factor,
    validate_weight_bounds, warn_tight_bounds, cluster_order_from_corr,
    top_corr_pairs, n_for_rebalance_choice, months_to_periods,
    near_psd_clip, ridge_regularize
)
from data_loader import (
    HAS_YF, load_excel, compute_returns, fetch_yahoo_prices,
    fetch_benchmark_prices, build_availability_from_union,
    fetch_risk_free_rate 
    # fetch_inflation_index a été supprimé
)
from core_optimizer import (
    HAS_LW, LedoitWolf, portfolio_perf, risk_contrib,
    max_sharpe_weights, min_var_weights, max_return_weights, risk_parity_weights,
    efficient_frontier, sample_bounded_simplex, feasible_start
)
from core_backtest import (
    run_backtest, perf_metrics, calculate_relative_metrics,
    calculate_market_regime_metrics, calculate_var_cvar,
    create_monthly_heatmap_df, find_drawdowns
)
from visualizations import (
    st_plotly_chart, plot_price_history, plot_asset_histogram,
    plot_efficient_frontier, plot_allocation_pies, plot_correlation_heatmap,
    plot_rolling_correlation, plot_backtest_wealth, plot_backtest_drawdown,
    plot_annual_returns, plot_monthly_heatmap, plot_allocation_history,
    plot_rolling_volatility, plot_monthly_histogram,
    plot_risk_contrib_history 
)

# ---- Configuration de la Page ----
setup_warnings()
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ---- CSS Personnalisé ----
st.markdown("""
<style>
[data-testid="stVerticalBlock"] > [style*="border: 1px solid"] { border-radius: 10px; }
[data-testid="stExpander"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================ UI PRINCIPALE ============================
st.title("Optimisation de portefeuille (Markowitz)")
st.markdown("Analysez la frontière efficiente et backtestez des portefeuilles selon la théorie de Markowitz.")
st.divider()

# ============================ 1. Source des données ============================
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

# ============================ 2. Paramètres de l'analyse ============================
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
                
            st.markdown("---") 
            
            c2_mu, c2_cov = st.columns(2)
            with c2_mu:
                mu_method_static = "Moyenne historique"
                st.markdown("Méthode d'estimation (μ) :")
                st.info("Moyenne historique (par défaut)")
            with c2_cov:
                cov_method_static = st.selectbox(
                    "Méthode d'estimation (Cov)", 
                    options=["Standard", "Ledoit-Wolf"], 
                    index=1, 
                    key="cov_method_static",
                    help="Comment calculer la matrice de covariance (risque). 'Ledoit-Wolf' est recommandé pour sa robustesse."
                )
            
            ewma_span_months_static = 12
            
            st.markdown("---")
            use_rf = st.checkbox(
                "Utiliser Taux Sans Risque (Rf)", 
                value=True, 
                key="use_rf_checkbox",
                help="Soustrait le taux sans risque (ex: ^IRX) pour les calculs de Sharpe, Sortino, Calmar et Alpha."
            )
            rf_ticker = "^IRX"
            if use_rf:
                rf_ticker_input = st.text_input(
                    "Ticker Taux Sans Risque",
                    value="^IRX",
                    key="rf_ticker_input",
                    help="Ticker Yahoo Finance pour le taux sans risque annualisé. ^IRX = US T-Bill 3M. Pour l'Europe, essayez ^TE (Allemagne 3M)."
                )
                rf_ticker = rf_ticker_input.split(',')[0].split(' ')[0].strip()
                if rf_ticker != rf_ticker_input and rf_ticker_input.strip():
                    st.caption(f"Note : Seul le premier ticker ({rf_ticker}) sera utilisé.")
                    

if start_date > end_date:
    st.error("La date de début doit être antérieure ou égale à la date de fin.")
    st.stop()

# ---- Filtrage final des données ----
mask = (prices_all.index.date >= start_date) & (prices_all.index.date <= end_date)
prices = prices_all.loc[mask].copy()
excluded = [t for t in st.session_state["excluded"] if t in prices.columns]
if excluded:
    prices = prices.drop(columns=excluded, errors="ignore")

tickers = [str(c).upper() for c in prices.columns]
prices.columns = tickers

if prices.shape[0] < 3: st.error("Trop peu d’observations après filtrage."); st.stop()
if len(tickers) < 2: st.error("Il faut au moins deux actifs après exclusions."); st.stop()

st.caption(f"Période utilisée : {prices.index.min().date()} → {prices.index.max().date()} | Observations : {len(prices)}")
if excluded: st.caption(f"Actifs exclus actuellement : {', '.join(excluded)}")

problems, hints = validate_weight_bounds(min_w, max_w, len(tickers))
if problems:
    st.error("**Bornes de poids infaisables.**\n\n" + "• " + "\n• ".join(problems) +
            ("\n\n**Comment corriger :**\n" + "• " + "\n• ".join(hints) if hints else ""))
    st.stop()
else:
    warn_tight_bounds(min_w, max_w, len(tickers))

# --- TÉLÉCHARGEMENT DU TAUX SANS RISQUE (MOYENNE ET SÉRIE) ---
rf_ann_mean = 0.0
rf_series_aligned = pd.Series(dtype=float) 
if use_rf and HAS_YF:
    rf_data_annual = fetch_risk_free_rate(rf_ticker, start_date, end_date)
    if not rf_data_annual.empty:
        rf_series_aligned = rf_data_annual.reindex(prices.index).ffill().bfill() 
        rf_ann_mean = rf_series_aligned.mean() 

        if pd.isna(rf_ann_mean):
            st.warning(f"Aucune donnée de taux sans risque trouvée pour {rf_ticker} dans la plage de dates. Utilisation de 0.")
            rf_ann_mean = 0.0
            rf_series_aligned = pd.Series(0.0, index=prices.index)
        else:
            st.caption(f"Taux sans risque moyen (annualisé) utilisé : {rf_ann_mean:.2%}")
            rf_series_aligned = rf_series_aligned.fillna(rf_ann_mean)
    else:
        rf_ann_mean = 0.0
        rf_series_aligned = pd.Series(0.0, index=prices.index)
elif use_rf and not HAS_YF:
    st.error("yfinance n'est pas installé. Impossible de charger le taux sans risque. Utilisation de 0.")
    rf_ann_mean = 0.0
    rf_series_aligned = pd.Series(0.0, index=prices.index)
else:
    rf_series_aligned = pd.Series(0.0, index=prices.index)
# --- FIN RF ---


# ====================================================================
# ========== SECTION DES CALCULS CACHÉS (MODIFIÉE) ==========
# ====================================================================
@st.cache_data(show_spinner="Calcul des statistiques (Rendements, Covariance)...")
def get_stats(
    prices_hash: str, 
    k: int, 
    data_freq: str,
    mu_method: str,
    cov_method: str,       
    ewma_span_months: int,
    _prices: pd.DataFrame
):
    returns = compute_returns(_prices)
    n_assets = _prices.shape[1]
    
    if cov_method == "Ledoit-Wolf" and HAS_LW:
        lw = LedoitWolf().fit(returns.values); cov_raw = lw.covariance_ * k
    else: # "Standard"
        cov_raw = returns.cov().values * k
        
    cov_raw = np.nan_to_num(cov_raw, nan=0.0, posinf=0.0, neginf=0.0)
    cov_static = near_psd_clip(cov_raw); cov_static = ridge_regularize(cov_static, ridge=1e-6)

    mu_static = returns.mean().values * k
    
    return returns, mu_static, cov_static

@st.cache_data(show_spinner="Calcul des portefeuilles optimaux (statiques)...")
def run_static_optimization(mu, cov, min_w, max_w, rf=0.0, tickers=None): 
    n = len(mu)
    w_ms, w_mv, w_mr, w_rp = (np.ones(n)/n,) * 4 
    try:
        w_ms = max_sharpe_weights(mu, cov, min_w, max_w, rf=rf, tickers=tickers) 
    except Exception: w_ms = feasible_start(n, min_w, max_w)
    try: w_mv = min_var_weights(mu, cov, min_w, max_w, rf=rf, tickers=tickers)
    except Exception: w_mv = feasible_start(n, min_w, max_w)
    try: w_mr = max_return_weights(mu, cov, min_w, max_w, rf=rf, tickers=tickers)
    except Exception: w_mr = feasible_start(n, min_w, max_w)
    try: w_rp = risk_parity_weights(mu, cov, min_w, max_w, rf=rf, tickers=tickers)
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


st.divider()
st.markdown("#### 3. Analyse")

# ====================================================================
# ========== SECTION DE NAVIGATION (AVEC ÉTAT) ==========
# ====================================================================

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Données" 

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
    rebase = col_left.checkbox(
        "Rebase (base=100)", 
        value=True, 
        key="data_plot_rebase",
        help="Affiche l'évolution de tous les actifs à partir d'une base commune de 100, au lieu de leurs prix réels."
    )
    log_scale_prices = col_right.checkbox(
        "Échelle log", 
        value=True, 
        key="data_plot_log",
        help="Utilise une échelle logarithmique sur l'axe Y, utile pour visualiser les variations en pourcentage sur de longues périodes."
    )

    fig_data = plot_price_history(prices, rebase, log_scale_prices)
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
        prices_hash = pd.util.hash_pandas_object(prices).sum()
        returns_stats, mu_static_stats, cov_static_stats = get_stats(
            prices_hash=prices_hash, k=k, data_freq=freq,
            mu_method=mu_method_static, cov_method=cov_method_static, 
            ewma_span_months=ewma_span_months_static, _prices=prices 
        )
        
        idx = tickers.index(ticker_to_analyze)
        ret_ann = mu_static_stats[idx]
        vol_ann = np.sqrt(cov_static_stats[idx, idx])
        shp = (ret_ann - rf_ann_mean) / vol_ann if vol_ann > 0 else np.nan
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
            help=f"Rendement annualisé ajusté au risque (basé sur Rf = {rf_ann_mean:.2%}). Mesure du rendement excédentaire par unité de risque."
        )
        
        fig_hist_ret = plot_asset_histogram(returns_stats[ticker_to_analyze], ticker_to_analyze, freq)
        st_plotly_chart(fig_hist_ret)

# ============================ Onglet Optimisation ============================
elif selected_tab == "⚙️ Optimisation":
    st.info("Optimisation **statique**, calculée sur **l'ensemble** de la période (section 2).")
    
    try:
        prices_hash = pd.util.hash_pandas_object(prices).sum()
        returns, mu_static, cov_static = get_stats(
            prices_hash=prices_hash, k=k, data_freq=freq,
            mu_method=mu_method_static, cov_method=cov_method_static, 
            ewma_span_months=ewma_span_months_static, _prices=prices 
        )
        
        w_ms_static, w_mv_static, w_mr_static, w_rp_static = run_static_optimization(
            mu_static, cov_static, min_w, max_w, 
            rf=rf_ann_mean, 
            tickers=tickers 
        )
    except Exception as e:
        st.error(f"Erreur irrécupérable lors de l'optimisation statique : {e}")
        st.exception(e) 
        st.stop()

    vol = np.sqrt(np.diag(cov_static))
    shp = np.where(vol > 0, (mu_static - rf_ann_mean)/vol, np.nan)
    
    st.subheader(f"Rendements, volatilités et Sharpe (Rf = {rf_ann_mean:.2%})")
    st.caption("Statistiques individuelles des actifs, calculées sur la période sélectionnée (section 2).")
    df_metrics = pd.DataFrame({"Return_ann": mu_static, "Vol_ann": vol, "Sharpe_ann": shp}, index=tickers)
    df_metrics_fmt = df_metrics.applymap(lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A')
    df_metrics_fmt["Sharpe_ann"] = df_metrics["Sharpe_ann"].map(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
    st.dataframe(df_metrics_fmt, width='stretch')

    def metrics_from_w(name, w):
        ret, vol_ = portfolio_perf(w, mu_static, cov_static)
        sharpe = (ret - rf_ann_mean)/vol_ if vol_ > 0 else 0
        return {"Portefeuille": name, "Return": ret, "Vol": vol_, "Sharpe": sharpe}

    df_res = pd.DataFrame([
        metrics_from_w("Max Sharpe", w_ms_static),
        metrics_from_w("Min Variance", w_mv_static),
        metrics_from_w("Max Return", w_mr_static),
        metrics_from_w("Risk Parity", w_rp_static),
    ]).set_index("Portefeuille")

    st.subheader("Résultats de l’optimisation")
    st.caption("Portefeuilles optimaux calculés selon la théorie de Markowitz, en respectant les contraintes de poids (section 2).")
    
    c1, c2, c3, c4 = st.columns(4) 
    r_ms = df_res.loc["Max Sharpe"]; r_mv = df_res.loc["Min Variance"]; r_rp = df_res.loc["Risk Parity"]
    r_mr = df_res.loc["Max Return"]
    c1.metric("🏆 Max Sharpe", f"{r_ms['Sharpe']:.2f}", f"Ret: {r_ms['Return']:.1%} | Vol: {r_ms['Vol']:.1%}", delta_color="off", help="Portefeuille qui maximise le Ratio de Sharpe (rendement ajusté au risque) sur la frontière efficiente.")
    c2.metric("🛡️ Min Variance", f"{r_mv['Vol']:.1%}", f"Ret: {r_mv['Return']:.1%} | Sharpe: {r_mv['Sharpe']:.2f}", delta_color="off", help="Portefeuille qui minimise la volatilité (risque) totale, quel que soit le rendement.")
    c3.metric("⚖️ Risk Parity", f"{r_rp['Sharpe']:.2f}", f"Ret: {r_rp['Return']:.1%} | Vol: {r_rp['Vol']:.1%}", delta_color="off", help="Portefeuille qui tente d'égaliser la contribution au risque de chaque actif (ERC).")
    c4.metric("🚀 Max Return", f"{r_mr['Return']:.1%}", f"Vol: {r_mr['Vol']:.1%} | Sharpe: {r_mr['Sharpe']:.2f}", delta_color="off", help="Portefeuille qui maximise le rendement, quel que soit le risque (généralement 100% sur l'actif le plus performant).")
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
    st.info("Optimisation **statique**, calculée sur **l'ensemble** de la période (section 2).")
    
    try:
        prices_hash = pd.util.hash_pandas_object(prices).sum()
        returns, mu_static, cov_static = get_stats(
            prices_hash=prices_hash, k=k, data_freq=freq,
            mu_method=mu_method_static, cov_method=cov_method_static, 
            ewma_span_months=ewma_span_months_static, _prices=prices 
        )
        
        w_ms_static, w_mv_static, w_mr_static, w_rp_static = run_static_optimization(
            mu_static, cov_static, min_w, max_w, 
            rf=rf_ann_mean, 
            tickers=tickers 
        )
        
        rets_mc, vols_mc = run_monte_carlo(mu_static, cov_static, min_w, max_w, n_mc, seed)
        front = run_efficient_frontier(mu_static, cov_static, min_w, max_w)
    except Exception as e:
        st.error(f"Erreur irrécupérable lors de l'optimisation statique : {e}")
        st.exception(e) 
        st.stop()

    n = len(tickers)
    st.subheader("Nuage de portefeuilles, frontière efficiente (bornée) et portefeuilles optimaux")
    st.caption("Visualisation de l'optimisation. Chaque point est un portefeuille.")
    
    port_points = []
    for name, w, sym, c in [
            ("Max Sharpe", w_ms_static, "star", "red"), 
            ("Min Variance", w_mv_static, "circle", "green"), 
            ("Risk Parity", w_rp_static, "diamond", "purple"),
            ("Max Return", w_mr_static, "square", "cyan")
        ]:
        r, v = portfolio_perf(w, mu_static, cov_static)
        port_points.append((name, r, v, sym, c))
    
    fig = plot_efficient_frontier(vols_mc, rets_mc, front, port_points)
    st_plotly_chart(fig)

    pies = plot_allocation_pies(w_ms_static, w_mv_static, w_rp_static, w_mr_static, tickers) 
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

    try:
        prices_hash = pd.util.hash_pandas_object(prices).sum()
        returns, _, _ = get_stats(
            prices_hash=prices_hash, k=k, data_freq=freq,
            mu_method=mu_method_static, cov_method=cov_method_static, 
            ewma_span_months=ewma_span_months_static, _prices=prices 
        )
    except Exception as e:
        st.error(f"Erreur lors du calcul des rendements : {e}")
        st.stop()

    st.subheader("Paramètres de corrélation")
    c1, c2, c3 = st.columns(3)
    corr_type = c1.selectbox(
        "Type de corrélation", 
        ["Pearson", "Spearman"], 
        index=0, 
        key="corr_type_select",
        help="Méthode de calcul : 'Pearson' (linéaire) ou 'Spearman' (basée sur les rangs, non linéaire)."
    )
    months_win = c2.number_input(
        "Fenêtre (mois) pour la matrice/rolling", 
        value=24, 
        step=1, 
        min_value=3, 
        max_value=240, 
        key="months_win_input",
        help="Nombre de mois à utiliser pour calculer la matrice de corrélation et la corrélation glissante."
    )
    freq_used = freq
    win = months_to_periods(int(months_win), freq_used)
    st.caption(f"Fenêtre utilisée : {win} périodes ({months_win} mois, fréquence {freq_used})")

    corr_data = returns.iloc[-win:] if len(returns) >= win else returns
    corr = corr_data.corr(method="spearman" if corr_type=="Spearman" else "pearson")
    ordered = cluster_order_from_corr(corr); corr_ord = corr.loc[ordered, ordered]

    st.subheader("Matrice de corrélation (ordre clusterisé)")
    st.caption("Matrice de corrélation (-1 à +1) calculée sur la fenêtre spécifiée. Les actifs sont regroupés par similarité (clustering).")
    heat = plot_correlation_heatmap(corr_ord)
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
    asset_a = c1.selectbox("Actif A", tickers, index=0, key="corr_a")
    asset_b = c2.selectbox("Actif B", tickers, index=min(1, len(tickers)-1), key="corr_b")
    if asset_a != asset_b:
        s1, s2 = returns[asset_a], returns[asset_b]
        if corr_type == "Spearman":
            s1r, s2r = s1.rank(method="average"), s2.rank(method="average")
            roll_corr = s1r.rolling(win).corr(s2r)
        else:
            roll_corr = s1.rolling(win).corr(s2)
        figc = plot_rolling_correlation(roll_corr, asset_a, asset_b, win)
        st_plotly_chart(figc)
    else:
        st.info("Sélectionnez deux actifs distincts.")

# ============================ Onglet Backtest ============================
elif selected_tab == "⏱️ Backtest":
    
    st.subheader("Configuration du Backtest")
    n = len(tickers)
    
    try:
        prices_hash = pd.util.hash_pandas_object(prices).sum()
        returns, mu_static_bt, cov_static_bt = get_stats(
            prices_hash=prices_hash, k=k, data_freq=freq,
            mu_method=mu_method_static, cov_method=cov_method_static, 
            ewma_span_months=ewma_span_months_static, _prices=prices 
        )
    except Exception as e:
        st.error(f"Erreur lors du calcul des statistiques initiales : {e}")
        st.stop()


    # --- UI BLOC 1: Stratégie d'allocation ---
    with st.container(border=True):
        st.markdown("##### 1. Stratégie d'allocation")
        c1, c2 = st.columns(2)
        alloc_mode = c1.radio(
            "Mode d'allocation",
            ["Statique", "Dynamique (Roulante)"],
            index=1,
            horizontal=True,
            key="bt_alloc_mode",
            help="**Statique** : Utilise une seule allocation (calculée sur toute la période) et la conserve. **Dynamique (Roulante)** : Recalcule l'allocation périodiquement en utilisant uniquement les données passées (plus réaliste)."
        )
        strategy_choice = c2.selectbox(
            "Stratégie (Allocation cible)",
            ["Manuel", "Égal-pondéré (1/N)", "Max Sharpe", "Min Variance", "Risk Parity", "Max Return"], 
            index=2, 
            key="bt_strategy_choice",
            help="La stratégie utilisée pour déterminer les poids cibles."
        )
        
        lookback_months = None
        lookback_n = None
        bt_mu_method = "Moyenne historique" 
        bt_cov_method = "Ledoit-Wolf" 
        bt_ewma_span_months = 12 
            
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
            
            bt_c1, bt_c2 = st.columns(2)
            with bt_c1:
                bt_mu_method = st.selectbox(
                    "Méthode d'estimation (μ) roulante",
                    options=["Moyenne historique", "Moyenne Exponentielle (EWMA)"],
                    index=0,
                    key="bt_mu_method",
                    help="Comment le 'mu' doit-il être calculé à l'intérieur de la fenêtre roulante."
                )
            with bt_c2:
                bt_cov_method = st.selectbox(
                    "Méthode d'estimation (Cov) roulante",
                    options=["Standard", "Ledoit-Wolf"],
                    index=1,
                    key="bt_cov_method",
                    help="Comment la covariance doit-elle être calculée à l'intérieur de la fenêtre roulante."
                )

            if bt_mu_method == "EWMA":
                bt_ewma_span_months = st.number_input(
                    "Fenêtre EWMA (mois) (Roulante)", 
                    min_value=3, max_value=120, value=12, step=1, 
                    key="bt_ewma_span_dynamic",
                    help="Span pour l'EWMA (mu). Doit être inférieur ou égal à la fenêtre de calcul."
                )
                if bt_ewma_span_months > lookback_months:
                    st.warning("La fenêtre EWMA est plus grande que la fenêtre de calcul (Lookback).")
            
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

    # --- DÉTERMINATION DE LA STRATÉGIE ---
    _strategy_func = None; w_static = None
    if strategy_choice == "Manuel":
        w_static = w_manual
    elif strategy_choice == "Égal-pondéré (1/N)":
        w_static = np.ones(n) / n
    else:
        strategy_map = {
            "Max Sharpe": max_sharpe_weights,
            "Min Variance": min_var_weights,
            "Risk Parity": risk_parity_weights,
            "Max Return": max_return_weights
        }
        if alloc_mode == "Statique":
            w_ms_static, w_mv_static, w_mr_static, w_rp_static = run_static_optimization(
                mu_static_bt, cov_static_bt, min_w, max_w, 
                rf=rf_ann_mean, 
                tickers=tickers 
            )
            static_weights_map = {
                "Max Sharpe": w_ms_static,
                "Min Variance": w_mv_static,
                "Risk Parity": w_rp_static,
                "Max Return": w_mr_static
            }
            w_static = static_weights_map[strategy_choice]
        else:
            _strategy_func = strategy_map[strategy_choice]
            if lookback_n is None:
                st.error("Erreur : Mode Dynamique sélectionné mais 'lookback_n' non défini.")
                st.stop()

    # --- UI BLOC 2: Rebalancement et Frais ---
    with st.container(border=True):
        st.markdown("##### 2. Rebalancement et Frais")
        c1, c2, c3 = st.columns(3)
        
        with c2:
            fee_ter_ann_pct = st.number_input(
                "Frais de gestion ann. (TER) (%)",
                min_value=0.0, max_value=10.0, value=0.0, step=0.01, format="%.2f",
                key="bt_fee_ter",
                help="Frais de gestion annuels (Total Expense Ratio), ex: 0.25 pour 0.25%. Déduits de la performance au pro-rata de la période."
            )
        with c3:
            fee_txn_pct = st.number_input(
                "Frais de transaction (%)",
                min_value=0.0, max_value=10.0, value=0.10, step=0.01, format="%.2f",
                key="bt_fee_txn",
                help="Coût (en %) appliqué à chaque transaction (achat ou vente) lors du rebalancement. Ex: 0.10 pour 0.10%."
            )

        with c1:
            reb_opts_scan = ["Mensuel", "Trimestriel", "Annuel"]
            all_reb_opts = ["Jamais (Buy & Hold)", "Mensuel", "Trimestriel", "Annuel", "Tous les N périodes"]
            
            @st.cache_data(show_spinner="Scan des fréquences de rebalancement...")
            def _get_best_reb_freq(
                returns_hash, strategy_choice_key, alloc_mode_key, lookback_n_key, 
                min_w_key, max_w_key, fee_ter_key, fee_txn_key, k_key,
                bt_mu_method_key, bt_cov_method_key, bt_ewma_span_key, freq_key, 
                rf_ann_key, tickers_key 
            ):
                def _score_for(choice_str):
                    n_trial = n_for_rebalance_choice(choice_str, freq, n_custom=1)
                    w_static_scan, _strategy_func_scan = None, None
                    if strategy_choice_key == "Manuel": w_static_scan = w_manual
                    elif strategy_choice_key == "Égal-pondéré (1/N)": w_static_scan = np.ones(n) / n
                    else:
                        strategy_map_scan = { 
                            "Max Sharpe": max_sharpe_weights, 
                            "Min Variance": min_var_weights, 
                            "Risk Parity": risk_parity_weights,
                            "Max Return": max_return_weights
                        }
                        if alloc_mode_key == "Statique":
                            w_ms, w_mv, w_mr, w_rp = run_static_optimization(
                                mu_static_bt, cov_static_bt, min_w_key, max_w_key, rf=rf_ann_key, tickers=tickers_key
                            )
                            static_weights_map_scan = { 
                                "Max Sharpe": w_ms, "Min Variance": w_mv, "Risk Parity": w_rp,
                                "Max Return": w_mr
                            }
                            w_static_scan = static_weights_map_scan[strategy_choice_key]
                        else: 
                            _strategy_func_scan = strategy_map_scan[strategy_choice_key]
                    
                    w_trial, _, _, _, _ = run_backtest(
                        returns, tickers_key, _strategy_func_scan, w_static_scan, cov_static_bt, n_trial, 
                        lookback_n_key, min_w_key, max_w_key, fee_ter_key, fee_txn_key, k_key,
                        bt_mu_method_key, bt_cov_method_key, bt_ewma_span_key, freq_key, rf_series_aligned
                    )
                    m_trial, _ = perf_metrics(w_trial, freq_k=k_key, rf_ann=rf_ann_key)
                    s = m_trial.get("Sharpe", np.nan)
                    return s if pd.notna(s) else -np.inf

                scores = {opt: _score_for(opt) for opt in reb_opts_scan}
                return max(scores, key=scores.get) if scores else "Trimestriel"

            returns_hash = pd.util.hash_pandas_object(returns).sum()
            best_opt = _get_best_reb_freq(
                returns_hash, strategy_choice, alloc_mode, lookback_n, 
                min_w, max_w, fee_ter_ann_pct, fee_txn_pct, k,
                bt_mu_method, bt_cov_method, bt_ewma_span_months, freq,
                rf_ann_mean, tickers
            )
            
            if (strategy_choice in ["Manuel", "Égal-pondéré (1/N)"] and alloc_mode == "Statique"):
                default_index = 0; best_opt_display = "Jamais (Buy & Hold)"
            else:
                default_index = all_reb_opts.index(best_opt) if best_opt in all_reb_opts else 1
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
        wealth, weights_history, target_weights_log, turnover_log, risk_contrib_log = run_backtest(
            returns=returns,
            tickers=tickers, 
            _strategy_func=_strategy_func, 
            w_static=w_static,
            cov_static=cov_static_bt,
            rebalance_freq_n=n_reb,
            lookback_window_n=lookback_n,
            min_w=min_w,
            max_w=max_w,
            fee_ter_ann_pct=fee_ter_ann_pct,
            fee_txn_pct=fee_txn_pct,
            k=k,
            mu_method=bt_mu_method,
            cov_method=bt_cov_method, 
            ewma_span_months=bt_ewma_span_months,
            data_freq=freq,
            rf_series=rf_series_aligned 
        )
    except Exception as e:
        st.error(f"Erreur lors de l'exécution du backtest : {e}")
        st.exception(e) 
        st.stop()

    # --- [REVERSION] Logique d'inflation supprimée ---
    # Les métriques sont calculées sur 'wealth' (nominal) et 'rf_ann_mean' (nominal)
    metrics, dd = perf_metrics(wealth, freq_k=k, rf_ann=rf_ann_mean) 
    ret_series = wealth.pct_change().dropna()
    
    st.divider()
    st.subheader(f"Métriques de performance du backtest (Rf = {rf_ann_mean:.2%})")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("CAGR (Net de frais)", f"{metrics['CAGR']*100:.2f}%" if pd.notna(metrics["CAGR"]) else "N/A", help="Compound Annual Growth Rate : Le taux de croissance annuel composé (net de frais de gestion et de transaction).")
    kpi2.metric("Volatilité Ann.", f"{metrics['Vol ann.']*100:.2f}%" if pd.notna(metrics["Vol ann."]) else "N/A", help="Volatilité (écart-type) annualisée des rendements du portefeuille. C'est la mesure standard du risque.")
    kpi3.metric("Ratio de Sharpe (Net)", f"{metrics['Sharpe']:.2f}" if pd.notna(metrics["Sharpe"]) else "N/A", help=f"CAGR (Net) moins le taux sans risque ({rf_ann_mean:.2%}), divisé par la Volatilité Annuelle. Mesure le rendement ajusté au risque.")
    kpi4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%" if pd.notna(metrics["Max Drawdown"]) else "N/A", delta_color="inverse", help="La plus grande perte (en pourcentage) depuis un pic jusqu'à un creux subséquent durant la période de backtest.")
    
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    kpi5.metric("Ratio de Sortino (Net)", f"{metrics['Sortino']:.2f}" if pd.notna(metrics["Sortino"]) else "N/A", help=f"Similaire au Sharpe, mais ne pénalise que la volatilité négative (ne tient compte que des rendements inférieurs au Rf de {rf_ann_mean:.2%}).")
    kpi6.metric("Ratio de Calmar (Net)", f"{metrics['Calmar']:.2f}" if pd.notna(metrics["Calmar"]) else "N/A", help=f"CAGR (Net) moins le taux sans risque ({rf_ann_mean:.2%}), divisé par la valeur absolue du Max Drawdown.")
    
    turnover_moyen = np.mean(turnover_log) if turnover_log else 0.0
    kpi7.metric(
        "Rotation Moyenne (Turnover)", 
        f"{turnover_moyen*100:.2f}%", 
        help="Le pourcentage moyen du portefeuille qui est vendu/acheté à chaque rebalancement. Un chiffre élevé implique des coûts de transaction et un 'frottement' plus importants."
    )
    
    kpi8.metric("Worst Period", f"{metrics['Worst period']*100:.2f}%" if pd.notna(metrics["Worst period"]) else "N/A", delta_color="inverse", help=f"La pire performance sur une seule période (ici, la pire fréquence : '{freq}').")

    # --- Benchmark section ---
    bench_returns = None
    bench_wealth = None
    bench_dd = None
    st.divider()
    st.subheader("Comparaison Benchmark")
    bench_c1, bench_c2 = st.columns([1,1])
    bench_ticker = st.text_input(
        "Ticker Benchmark", 
        value="^GSPC", 
        key="benchmark_ticker_input", 
        help="Ticker Yahoo Finance (ex: ^GSPC pour S&P 500, 'IEUR.AS' pour un ETF) à utiliser comme référence."
    )
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
                    rel_metrics = calculate_relative_metrics(port_returns_common, bench_returns_common, k, rf_ann=rf_ann_mean) 
                    
                    rkpi1, rkpi2, rkpi3, rkpi4 = st.columns(4)
                    rkpi1.metric("Beta", f"{rel_metrics['Beta']:.2f}" if pd.notna(rel_metrics["Beta"]) else "N/A", help="Mesure de la volatilité du portefeuille par rapport au benchmark. Beta > 1 = plus volatil. Beta < 1 = moins volatil.")
                    rkpi2.metric("Alpha (ann.)", f"{rel_metrics['Alpha (ann.)']*100:.2f}%" if pd.notna(rel_metrics["Alpha (ann.)"]) else "N/A", help=f"Surperformance annualisée ajustée du risque (Beta) et du taux sans risque ({rf_ann_mean:.2%}). C'est le 'vrai' gain du gérant.")
                    rkpi3.metric("Tracking Error", f"{rel_metrics['Tracking Error']*100:.2f}%" if pd.notna(rel_metrics["Tracking Error"]) else "N/A", help="Volatilité (écart-type) de la *différence* de rendement entre le portefeuille et le benchmark.")
                    rkpi4.metric("Information Ratio", f"{rel_metrics['Information Ratio']:.2f}" if pd.notna(rel_metrics["Information Ratio"]) else "N/A", help="Surperformance (Portefeuille - Benchmark) divisée par la Tracking Error. Mesure la constance de la surperformance.")
                    
                    bench_wealth = (1 + bench_returns).cumprod()
                    # [REVERSION] Logique d'inflation benchmark supprimée
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
                        
                        rkpi5, rkpi6, rkpi7, _ = st.columns(4) 
                        rkpi5.metric("Batting Average", f"{regime_metrics.get('Batting Average', np.nan)*100:.1f}%", help="Pourcentage de périodes (mois) où le portefeuille a surperformé le benchmark.")
                        rkpi6.metric("Up-Capture Ratio", f"{regime_metrics.get('Up-Capture', np.nan):.1f}%", help="Pourcentage de la performance du benchmark capturée par le portefeuille pendant les mois de *hausse* du benchmark. >100% = surperformance en hausse.")
                        rkpi7.metric("Down-Capture Ratio", f"{regime_metrics.get('Down-Capture', np.nan):.1f}%", help="Pourcentage de la performance du benchmark subie par le portefeuille pendant les mois de *baisse* du benchmark. <100% = protection en baisse.")
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
    
    figw = plot_backtest_wealth(wealth_plot, bench_wealth, log_scale, bench_ticker)
    figw.data[0].name = "Portefeuille (Net)" # [REVERSION] Label fixe
    if bench_wealth is not None:
        figw.data[1].name = f"Benchmark ({bench_ticker})" # [REVERSION] Label fixe
    figw.update_layout(yaxis_title="Valeur du portefeuille (Nette de frais)") # [REVERSION] Titre fixe
    st_plotly_chart(figw)
    
    st.divider()
    figd = plot_backtest_drawdown(dd, bench_dd, bench_ticker)
    figd.data[0].name = "Drawdown Portefeuille" # [REVERSION] Label fixe
    if bench_dd is not None:
        figd.data[1].name = f"Drawdown Benchmark ({bench_ticker})" # [REVERSION] Label fixe
    st_plotly_chart(figd)

    st.divider()
    st.subheader("Rendements Annuels")
    
    view_mode_annual = st.radio(
        "Changer la vue du graphique :",
        ["Portefeuille vs. Benchmark", "Actifs Individuels"],
        horizontal=True,
        label_visibility="collapsed",
        key="annual_view_mode"
    )

    try:
        port_annual_ret = (1 + ret_series).resample("YE").prod() - 1
        port_annual_ret.index = port_annual_ret.index.year
        port_annual_ret.name = "Portefeuille (Net)" # [REVERSION] Label fixe
        
        assets_annual_ret_df = (1 + returns).resample("YE").prod() - 1
        assets_annual_ret_df.index = assets_annual_ret_df.index.year
            
        fig_annual = plot_annual_returns(port_annual_ret, bench_annual_ret, assets_annual_ret_df, view_mode_annual)
        st_plotly_chart(fig_annual)

    except Exception as e_annual:
        st.warning(f"Impossible de générer le graphique des rendements annuels : {e_annual}")
    
    st.divider()
    st.subheader("Calendrier des rendements mensuels (Heatmap)")
    st.caption("Visualisation des rendements mensuels (nets de frais) pour chaque mois et chaque année. Vert = positif, Rouge = négatif.")
    heatmap_df = create_monthly_heatmap_df(monthly_ret)
    if not heatmap_df.empty:
        fig_heat = plot_monthly_heatmap(monthly_ret, heatmap_df)
        st_plotly_chart(fig_heat)
    else:
        st.info("Pas assez de données pour générer la heatmap mensuelle (nécessite au moins 1 mois).")
    
    st.subheader("Évolution de l'allocation du portefeuille")
    st.caption(
        "Ce graphique montre l'évolution de la valeur relative de chaque actif dans le portefeuille. "
        "En mode 'Buy & Hold', il montre la **dérive**. "
        "En mode 'Rebalancé' ou 'Dynamique', il montre les **réajustements périodiques** (les 'sauts') vers les poids cibles (visibles dans le tableau ci-dessous)."
    )
    
    fig_alloc = plot_allocation_history(weights_history)
    st_plotly_chart(fig_alloc)

    if risk_contrib_log and len(risk_contrib_log) > 1:
        st.subheader("Évolution de la Contribution au Risque")
        st.caption(
            "Ce graphique montre la part de risque de chaque actif (en %) à chaque rebalancement. "
            "Pour 'Risk Parity', les barres devraient être égales. "
            "Pour 'Min Variance', un seul actif peut dominer."
        )
        fig_rc = plot_risk_contrib_history(risk_contrib_log)
        st_plotly_chart(fig_rc)
    
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
                    if i < len(entry['Weights']):
                        row[ticker] = f"{entry['Weights'][i]*100:.2f}%"
                    else:
                        pass 
                log_data.append(row)
            
            log_df = pd.DataFrame(log_data).set_index("Date")
            st.dataframe(log_df, width='stretch')
    
    st.subheader("Analyse détaillée des Drawdowns")
    top_n_dd = st.number_input(
        "Afficher le Top N des pires drawdowns", 
        min_value=1, max_value=20, value=5, step=1, 
        key="topn_dd_input", 
        help="Classe les pires 'drawdowns' (pertes pic-à-creux) et affiche les N plus importants."
    )
    
    with st.spinner("Analyse des périodes de drawdown..."):
        dd_table = find_drawdowns(wealth, top_n=top_n_dd) 
        if dd_table.empty:
            st.info("Aucun drawdown enregistré (performance toujours positive).")
        else:
            dd_table_fmt = dd_table.copy()
            
            dd_table_fmt["Max Drawdown"] = dd_table_fmt["Max Drawdown"].map(lambda x: f"{x*100:.2f}%")
            
            dd_table_fmt["Peak Date"] = dd_table_fmt["Peak Date"].astype(str)
            dd_table_fmt["Trough Date"] = dd_table_fmt["Trough Date"].astype(str)
            dd_table_fmt["Recovery Date"] = dd_table_fmt["Recovery Date"].astype(str).replace("NaT", "En cours")
            dd_table_fmt["Total Duration (Days)"] = dd_table_fmt["Total Duration (Days)"].fillna("-").astype(str)
            dd_table_fmt["Peak-to-Trough (Days)"] = dd_table_fmt["Peak-to-Trough (Days)"].astype(str)
            
            st.dataframe(dd_table_fmt, width='stretch', hide_index=True)
    
    st.subheader("Analyses additionnelles")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Volatilité glissante (annualisée)**")
        roll_window = k 
        st.caption(f"Affiche la volatilité annualisée du portefeuille sur une fenêtre glissante de {roll_window} périodes (environ 1 an de données {freq}).")
        if len(ret_series) > roll_window:
            rolling_vol = ret_series.rolling(window=roll_window).std(ddof=1) * np.sqrt(k)
            fig_roll_vol = plot_rolling_volatility(rolling_vol)
            st_plotly_chart(fig_roll_vol)
        else:
            st.info(f"Pas assez de données pour une fenêtre glissante de {roll_window} périodes.")
    with c4:
        st.markdown("**Distribution des rendements mensuels**")
        st.caption("Histogramme montrant la distribution (fréquence) des rendements mensuels du portefeuille (nets de frais).")
        if len(monthly_ret) > 1:
            fig_hist_ret_monthly = plot_monthly_histogram(monthly_ret)
            st_plotly_chart(fig_hist_ret_monthly)
        else:
            st.info("Pas assez de données pour l'histogramme mensuel.")