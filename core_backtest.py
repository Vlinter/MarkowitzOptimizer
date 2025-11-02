# core_backtest.py
import numpy as np
import pandas as pd
import streamlit as st
from core_optimizer import feasible_start, risk_contrib, max_sharpe_weights
from utils import near_psd_clip, ridge_regularize, months_to_periods

# Optionnel: Ledoit–Wolf (scikit-learn)
try:
    from sklearn.covariance import LedoitWolf
    HAS_LW = True
except Exception:
    HAS_LW = False

# === MOTEUR DE BACKTEST ===
@st.cache_data(show_spinner="Exécution du backtest...")
def run_backtest(
    returns: pd.DataFrame, 
    tickers: list,
    _strategy_func: callable, 
    w_static: np.ndarray, 
    cov_static: np.ndarray,
    rebalance_freq_n: int, 
    lookback_window_n: int, 
    min_w: float, 
    max_w: float, 
    fee_ter_ann_pct: float, 
    fee_txn_pct: float, 
    k: int,
    mu_method: str,
    cov_method: str,       
    ewma_span_months: int,  
    data_freq: str,
    rf_series: pd.Series    
) -> tuple[pd.Series, pd.DataFrame, list, list, list]: 
    """
    Moteur de backtest unifié.
    RETOURNE : wealth, weights_history, target_weights_log, turnover_log, risk_contrib_log
    """
    n = returns.shape[1]
    
    # --- Initialisation ---
    if w_static is not None:
        w_target = w_static / w_static.sum()
    else:
        w_target = feasible_start(n, min_w, max_w) 
    
    w_cur = w_target.copy()
    values = np.empty(len(returns), dtype=float)
    weights_history = []
    target_weights_log = [] 
    turnover_log = []       
    risk_contrib_log = []   
    port_val = 1.0
    
    # Log initial
    target_weights_log.append({ "Date": returns.index[0], "Weights": w_target })
    
    initial_rc = risk_contrib(w_target, cov_static) 
    risk_contrib_log.append({ "Date": returns.index[0], "RC": initial_rc, "Tickers": tickers })


    # --- Frais ---
    fee_ter_periodic = (fee_ter_ann_pct / 100.0) / k
    fee_txn = fee_txn_pct / 100.0
    span_periods = months_to_periods(ewma_span_months, data_freq)
    min_p_ewma = max(1, int(span_periods / 2)) 

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
                    
                    # --- Calcul MU Dynamique (le choix est conservé) ---
                    if mu_method == "Moyenne Exponentielle (EWMA)":
                        mu_slice = returns_slice.ewm(span=span_periods, min_periods=min_p_ewma).mean().iloc[-1].values * k
                    else: # "Moyenne historique"
                        mu_slice = returns_slice.mean().values * k
                    
                    # --- Calcul COV Dynamique ([MODIFIÉ] EWMA supprimé) ---
                    if cov_method == "Ledoit-Wolf" and HAS_LW:
                        lw = LedoitWolf().fit(returns_slice.values); cov_raw_slice = lw.covariance_ * k
                    else: # "Standard"
                        cov_raw_slice = returns_slice.cov().values * k
                        
                    # --- [CORRECTION] ---
                    # Nettoyer les NaN/Inf qui peuvent provenir de ewm.cov() sur de courtes périodes
                    cov_raw_slice = np.nan_to_num(cov_raw_slice, nan=0.0, posinf=0.0, neginf=0.0)
                    # --- [FIN CORRECTION] ---

                    cov_slice = near_psd_clip(ridge_regularize(cov_raw_slice, ridge=1e-5))
                    
                    # --- Calcul RF Dynamique ---
                    rf_t = 0.0
                    if not rf_series.empty:
                        try:
                            rf_t = rf_series.loc[:returns_slice.index[-1]].iloc[-1]
                            if pd.isna(rf_t): rf_t = 0.0 
                        except IndexError:
                            rf_t = 0.0 
                    
                    try:
                        # [MODIFIÉ] HRP n'est plus une option valide pour _strategy_func
                        w_target = _strategy_func(mu_slice, cov_slice, min_w, max_w, rf=rf_t, tickers=tickers)
                    except Exception as e:
                        pass 
            
            else:
                # Mode Statique
                cov_slice = cov_static 
                pass 

            target_weights_log.append({ "Date": returns.index[t], "Weights": w_target })

            # --- 1b. Appliquer les Frais de Transaction ---
            turnover = np.sum(np.abs(w_target - w_cur)) / 2.0 
            turnover_log.append(turnover) 
            txn_cost = turnover * fee_txn
            port_val *= (1.0 - txn_cost) 
            
            # --- 1c. Mettre à jour les poids ---
            w_cur = w_target.copy()
            
            # --- 1d. Log Contrib. Risque ---
            if 'cov_slice' in locals(): 
                rc = risk_contrib(w_target, cov_slice)
                risk_contrib_log.append({ "Date": returns.index[t], "RC": rc, "Tickers": tickers })


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
    if not wealth.empty:
        wealth.iloc[0] = 1.0 
    weights_df = pd.DataFrame(weights_history, index=returns.index, columns=returns.columns)
    
    return wealth, weights_df, target_weights_log, turnover_log, risk_contrib_log
# === FIN DU MOTEUR DE BACKTEST ===


# === Fonctions de Métriques ===
def perf_metrics(wealth: pd.Series, freq_k: int, rf_ann: float = 0.0):
    ret = wealth.pct_change().dropna()
    if ret.empty: return {"Total Return": 0, "CAGR": np.nan, "Vol ann.": np.nan, "Sharpe": np.nan, "Max Drawdown": 0, "Worst period": np.nan, "Sortino": np.nan, "Calmar": np.nan}, pd.Series([0.0], index=wealth.index)
    
    total_return = wealth.iloc[-1] - 1.0
    years = len(ret) / freq_k if freq_k > 0 else np.nan
    cagr = wealth.iloc[-1] ** (1/years) - 1 if years and years > 0 and wealth.iloc[-1] > 0 else np.nan
    vol_ann = ret.std(ddof=1) * np.sqrt(freq_k)
    
    cagr_excess = cagr - rf_ann
    sharpe = cagr_excess / vol_ann if vol_ann > 0 else np.nan
    
    rf_periodic = (1 + rf_ann)**(1/freq_k) - 1
    negative_returns = ret[ret < rf_periodic].dropna()
    downside_std = negative_returns.std(ddof=1) * np.sqrt(freq_k)
    
    sortino = cagr_excess / downside_std if downside_std > 0 else np.nan
    
    cummax = wealth.cummax(); dd = wealth / cummax - 1.0
    max_dd = dd.min(); worst = ret.min()
    
    calmar = cagr_excess / abs(max_dd) if max_dd < 0 and pd.notna(cagr) else np.nan
    
    return {"Total Return": total_return, "CAGR": cagr, "Vol ann.": vol_ann,
            "Sharpe": sharpe, "Max Drawdown": max_dd, "Worst period": worst,
            "Sortino": sortino, "Calmar": calmar}, dd

def calculate_relative_metrics(port_returns: pd.Series, bench_returns: pd.Series, freq_k: int, rf_ann: float = 0.0) -> dict:
    df = pd.DataFrame({'port': port_returns, 'bench': bench_returns}).dropna()
    if len(df) < 2: return {"Beta": np.nan, "Alpha (ann.)": np.nan, "Tracking Error": np.nan, "Information Ratio": np.nan}
    
    cov_matrix = df.cov() * freq_k
    covariance = cov_matrix.iloc[0, 1]
    bench_var = df['bench'].var() * freq_k
    beta = covariance / bench_var if bench_var > 0 else np.nan
    
    years = len(df) / freq_k
    port_cagr = (1 + df['port']).prod()**(1/years) - 1 if years > 0 and (1 + df['port']).prod() > 0 else np.nan
    bench_cagr = (1 + df['bench']).prod()**(1/years) - 1 if years > 0 and (1 + df['bench']).prod() > 0 else np.nan
    
    alpha = (port_cagr - rf_ann) - (beta * (bench_cagr - rf_ann)) if pd.notna(port_cagr) and pd.notna(bench_cagr) and pd.notna(beta) else np.nan
    
    diff_ret = df['port'] - df['bench']
    tracking_error = diff_ret.std(ddof=1) * np.sqrt(freq_k)
    information_ratio = (port_cagr - bench_cagr) / tracking_error if tracking_error > 0 and pd.notna(port_cagr) and pd.notna(bench_cagr) else np.nan
    
    return {"Beta": beta, "Alpha (ann.)": alpha, "Tracking Error": tracking_error, "Information Ratio": information_ratio}

def calculate_market_regime_metrics(port_returns_monthly: pd.Series, bench_returns_monthly: pd.Series) -> dict:
    df = pd.DataFrame({'port': port_returns_monthly, 'bench': bench_returns_monthly}).dropna()
    if df.empty or len(df) < 2: return {"Batting Average": np.nan, "Up-Capture": np.nan, "Down-Capture": np.nan}
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
    if returns.empty: return {"VaR": np.nan, "CVaR": np.nan}
    var = returns.quantile(1.0 - percentile)
    cvar = returns[returns <= var].mean()
    return {"VaR": var, "CVaR": cvar}

def create_monthly_heatmap_df(monthly_returns: pd.Series) -> pd.DataFrame:
    if monthly_returns.empty: return pd.DataFrame()
    df = pd.DataFrame(monthly_returns); df.columns = ['Return']
    df['Year'] = df.index.year; df['Month'] = df.index.strftime('%b')
    heatmap_df = df.pivot(index='Year', columns='Month', values='Return')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    available_cols = [m for m in month_order if m in heatmap_df.columns]
    return heatmap_df[available_cols]

def find_drawdowns(wealth: pd.Series, top_n: int = 5) -> pd.DataFrame:
    drawdowns = []
    peak_date = wealth.index[0]; peak_value = wealth.iloc[0]
    in_drawdown = False; trough_date = peak_date; trough_value = peak_value
    for t, v in wealth.iloc[1:].items():
        if v > peak_value:
            if in_drawdown:
                recovery_date = t; max_dd = (trough_value / peak_value) - 1.0
                p_to_t_days = (trough_date - peak_date).days
                total_duration_days = (recovery_date - peak_date).days
                drawdowns.append({"Peak Date": peak_date.date(), "Trough Date": trough_date.date(), "Recovery Date": recovery_date.date(), "Max Drawdown": max_dd, "Peak-to-Trough (Days)": p_to_t_days, "Total Duration (Days)": total_duration_days})
            peak_date = t; peak_value = v; trough_date = t; trough_value = v; in_drawdown = False
        elif v < trough_value:
            in_drawdown = True; trough_date = t; trough_value = v
    if in_drawdown and trough_value < peak_value:
        max_dd = (trough_value / peak_value) - 1.0; p_to_t_days = (trough_date - peak_date).days
        drawdowns.append({"Peak Date": peak_date.date(), "Trough Date": trough_date.date(), "Recovery Date": pd.NaT, "Max Drawdown": max_dd, "Peak-to-Trough (Days)": p_to_t_days, "Total Duration (Days)": np.nan})
    if not drawdowns: return pd.DataFrame(columns=["Peak Date", "Trough Date", "Recovery Date", "Max Drawdown", "Peak-to-Trough (Days)", "Total Duration (Days)"])
    df = pd.DataFrame(drawdowns); df = df.sort_values(by="Max Drawdown", ascending=True)
    return df.head(top_n)