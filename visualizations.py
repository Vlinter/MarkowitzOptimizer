# visualizations.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Plotly helper ----
def st_plotly_chart(fig: go.Figure) -> None:
    fig.update_layout(autosize=True, margin=dict(l=40, r=20, t=40, b=40))
    plotly_config = {"displaylogo": False, "scrollZoom": True, "responsive": True}
    st.plotly_chart(fig, config=plotly_config, theme=None, use_container_width=True)

# ---- Fonctions de création de graphiques ----

def plot_price_history(prices: pd.DataFrame, rebase: bool, log_scale: bool) -> go.Figure:
    plot_df = prices.copy()
    if rebase:
        first_prices = plot_df.iloc[0]
        safe_first_prices = first_prices.replace(0, np.nan)
        plot_df = (plot_df / safe_first_prices) * 100.0
        plot_df = plot_df.fillna(100.0)

    fig = go.Figure()
    for col in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df[col].values,
            mode="lines",
            name=col,
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{'Index' if rebase else 'Prix'} {col}: %{{y:.2f}}<extra></extra>"
        ))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title=("Index (base=100)" if rebase else "Prix"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(rangeslider_visible=True)
    if log_scale:
        fig.update_yaxes(type="log")
    return fig

def plot_asset_histogram(returns_series: pd.Series, ticker_name: str, freq: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns_series, 
        name="Rendements", 
        nbinsx=40, 
        histnorm='probability density', 
        marker_color='cornflowerblue'
    ))
    fig.update_layout(
        template="plotly_dark", 
        yaxis_title="Densité", 
        xaxis_title=f"Rendement ({freq})", 
        barmode="overlay",
        title=f"Distribution des rendements ({freq}) pour {ticker_name}"
    )
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(tickformat=".1%")
    return fig

def plot_efficient_frontier(vols_mc, rets_mc, front, port_points: list) -> go.Figure:
    fig = go.Figure()
    # Monte Carlo
    fig.add_trace(go.Scatter(
        x=vols_mc, y=rets_mc, mode="markers",
        marker=dict(size=4, opacity=0.35, color="#0A84FF"), 
        name="Portefeuilles aléatoires (bornés)",
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
    ))
    # Frontière
    if front.size > 0:
        fig.add_trace(go.Scatter(
            x=front[:,1], y=front[:,0], mode="lines",
            name="Frontière efficiente",
            line=dict(width=3, color="white"), 
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
        ))
    # Points optimaux
    for name, r, v, sym, c in port_points:
        fig.add_trace(go.Scatter(
            x=[v], y=[r], mode="markers", name=name,
            marker=dict(size=12, symbol=sym, color=c, line=dict(width=1, color="black")),
            hovertemplate=f"{name}<br>Vol: %{{x:.2%}}<br>Ret: %{{y:.2%}}<extra></extra>"
        ))
        
    fig.update_layout(
        xaxis_title="Volatilité (ann.)", yaxis_title="Rendement (ann.)",
        template="plotly_dark", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        title="Nuage de portefeuilles et frontière efficiente (bornée)"
    )
    fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
    return fig

def plot_allocation_pies(w_ms, w_mv, w_rp, w_mr, tickers) -> go.Figure:
    """
    [MODIFIÉ] HRP a été supprimé. La grille est maintenant de 1x4.
    """
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
    s_rp = pie_series(w_rp, tickers, threshold=0.02)
    s_mr = pie_series(w_mr, tickers, threshold=0.02)

    pies = make_subplots(rows=1, cols=4, specs=[[{"type": "domain"}]*4],
                         subplot_titles=("Max Sharpe", "Min Variance", "Risk Parity (ERC)", "Max Return"))
    
    def add_pie(fig_, r, c, s: pd.Series):
        if len(s) == 1:
            fig_.add_trace(go.Pie(labels=[s.index[0]], values=[1.0], hole=0.35,
                                  sort=False, textinfo="label+percent", textposition="inside", showlegend=False), r, c)
        else:
            fig_.add_trace(go.Pie(labels=s.index.tolist(), values=s.values.tolist(), hole=0.35,
                                  sort=False, textinfo="percent+label"), r, c)
    
    add_pie(pies, 1, 1, s_ms); add_pie(pies, 1, 2, s_mv); add_pie(pies, 1, 3, s_rp); 
    add_pie(pies, 1, 4, s_mr)
    
    pies.update_layout(template="plotly_dark", showlegend=False)
    return pies

def plot_correlation_heatmap(corr_ord: pd.DataFrame) -> go.Figure:
    heat = go.Figure(data=go.Heatmap(
        z=corr_ord.values, x=corr_ord.columns, y=corr_ord.index,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0, colorbar=dict(title="Corr")
    ))
    heat.update_layout(template="plotly_dark", height=600, xaxis_showgrid=False, yaxis_showgrid=False,
                       title="Matrice de corrélation (ordre clusterisé)")
    return heat

def plot_rolling_correlation(roll_corr: pd.Series, asset_a: str, asset_b: str, win: int) -> go.Figure:
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
    return figc

def plot_backtest_wealth(wealth: pd.Series, bench_wealth: pd.Series, log_scale: bool, bench_ticker: str) -> go.Figure:
    wealth_plot = wealth.where(wealth > 0, np.nan) if log_scale else wealth
    figw = go.Figure()
    figw.add_trace(go.Scatter(x=wealth_plot.index, y=wealth_plot.values, mode="lines", name="Portefeuille", line=dict(color="#0A84FF")))
    if bench_wealth is not None and not bench_wealth.empty:
        figw.add_trace(go.Scatter(x=bench_wealth.index, y=bench_wealth.values, mode="lines", name=f"Benchmark ({bench_ticker})", line=dict(color='gray', dash='dash')))
    figw.update_layout(
        template="plotly_dark", 
        yaxis_title="Valeur du portefeuille (Nette de frais)", 
        xaxis_title="Date", 
        legend=dict(x=0.01, y=0.99),
        title="Valeur du portefeuille (Wealth)"
    )
    if log_scale: figw.update_yaxes(type="log")
    return figw

def plot_backtest_drawdown(dd: pd.Series, bench_dd: pd.Series, bench_ticker: str) -> go.Figure:
    figd = go.Figure()
    figd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown Portefeuille", fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='crimson')))
    if bench_dd is not None and not bench_dd.dropna().empty:
        figd.add_trace(go.Scatter(
            x=bench_dd.index, y=bench_dd.values, mode="lines",
            name=f"Drawdown Benchmark ({bench_ticker})",
            line=dict(color='steelblue', dash='dot'),
            fill='tozeroy', fillcolor='rgba(70,130,180,0.15)'
        ))
    figd.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    figd.update_layout(
        template="plotly_dark", 
        yaxis_title="Drawdown", 
        xaxis_title="Date", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), 
        margin=dict(t=80),
        title="Évolution du Drawdown"
    )
    figd.update_yaxes(tickformat=".0%")
    return figd

def plot_annual_returns(port_annual_ret, bench_annual_ret, assets_annual_ret_df, view_mode) -> go.Figure:
    fig_annual = go.Figure()
        
    if view_mode == "Portefeuille vs. Benchmark":
        all_series_to_plot = {port_annual_ret.name: port_annual_ret}
        if bench_annual_ret is not None and not bench_annual_ret.empty:
            all_series_to_plot[bench_annual_ret.name] = bench_annual_ret

        all_indices = [s.index for s in all_series_to_plot.values()]
        
        all_years = pd.Index([])
        if all_indices:
            all_years = all_indices[0] 
            if len(all_indices) > 1:
                for idx in all_indices[1:]:
                    all_years = all_years.union(idx)
        all_years = all_years.sort_values()

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
        # VUE 2 : Actifs Individuels
        all_years = assets_annual_ret_df.index
        for asset in assets_annual_ret_df.columns:
            fig_annual.add_trace(go.Bar(
                x=all_years, 
                y=assets_annual_ret_df[asset], 
                name=asset,
                hovertemplate="Année: %{x}<br>Rendement: %{y:.2%}<extra></extra>"
            ))
        fig_annual.update_layout(title_text="Rendements Annuels (Actifs Individuels)")

    # Paramètres communs
    fig_annual.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig_annual.update_layout(
        template="plotly_dark", barmode='group', yaxis_title="Rendement Annuel",
        xaxis_title="Année", yaxis_tickformat=".0%",
        xaxis=dict(tickmode='linear', dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig_annual

def plot_monthly_heatmap(monthly_ret: pd.Series, heatmap_df: pd.DataFrame) -> go.Figure:
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
    fig_heat.update_layout(
        template="plotly_dark", 
        yaxis_title="Année", 
        xaxis_title="Mois", 
        xaxis=dict(showgrid=False, side="top"), 
        yaxis=dict(tickmode='linear', showgrid=False),
        title="Calendrier des rendements mensuels (Heatmap)"
    )
    return fig_heat

def plot_allocation_history(weights_history: pd.DataFrame) -> go.Figure:
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
    fig_alloc.update_layout(
        template="plotly_dark", 
        yaxis_title="Allocation (%)", 
        xaxis_title="Date", 
        hovermode="x unified", 
        yaxis_tickformat=".0%", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Évolution de l'allocation du portefeuille"
    )
    return fig_alloc

def plot_rolling_volatility(rolling_vol: pd.Series) -> go.Figure:
    fig_roll_vol = go.Figure()
    fig_roll_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode="lines", name="Volatilité glissante", line=dict(color="tomato")))
    fig_roll_vol.update_layout(
        template="plotly_dark", 
        yaxis_title="Volatilité Ann.", 
        xaxis_title="Date",
        title="Volatilité glissante (annualisée)"
    )
    fig_roll_vol.update_yaxes(tickformat=".0%")
    return fig_roll_vol

def plot_monthly_histogram(monthly_ret: pd.Series) -> go.Figure:
    fig_hist_ret = go.Figure()
    fig_hist_ret.add_trace(go.Histogram(x=monthly_ret, name="Rendements Mensuels", nbinsx=30, histnorm='probability density', marker_color='cornflowerblue'))
    fig_hist_ret.update_layout(
        template="plotly_dark", 
        yaxis_title="Densité", 
        xaxis_title="Rendement Mensuel", 
        barmode="overlay",
        title="Distribution des rendements mensuels"
    )
    fig_hist_ret.update_traces(opacity=0.75)
    fig_hist_ret.update_xaxes(tickformat=".1%")
    return fig_hist_ret

# --- NOUVELLE FONCTION ---
def plot_risk_contrib_history(risk_contrib_log: list) -> go.Figure:
    """
    Crée un graphique en barres empilées de la contribution au risque à chaque rebalancement.
    """
    if not risk_contrib_log:
        return go.Figure()

    dates = [entry['Date'] for entry in risk_contrib_log]
    tickers = risk_contrib_log[0]['Tickers']
    rc_data = {ticker: [] for ticker in tickers}
    
    for entry in risk_contrib_log:
        # Gérer le cas où les tickers pourraient changer
        entry_rc_map = {ticker: rc for ticker, rc in zip(entry.get('Tickers', tickers), entry['RC'])}
        for ticker in tickers:
            rc_data[ticker].append(entry_rc_map.get(ticker, 0.0))
            
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Bar(
            x=dates,
            y=np.array(rc_data[ticker]),
            name=ticker,
            hovertemplate=f"Date: %{{x|%Y-%m-%d}}<br>{ticker} RC: %{{y:.1%}}<extra></extra>"
        ))
        
    fig.update_layout(
        barmode='stack',
        template="plotly_dark", 
        yaxis_title="Contribution au Risque (%)", 
        xaxis_title="Date de Rebalancement", 
        hovermode="x unified", 
        yaxis_tickformat=".0%", 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Contribution au Risque du Portefeuille (aux dates de rebalancement)"
    )
    # Assurer que la somme est 100%
    fig.update_yaxes(range=[0, 1])
    return fig