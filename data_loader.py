# data_loader.py
import numpy as np
import pandas as pd
import streamlit as st
from utils import prices_to_returns, build_availability_from_union

# Yahoo Finance
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

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
    """
    [MODIFIÉ] Simplifié pour être plus robuste.
    Télécharge les données, extrait 'Close', et utilise build_availability_from_union.
    """
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
    
    # Extraire 'Close' de manière robuste, que ce soit un MultiIndex ou non
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    if isinstance(df.columns, pd.MultiIndex):
        # Cas 1: MultiIndex (plusieurs tickers)
        # Sélectionner (Ticker, 'Close') et renommer la colonne en Ticker
        prices_union = pd.concat([
            df[(t, 'Close')].rename(t) for t in tickers_up if (t, 'Close') in df.columns
        ], axis=1)
    elif len(tickers_up) == 1 and 'Close' in df.columns:
        # Cas 2: Index simple (un seul ticker)
        prices_union = df[['Close']].rename(columns={'Close': tickers_up[0]})
    else:
        # Cas 3: Index simple (plusieurs tickers, mais pas de group_by, ex: erreur)
        # Tente de trouver les tickers directement
        prices_union = df[[t for t in tickers_up if t in df.columns]]

    if prices_union.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Utiliser la fonction de utils.py pour l'analyse de disponibilité
    availability_df = build_availability_from_union(prices_union)
    
    # L'intersection est simplement les lignes sans aucun NaN
    prices_intersection = prices_union.dropna(how="any").sort_index()
    
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

@st.cache_data(show_spinner="Chargement du taux sans risque...")
def fetch_risk_free_rate(ticker: str, start_date, end_date) -> pd.Series:
    """
    Charge le taux sans risque (ex: ^IRX) et le retourne en décimal (ex: 5.25 -> 0.0525).
    Les données de YFinance pour les taux sont en pourcentage.
    """
    if not ticker:
        return pd.Series(dtype=float)
        
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        st.warning(f"Impossible de télécharger les données du taux sans risque {ticker}.")
        return pd.Series(dtype=float)
    
    rf_series = df['Close']
    
    # --- [CORRECTION] ---
    if isinstance(rf_series, pd.DataFrame):
        st.warning(f"Le ticker {ticker} a renvoyé un DataFrame pour 'Close'. Utilisation de la première colonne.")
        rf_series = rf_series.iloc[:, 0]
    # --- [FIN CORRECTION] ---

    rf_series = rf_series / 100.0
    rf_series.name = "RF_ANNUAL"
    
    return rf_series.ffill().dropna()