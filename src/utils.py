from yahooquery import Ticker
import pandas as pd
import numpy as np

def load_data_with_indicators(ticker, start_date, end_date):
    """
    Descarga datos con YahooQuery y asegura exactamente 12 columnas para el scaler,
    incluyendo 'adjclose' que es la característica faltante.
    """
    print(f"--- Descargando con YahooQuery: {ticker} ---")
    
    try:
        t = Ticker(ticker)
        df = t.history(period='2y', interval='1d')
        
        if df.empty or (isinstance(df, dict) and ticker in df):
            raise ValueError(f"No se obtuvieron datos para {ticker}")

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level=0)
            
    except Exception as e:
        print(f"Error en YahooQuery: {e}")
        raise ValueError(f"Error al obtener datos de {ticker}: {str(e)}")

    # 1. Preparación de columnas base
    df = df.copy()
    
    # Creamos 'Close' (con C mayúscula) para cálculos, pero mantenemos adjclose (minúscula)
    df['Close'] = df['adjclose'] if 'adjclose' in df.columns else df['close']

    # 2. Cálculo de Indicadores Técnicos
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    
    # 3. FILTRADO ESTRICTO DE 12 COLUMNAS
    # Definimos la lista exacta de las 12 columnas que el scaler espera.
    # El orden es importante para que coincida con el entrenamiento.
    cols_to_keep = [
        'open', 'high', 'low', 'close', 'adjclose', 'volume', 
        'Close', 'SMA_20', 'SMA_50', 'RSI', 'BB_Mid', 'BB_Std'
    ]
    
    # Aseguramos que todas las columnas existan antes de filtrar
    for col in cols_to_keep:
        if col not in df.columns:
            df[col] = 0  # Relleno de seguridad si falta alguna
    
    # Seleccionamos solo las 12 columnas en el orden definido
    df_clean = df[cols_to_keep].dropna().copy()
    
    # Verificación de seguridad en logs
    print(f"DEBUG: Columnas finales ({len(df_clean.columns)}): {df_clean.columns.tolist()}")
    
    return df_clean

def download_data(ticker, start_date, end_date):
    t = Ticker(ticker)
    return t.history(period='2y')
