from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import sys
import datetime

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Importaciones de tu proyecto
from src.model import create_sequences
from src.utils import load_data_with_indicators
from src.strategy import get_signal, calculate_sl_tp

app = FastAPI(title="Stock Prediction API")

# Rutas de archivos locales en Render
MODEL_PATH = 'models/lstm_model.h5'
SCALER_PATH = 'models/scaler.pkl'

model = None
scaler = None

def load_resources():
    global model, scaler
    if model is None:
        try:
            if os.path.exists(MODEL_PATH):
                print(f"Cargando modelo desde {MODEL_PATH}...")
                model = tf.keras.models.load_model(MODEL_PATH)
                print("Modelo cargado exitosamente.")
            else:
                print(f"ADVERTENCIA: No se encontró el modelo en {MODEL_PATH}")
        except Exception as e:
            print(f"Error cargando modelo: {e}")

    if scaler is None:
        try:
            if os.path.exists(SCALER_PATH):
                print(f"Cargando scaler desde {SCALER_PATH}...")
                scaler = joblib.load(SCALER_PATH)
                print("Scaler cargado exitosamente.")
            else:
                print(f"ADVERTENCIA: No se encontró el scaler en {SCALER_PATH}")
        except Exception as e:
            print(f"Error cargando scaler: {e}")

@app.on_event("startup")
async def startup_event():
    load_resources()

class TickerRequest(BaseModel):
    ticker: str

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Predictor AI - Capstone</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    </head>
    <body class="bg-slate-900 text-white min-h-screen flex flex-col items-center justify-center p-4 font-sans">
        <div class="max-w-md w-full bg-slate-800 rounded-2xl shadow-2xl p-6 border border-slate-700">
            <div class="text-center mb-6">
                <div class="inline-block p-3 rounded-full bg-blue-500/10 mb-3">
                    <i class="fas fa-robot text-blue-500 text-3xl"></i>
                </div>
                <h1 class="text-2xl font-black tracking-tight uppercase">Stock<span class="text-blue-500">AI</span> Predictor</h1>
                <p class="text-slate-400 text-[10px] uppercase tracking-[0.2em]">LSTM Neural Network Engine</p>
            </div>

            <div class="space-y-3">
                <div class="relative">
                    <input type="text" id="tickerInput" placeholder="EJ: AAPL o GOOG" 
                           class="w-full p-4 pl-12 rounded-xl bg-slate-900 border border-slate-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none uppercase font-bold text-lg transition-all">
                    <i class="fas fa-chart-line absolute left-4 top-5 text-slate-500"></i>
                </div>
                <button onclick="makePrediction()" id="btnPredict"
                        class="w-full bg-blue-600 hover:bg-blue-700 text-white font-black py-4 rounded-xl transition-all duration-200 shadow-lg active:scale-95 flex items-center justify-center gap-2">
                    <i class="fas fa-bolt"></i> ANALIZAR MERCADO
                </button>
            </div>

            <div id="loader" class="hidden mt-6 text-center">
                <i class="fas fa-brain text-blue-500 text-3xl fa-bounce mb-2"></i>
                <p class="text-xs text-slate-400 font-medium">Ejecutando predicción neuronal...</p>
            </div>

            <div id="result" class="hidden mt-6 space-y-3 animate-in fade-in zoom-in duration-300">
                <div class="grid grid-cols-2 gap-3">
                    <div class="bg-slate-900 p-3 rounded-xl border border-slate-700">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-tight">Precio Actual</p>
                        <p id="currPrice" class="text-lg font-mono font-bold"></p>
                    </div>
                    <div class="bg-slate-900 p-3 rounded-xl border border-slate-700">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-tight">Predicción</p>
                        <p id="predPrice" class="text-lg font-mono font-bold text-blue-400"></p>
                    </div>
                </div>
                
                <div class="grid grid-cols-2 gap-3">
                    <div class="bg-slate-900 p-3 rounded-xl border border-slate-700 border-l-red-500 border-l-4">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-tight">Stop Loss</p>
                        <p id="stopLoss" class="text-lg font-mono font-bold text-red-400"></p>
                    </div>
                    <div class="bg-slate-900 p-3 rounded-xl border border-slate-700 border-l-green-500 border-l-4">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-tight">Take Profit</p>
                        <p id="takeProfit" class="text-lg font-mono font-bold text-green-400"></p>
                    </div>
                </div>

                <div class="bg-slate-900 p-4 rounded-xl border border-slate-700 flex justify-between items-center">
                    <div>
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-tight">Variación Estimada</p>
                        <p id="pctChange" class="text-xl font-bold"></p>
                    </div>
                    <div class="text-right">
                        <p class="text-[10px] text-slate-500 uppercase font-bold tracking-tight mb-1">Señal de Trading</p>
                        <span id="signal" class="px-4 py-1 rounded-full text-[10px] font-black uppercase tracking-widest"></span>
                    </div>
                </div>
            </div>
        </div>
        <p class="mt-6 text-slate-600 text-[10px] tracking-widest uppercase">Capstone Final Project © 2025</p>

        <script>
            async function makePrediction() {
                const tickerInput = document.getElementById('tickerInput');
                const ticker = tickerInput.value.trim().toUpperCase();
                if (!ticker) return;

                const btn = document.getElementById('btnPredict');
                const loader = document.getElementById('loader');
                const result = document.getElementById('result');

                btn.disabled = true;
                btn.style.opacity = '0.5';
                loader.classList.remove('hidden');
                result.classList.add('hidden');

                try {
                    const response = await fetch('/predict_json', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ticker: ticker})
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('currPrice').innerText = '$' + data.current_price;
                        document.getElementById('predPrice').innerText = '$' + data.predicted_price;
                        document.getElementById('stopLoss').innerText = '$' + data.stop_loss;
                        document.getElementById('takeProfit').innerText = '$' + data.take_profit;
                        document.getElementById('pctChange').innerText = data.percent_change;
                        
                        const signalEl = document.getElementById('signal');
                        signalEl.innerText = data.signal;
                        
                        if(data.signal === 'BUY') {
                            signalEl.className = 'px-4 py-1 rounded-full text-[10px] font-black bg-emerald-500/20 text-emerald-400 border border-emerald-500/50';
                            document.getElementById('pctChange').className = 'text-xl font-bold text-emerald-400';
                        } else {
                            signalEl.className = 'px-4 py-1 rounded-full text-[10px] font-black bg-rose-500/20 text-rose-400 border border-rose-500/50';
                            document.getElementById('pctChange').className = 'text-xl font-bold text-rose-400';
                        }
                        
                        result.classList.remove('hidden');
                    } else {
                        alert('Error: ' + (data.detail || 'Fallo en la predicción'));
                    }
                } catch (err) {
                    alert('Error de conexión con la API');
                } finally {
                    btn.disabled = false;
                    btn.style.opacity = '1';
                    loader.classList.add('hidden');
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict_json")
def predict_json(request: TickerRequest):
    load_resources()
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Recursos del modelo no cargados en el servidor.")

    ticker = request.ticker.upper()
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=730)

    try:
        # 1. Cargar datos e indicadores (YahooQuery según tu log anterior)
        df = load_data_with_indicators(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # 2. Seleccionar columnas numéricas (El log mostró 12 columnas incluyendo indicadores)
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Obtener dimensiones esperadas por el modelo cargado
        # Forma esperada: (None, timesteps, features)
        expected_timesteps = model.input_shape[1]
        expected_features = model.input_shape[2] 
        
        if len(numeric_df) < expected_timesteps:
             raise HTTPException(status_code=400, detail=f"Datos históricos insuficientes para {ticker}. Se requieren {expected_timesteps} días.")

        # 3. Preparación de Features (Asegurar que enviamos las 12 columnas)
        # Tomamos las últimas columnas según lo que espera el modelo
        data_to_scale = numeric_df.iloc[:, :expected_features].values
        
        # 4. Escalar y Crear Secuencia
        # Si el scaler fue entrenado con 12 columnas, transformará todo. 
        # Si fue entrenado con 1, usaremos un escalado manual o parcial.
        try:
            scaled_data = scaler.transform(data_to_scale)
        except:
            # Fallback en caso de que el scaler pida solo 1 columna pero el modelo pida 12
            # Escalamos solo la columna Close (asumiendo que es la principal) y repetimos o ajustamos
            close_scaled = scaler.transform(numeric_df[['Close']].values)
            # Creamos una matriz de ceros con la forma correcta y ponemos el close escalado
            scaled_data = np.zeros((len(close_scaled), expected_features))
            scaled_data[:, 0] = close_scaled.flatten() 

        last_sequence = scaled_data[-expected_timesteps:]
        last_sequence = np.expand_dims(last_sequence, axis=0) # Shape: (1, timesteps, 12)

        # 5. Ejecutar Predicción
        prediction_scaled_val = model.predict(last_sequence, verbose=0)[0, 0]

        # 6. Des-escalado Manual (Inversa de Min-Max)
        current_price = float(df['Close'].iloc[-1])
        real_min = float(df['Close'].min())
        real_max = float(df['Close'].max())
        
        # La fórmula: x = x_scaled * (max - min) + min
        prediction_final = prediction_scaled_val * (real_max - real_min) + real_min
        
        # 7. Cálculo de Estrategia y Respuesta
        volatility = float(df['BB_Std'].iloc[-1]) if 'BB_Std' in df.columns else 0
        signal = get_signal(current_price, prediction_final)
        stop_loss, take_profit = calculate_sl_tp(current_price, signal, volatility)
        pct_change = ((prediction_final - current_price) / current_price) * 100
        
        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "predicted_price": round(float(prediction_final), 2),
            "percent_change": f"{round(float(pct_change), 2)}%",
            "signal": signal,
            "stop_loss": round(float(stop_loss), 2),
            "take_profit": round(float(take_profit), 2),
            "status": "Success"
        }
    except Exception as e:
        print(f"Error crítico en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
