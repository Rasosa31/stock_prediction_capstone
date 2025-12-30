# Stock Prediction Capstone: Deep Learning & Serverless

This project is a Capstone implementation for the Machine Learning course. It upgrades the Midterm Project by introducing **Deep Learning (LSTM)**, **Advanced Technical Indicators**, and **Serverless Deployment (AWS Lambda)**.

## Project Structure
```
stock_prediction_capstone/
├── app/                # Deployment code (Lambda/Docker)
│   ├── app.py          # FastAPI handler adapted with Mangum
│   └── Dockerfile      # Container definition
├── data/               # Local data storage
├── models/             # Saved models and scalers
├── notebooks/          # Exploratory Analysis and Training Notebooks
│   ├── eda.ipynb       # EDA visualizations
│   └── train_lstm.ipynb# Model training playground
├── src/                # Project source code
│   ├── model.py        # LSTM architecture & training loop
│   ├── strategy.py     # Buy/Sell signal & Risk Management logic
│   └── utils.py        # Data loading & Indicator calculation
├── train.py            # Script for reproducing training
├── requirements.txt    # Dependencies
└── README.md           # This documentation
```

## Features
- **Deep Learning**: Uses an LSTM (Long Short-Term Memory) neural network for time-series forecasting.
- **Technical Analysis**: Incorporates SMA, RSI, and Bollinger Bands.
- **Trading Strategy**: Generates BUY/SELL signals with automatic Stop Loss and Take Profit levels based on volatility.
- **Serverless API**: Deploys as a Docker container on AWS Lambda with FastAPI and Mangum.

## Setup & Local Execution

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run EDA**
   Open `notebooks/eda.ipynb` in Jupyter/VSCode or run:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

3. **Train Model**
   You can use the notebook `notebooks/train_lstm.ipynb` or the script:
   ```bash
   python train.py --ticker AAPL --epochs 20
   ```
   This saves `lstm_model.h5` and `scaler.pkl` to `models/`.

## Deployment (AWS Lambda)

The application is containerized for AWS Lambda to handle large dependencies like TensorFlow.

1. **Build Docker Image**
   ```bash
   docker build -t stock-pred -f app/Dockerfile .
   ```

2. **Run Locally (Test)**
   ```bash
   docker run -p 8080:8080 stock-pred
   ```
   Send a request:
   ```bash
   curl -X POST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{"body": "{\"ticker\": \"AAPL\"}"}'
   ```
   *Note: The local Lambda emulator endpoint format differs slightly from direct API Gateway calls.*

3. **Deploy to AWS**
   - Push image to AWS ECR.
   - Create Lambda function from Container Image.
   - Set up API Gateway to trigger the Lambda.

## Strategy Logic
The model predicts the next day's Close price.
- **Signal**: BUY if predicted > current + threshold, SELL if predicted < current - threshold.
- **Risk Management**: Stop Loss and Take Profit are calculated using volatility ($2 \times \sigma$).

## Deployment (Render)

This project can be deployed to Render as a simple Python web service using `uvicorn`.

Quick steps:

1. Ensure the model files are available. The app expects:
   - `models/lstm_model.h5`
   - `models/scaler.pkl`

   Options to provide models:
   - Commit small model files into the `models/` folder in the repo (not recommended for large files).
   - Store models in an object storage (e.g., AWS S3) and set a startup step to download them. Add env vars like `MODEL_S3_URL` and change the app to download on startup.

2. Connect your GitHub repo to Render: https://dashboard.render.com/new

3. If you include `render.yaml` at the repo root (already included), Render will auto-create a service using that spec. The `render.yaml` in this repo sets the start command to:
   ```bash
   uvicorn app.app:app --host 0.0.0.0 --port $PORT
   ```

4. Required environment variables (set these in Render Dashboard or in `render.yaml` `envVars`):
   - `MODEL_PATH` (default `models/lstm_model.h5`)
   - `SCALER_PATH` (default `models/scaler.pkl`)

5. Deploy: Render will build using `pip install -r requirements.txt` and then run the start command. Watch the service logs to confirm the model loads on startup.

Notes & troubleshooting:
- If your model files are large, use S3 and modify `app/app.py` to download and cache the files at startup.
- If you prefer a Docker-based deploy, the repo includes `app/Dockerfile` for an AWS Lambda container image; for Render you can alternatively provide a Dockerfile that runs `uvicorn` instead.

If you want, I can:
- Add a small startup helper in `app/app.py` to download models from an S3 URL when `MODEL_S3_URL` is set.
- Create a second Dockerfile optimized for running on Render with `uvicorn`.
