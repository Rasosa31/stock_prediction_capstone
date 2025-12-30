import sys
import os
import argparse
from src.utils import load_data_with_indicators
from src.model import train_and_save_model

def main():
    parser = argparse.ArgumentParser(description='Train LSTM Stock Prediction Model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock Ticker')
    parser.add_argument('--start', type=str, default='2015-01-01', help='Start Date')
    parser.add_argument('--end', type=str, default='2023-01-01', help='End Date')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    
    args = parser.parse_args()
    
    print(f"Training for {args.ticker} from {args.start} to {args.end}")
    
    try:
        df = load_data_with_indicators(args.ticker, args.start, args.end)
        
        # Select numeric features
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        os.makedirs('models', exist_ok=True)
        
        model, history, scaler = train_and_save_model(
            numeric_df, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            model_path='models/lstm_model.h5',
            scaler_path='models/scaler.pkl'
        )
        
        print("Training complete.")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
