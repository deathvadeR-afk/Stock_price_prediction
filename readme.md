# NVIDIA Stock Price Prediction

This project leverages deep learning and financial data to predict NVIDIA's stock price trends. It includes a Jupyter Notebook for model development and a Streamlit web app for interactive predictions.

## Project Structure

- `stock_prediction_model.ipynb`: Jupyter Notebook for data collection, preprocessing, model training, and evaluation.
- `stock_price_predictor.py`: Streamlit app for interactive stock price prediction and visualization.
- `Latest_stock_price_model.keras`: Saved Keras model trained on NVIDIA stock data.
- `requirements.txt`: List of required Python packages.

## Features

- Downloads 20 years of NVIDIA (NVDA) stock data using Yahoo Finance.
- Visualizes historical prices and moving averages.
- Trains an LSTM neural network to predict future closing prices.
- Evaluates model performance and visualizes predictions.
- Provides a Streamlit web interface for user-friendly predictions and charting.

## Setup Instructions

1. **Clone the repository**
   ```powershell
   git clone https://github.com/deathvadeR-afk/Stock_price_prediction.git
   cd NVIDIA_Stock_Prediction
   ```

2. **Install dependencies**
   It is recommended to use a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook (optional)**
   To explore or retrain the model:
   ```powershell
   jupyter notebook stock_prediction_model.ipynb
   ```

4. **Run the Streamlit App**
   ```powershell
   streamlit run stock_price_predictor.py
   ```
   - Enter a stock ticker (default: NVDA) to view predictions and charts.

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies (yfinance, matplotlib, scikit-learn, numpy, keras, pandas, streamlit, tensorflow)

## Notes

- The default model is trained on NVIDIA (NVDA) stock data. You can modify the ticker in the app for other stocks.
- For best results, ensure your environment has sufficient memory for model training and prediction.

## License

This project is for educational purposes. Please check individual package licenses for more details.
