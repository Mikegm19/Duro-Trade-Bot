import ccxt  # For fetching real-time crypto data from exchanges
import pandas as pd  # For calculating Moving Averages
import numpy as np  # For numerical operations
from scipy.signal import find_peaks  # For detecting peaks/troughs
import matplotlib.pyplot as plt  # For plotting charts
import time  # For real-time looping
import signal  # For handling graceful shutdown
import sys  # For exiting the program
import logging  # For better logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for configuration
SYMBOL = 'BTC/USDT'  # Crypto pair to analyze
EXCHANGE = ccxt.binance()  # Use Binance exchange (public access, no API key needed)
TIMEFRAME = '1m'  # 1-minute candles
DATA_LIMIT = 200  # Number of historical candles
MA_PERIOD = 50  # Period for Moving Average

def fetch_data(max_retries=3, retry_delay=5):
    """
    Fetches real-time OHLCV data from the exchange with retries.
    Returns: List of closing prices (numpy array) or None on failure.
    """
    for attempt in range(max_retries):
        try:
            ohlcv = EXCHANGE.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=DATA_LIMIT)
            if len(ohlcv) == 0:
                logging.warning("No data returned from exchange.")
                return None
            closes = np.array([candle[4] for candle in ohlcv])
            logging.info(f"Fetched {len(closes)} closing prices. Latest price: {closes[-1]}")
            return closes
        except ccxt.NetworkError as e:
            logging.warning(f"Network error: {e}. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay)
        except ccxt.ExchangeError as e:
            logging.error(f"Exchange error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None
    logging.error(f"Failed to fetch data after {max_retries} attempts.")
    return None

def calculate_indicators(prices):
    """
    Calculates MA and trend lines.
    Returns: Dict with 'ma', 'support', 'resistance' or None if insufficient data.
    """
    if len(prices) < MA_PERIOD + 1:
        logging.warning(f"Not enough data. Need at least {MA_PERIOD + 1} prices, got {len(prices)}.")
        return None
    
    # Calculate Simple Moving Average
    ma = pd.Series(prices).rolling(window=MA_PERIOD).mean().values
    if np.all(np.isnan(ma)):
        logging.warning("Moving Average calculation failed (all NaN).")
        return None
    
    # Detect peaks and troughs
    peaks, _ = find_peaks(prices, distance=10, prominence=0.01 * (prices.max() - prices.min()))
    troughs, _ = find_peaks(-prices, distance=10, prominence=0.01 * (prices.max() - prices.min()))
    logging.info(f"Detected {len(peaks)} peaks and {len(troughs)} troughs")
    
    # Fit trend lines
    resistance_slope, resistance_intercept = None, None
    if len(peaks) >= 2:
        x_peaks = np.arange(len(prices))[peaks]
        y_peaks = prices[peaks]
        resistance_slope, resistance_intercept = np.polyfit(x_peaks, y_peaks, 1)
        logging.info(f"Resistance line: slope={resistance_slope:.4f}, intercept={resistance_intercept:.4f}")
    
    support_slope, support_intercept = None, None
    if len(troughs) >= 2:
        x_troughs = np.arange(len(prices))[troughs]
        y_troughs = prices[troughs]
        support_slope, support_intercept = np.polyfit(x_troughs, y_troughs, 1)
        logging.info(f"Support line: slope={support_slope:.4f}, intercept={support_intercept:.4f}")
    
    return {
        'ma': ma,
        'support': (support_slope, support_intercept),
        'resistance': (resistance_slope, resistance_intercept)
    }

def visualize(prices, indicators):
    """
    Plots the chart with prices, MA, and trend lines.
    Saves to 'crypto_chart.png'.
    """
    if indicators is None:
        logging.warning("No indicators to visualize.")
        return
    
    x = np.arange(len(prices))
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, prices, label='Price', color='blue')
    
    # Plot MA, skipping NaN values
    ma = indicators['ma']
    valid_ma = ~np.isnan(ma)
    plt.plot(x[valid_ma], ma[valid_ma], label=f'{MA_PERIOD}-Period MA', color='orange')
    
    # Plot support line
    if indicators['support'][0] is not None:
        support_y = indicators['support'][0] * x + indicators['support'][1]
        plt.plot(x, support_y, label='Support Trend Line', color='green', linestyle='--')
    
    # Plot resistance line
    if indicators['resistance'][0] is not None:
        resistance_y = indicators['resistance'][0] * x + indicators['resistance'][1]
        plt.plot(x, resistance_y, label='Resistance Trend Line', color='red', linestyle='--')
    
    plt.title(f'{SYMBOL} Price Chart with Indicators')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('crypto_chart.png')
    logging.info("Chart saved to 'crypto_chart.png'")
    
    try:
        plt.show()
    except Exception as e:
        logging.warning(f"Cannot display plot (possibly non-GUI environment): {e}")
    
    plt.close()

def get_valuation(prices, indicators):
    """
    Provides valuation based on price position.
    Returns: List of signal strings.
    """
    if indicators is None:
        return ["Insufficient data for valuation."]
    
    latest_price = prices[-1]
    latest_ma = indicators['ma'][-1]
    
    if np.isnan(latest_ma):
        return ["Moving Average is NaN, cannot evaluate."]
    
    signals = []
    if latest_price > latest_ma:
        signals.append("Bullish: Price above MA (potential uptrend).")
    elif latest_price < latest_ma:
        signals.append("Bearish: Price below MA (potential downtrend).")
    else:
        signals.append("Neutral: Price at MA.")
    
    x_latest = len(prices) - 1
    if indicators['support'][0] is not None:
        support_at_latest = indicators['support'][0] * x_latest + indicators['support'][1]
        if latest_price < support_at_latest:
            signals.append("Warning: Price below support (possible breakdown).")
    
    if indicators['resistance'][0] is not None:
        resistance_at_latest = indicators['resistance'][0] * x_latest + indicators['resistance'][1]
        if latest_price > resistance_at_latest:
            signals.append("Warning: Price above resistance (possible overvaluation).")
    
    return signals

def signal_handler(sig, frame):
    """Handle graceful shutdown."""
    logging.info("Shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination
    
    while True:
        start_time = time.time()
        prices = fetch_data()
        if prices is not None:
            indicators = calculate_indicators(prices)
            if indicators is not None:
                visualize(prices, indicators)
                valuation = get_valuation(prices, indicators)
                logging.info("Valuation:")
                for signal in valuation:
                    logging.info(f"- {signal}")
        else:
            logging.warning("Skipping this cycle due to data fetch error.")
            time.sleep(10)  # Shorter sleep on error
        
        # Sleep to maintain 60-second cycle
        elapsed = time.time() - start_time
        sleep_time = max(60 - elapsed, 0)
        time.sleep(sleep_time)
