import time
import numpy as np
import pandas as pd
import ccxt

def generate_synthetic_futures_data(
    n_steps: int = 5000,
    base_price: float = 100.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Fallback Synthetic commodity data generator (retained for backward compatibility and offline testing).
    Models Ornstein-Uhlenbeck prices, convenience yields, basis, open interest, and alternative data.
    """
    np.random.seed(seed)
    
    # Base Price Process (OU mean-reverting)
    theta = 0.03
    mu = base_price
    sigma = 1.5
    dt = 1.0
    
    prices = [base_price]
    for _ in range(1, n_steps):
        dp = theta * (mu - prices[-1]) * dt + sigma * np.random.normal()
        prices.append(max(10.0, prices[-1] + dp))
    prices = np.array(prices)
    
    y_theta = 0.05
    y_mu = 0.02
    y_sigma = 0.01
    y_current = y_mu
    y_series = []
    for _ in range(n_steps):
        dy = y_theta * (y_mu - y_current) + y_sigma * np.random.normal()
        y_current = np.clip(y_current + dy, -0.05, 0.15)
        y_series.append(y_current)
    convenience_yield = np.array(y_series)
    
    r = 0.05 
    tau = 0.25
    futures_prices = prices * np.exp((r - convenience_yield) * tau)
    basis = futures_prices - prices
    carry = r - convenience_yield
    roll_yield = (prices - futures_prices) / prices
    
    oi_base = 50000.0
    oi_series = [oi_base]
    for _ in range(1, n_steps):
        doi = 0.02 * (oi_base - oi_series[-1]) + 500.0 * np.random.normal()
        oi_series.append(max(5000.0, oi_series[-1] + doi))
    open_interest = np.array(oi_series)
    
    speculator_ratio = 0.6 * np.sin(np.linspace(0, 4 * np.pi, n_steps)) + 0.1 * np.random.normal(size=n_steps)
    speculator_ratio = np.clip(speculator_ratio, -1.0, 1.0)
    
    df_prices = pd.Series(prices)
    volatility = df_prices.rolling(window=20, min_periods=1).std().fillna(sigma).values
    
    spread = 0.0005 * prices + 0.0002 * volatility * np.random.exponential(scale=1.0, size=n_steps)
    spread = np.clip(spread, 0.01, 10.0)
    
    bid_depth = 5000.0 + 2000.0 * np.random.normal(size=n_steps) + speculator_ratio * 1500.0
    ask_depth = 5000.0 + 2000.0 * np.random.normal(size=n_steps) - speculator_ratio * 1500.0
    bid_depth = np.clip(bid_depth, 100.0, 20000.0)
    ask_depth = np.clip(ask_depth, 100.0, 20000.0)
    
    sentiment_series = [0.0]
    for _ in range(1, n_steps):
        dsent = 0.1 * (0.0 - sentiment_series[-1]) + 0.15 * np.random.normal()
        sentiment_series.append(np.clip(sentiment_series[-1] + dsent, -1.0, 1.0))
    sentiment = np.array(sentiment_series)
    
    surprise = np.zeros(n_steps)
    shocks = np.random.choice(n_steps, size=int(n_steps * 0.02), replace=False)
    surprise[shocks] = np.random.uniform(0.5, 1.0, size=len(shocks))
    
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-01-01", periods=n_steps, freq="h"),
        "mid_price": prices,
        "futures_price": futures_prices,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "convenience_yield": convenience_yield,
        "basis": basis,
        "carry": carry,
        "roll_yield": roll_yield,
        "open_interest": open_interest,
        "speculator_ratio": speculator_ratio,
        "sentiment": sentiment,
        "surprise": surprise,
        "volatility": volatility
    })
    
    df.set_index("timestamp", inplace=True)
    return df

def fetch_real_binance_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days_back: int = 60,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Downloads real historical market data from Binance Spot and Futures endpoints using CCXT:
    1. Chronological download loops for Spot & Futures OHLCV.
    2. Downloads historical Funding Rates and Open Interest (OI).
    3. Aligns and synchronizes all features onto a clean time-series index.
    4. Computes true mathematical features (Basis, Realized Volatility, Volume Imbalance).
    
    Returns:
        pd.DataFrame: Aligned historical market dataset.
    """
    print(f"[Data Pipeline] Fetching {days_back} days of real Binance data for {symbol}...")
    
    # Initialize Spot & Futures exchange instances
    spot_exchange = ccxt.binance({'enableRateLimit': True})
    futures_exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'} # Linear Futures/Swap
    })
    
    # فرمت‌بندی ایمن نمادها جهت جلوگیری از ثبت پسوند تکراری (مانند ETH/USDT:USDT:USDT)
    clean_symbol = symbol.split(':')[0]
    futures_symbol = f"{clean_symbol}:USDT" # Binance linear perpetual
    spot_symbol = clean_symbol
    
    # Calculate start time
    now_ms = futures_exchange.milliseconds()
    start_time_ms = now_ms - (days_back * 24 * 60 * 60 * 1000)
    
    # A. Fetch Spot and Futures OHLCV
    def fetch_ohlcv_chronological(exchange, sym, start_ms):
        all_ohlcv = []
        current_since = start_ms
        retries = 0
        
        while current_since < now_ms:
            try:
                ohlcv = exchange.fetch_ohlcv(sym, timeframe, current_since, limit=1000)
                if len(ohlcv) == 0:
                    break
                all_ohlcv.extend(ohlcv)
                # Shift forward to avoid repeating the last candle
                current_since = ohlcv[-1][0] + 1
                retries = 0
                time.sleep(0.15) # respect rate limits
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    print(f"[Data Pipeline] Error downloading {sym}: {e}. Retries exceeded.")
                    break
                time.sleep(1.0)
        return all_ohlcv

    print("[Data Pipeline] Fetching Futures OHLCV...")
    futures_ohlcv = fetch_ohlcv_chronological(futures_exchange, futures_symbol, start_time_ms)
    print(f"[Data Pipeline] Fetched {len(futures_ohlcv)} Futures bars.")
    
    print("[Data Pipeline] Fetching Spot OHLCV...")
    spot_ohlcv = []
    # Check if the symbol is available on Binance Spot before attempting fetch
    is_spot_available = False
    try:
        spot_exchange.load_markets()
        if spot_symbol in spot_exchange.symbols:
            is_spot_available = True
    except Exception as e:
        print(f"[Data Pipeline Warning] Failed to load spot markets: {e}")
        is_spot_available = True

    if is_spot_available:
        try:
            spot_ohlcv = fetch_ohlcv_chronological(spot_exchange, spot_symbol, start_time_ms)
        except Exception as e:
            print(f"[Data Pipeline Warning] Failed Spot OHLCV fetch for {spot_symbol}: {e}")
    else:
        print(f"[Data Pipeline Info] Symbol {spot_symbol} is not listed on Binance Spot. Skipping Spot download and using Futures data.")
    
    print(f"[Data Pipeline] Fetched {len(spot_ohlcv)} Spot bars.")
    
    if len(futures_ohlcv) == 0:
        raise ValueError("[Data Pipeline] Failed to download Futures OHLCV bars. Please check connection.")
        
    if len(spot_ohlcv) == 0:
        print("[Data Pipeline] Warning: No Spot bars found. Falling back to copy Futures prices as Spot prices.")
        # Mock Spot price using Futures price: [timestamp, open, high, low, close, volume]
        spot_ohlcv = [[bar[0], bar[1], bar[2], bar[3], bar[4], bar[5]] for bar in futures_ohlcv]
        
    # Build DataFrames
    df_futures = pd.DataFrame(futures_ohlcv, columns=["timestamp", "f_open", "f_high", "f_low", "f_close", "f_volume"])
    df_futures["timestamp"] = pd.to_datetime(df_futures["timestamp"], unit="ms")
    df_futures.set_index("timestamp", inplace=True)
    
    df_spot = pd.DataFrame(spot_ohlcv, columns=["timestamp", "s_open", "s_high", "s_low", "s_close", "s_volume"])
    df_spot["timestamp"] = pd.to_datetime(df_spot["timestamp"], unit="ms")
    df_spot.set_index("timestamp", inplace=True)
    
    # B. Fetch Historical Funding Rates
    print("[Data Pipeline] Fetching Historical Funding Rates...")
    all_funding = []
    funding_since = start_time_ms
    while funding_since < now_ms:
        try:
            funding = futures_exchange.fetch_funding_rate_history(futures_symbol, since=funding_since, limit=100)
            if len(funding) == 0:
                break
            all_funding.extend(funding)
            funding_since = funding[-1]["timestamp"] + 1
            time.sleep(0.15)
        except Exception as e:
            print(f"[Data Pipeline] Funding Rate error/unsupported: {e}")
            break
            
    # C. Fetch Historical Open Interest
    print("[Data Pipeline] Fetching Historical Open Interest...")
    all_oi = []
    oi_since = start_time_ms
    while oi_since < now_ms:
        try:
            # Query Binance specific historical Open Interest hist endpoint via CCXT
            # If timeframe is 1m, use 5m for Open Interest since Binance doesn't support 1m OI history
            oi_timeframe = "5m" if timeframe == "1m" else timeframe
            oi_data = futures_exchange.fetch_open_interest_history(futures_symbol, oi_timeframe, since=oi_since, limit=500)
            if len(oi_data) == 0:
                break
            all_oi.extend(oi_data)
            oi_since = oi_data[-1]["timestamp"] + 1
            time.sleep(0.15)
        except Exception as e:
            print(f"[Data Pipeline] Open Interest error/unsupported: {e}")
            break
    # Translate CCXT timeframe to Pandas 3.0 compatible offset
    pandas_offset = timeframe
    if timeframe.endswith("m"):
        pandas_offset = timeframe.replace("m", "min")
    elif timeframe.endswith("h"):
        pandas_offset = timeframe.replace("h", "h")
    elif timeframe.endswith("d"):
        pandas_offset = timeframe.replace("d", "D")

    # D. Build Auxiliary DataFrames & Align
    # Aligns all datasets on clean time series index
    df_align = df_futures.join(df_spot, how="inner")
    
    # Merge Funding Rates
    if len(all_funding) > 0:
        df_funding = pd.DataFrame(all_funding)
        df_funding["timestamp"] = pd.to_datetime(df_funding["timestamp"], unit="ms")
        df_funding.set_index("timestamp", inplace=True)
        # Resample to align with timeframe index using forward fill
        df_funding = df_funding.resample(pandas_offset).ffill()
        df_align = df_align.join(df_funding[["fundingRate"]], how="left")
    else:
        df_align["fundingRate"] = 0.0001 # Default 0.01% base rate if API fails
        
    # Merge Open Interest
    if len(all_oi) > 0:
        df_oi = pd.DataFrame(all_oi)
        df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")
        df_oi.set_index("timestamp", inplace=True)
        # Resample and merge
        df_oi = df_oi.resample(pandas_offset).ffill()
        df_align = df_align.join(df_oi[["openInterestAmount"]], how="left")
    else:
        # Default mock OI based on spot volume proxy if API fails or is not enabled
        df_align["openInterestAmount"] = df_align["f_volume"] * 10.0
        
    # E. Real-world feature engineering (look-ahead free)
    # Fill remaining NaNs using forward fill then backward fill
    df_align.ffill(inplace=True)
    df_align.bfill(inplace=True)
    
    # 1. Prices
    df_align["mid_price"] = df_align["s_close"]
    df_align["futures_price"] = df_align["f_close"]
    
    # 2. Volatility (rolling annualized volatility based on futures hourly close log returns)
    df_align["log_returns"] = np.log(df_align["futures_price"] / df_align["futures_price"].shift(1))
    df_align["volatility"] = df_align["log_returns"].rolling(window=20).std().fillna(0.001)
    
    # 3. Spread (Historical Proxy: Volatility scaled spread)
    df_align["spread"] = 0.0002 * df_align["mid_price"] + 0.1 * df_align["volatility"] * df_align["mid_price"]
    
    # 4. Bid/Ask Depth Imbalance Historical Proxy:
    # Liquidity imbalance is strongly correlated with short-term price momentum and relative volume imbalance
    df_align["depth_imbalance"] = (df_align["f_close"] - df_align["f_open"]) / (df_align["f_high"] - df_align["f_low"] + 1e-8)
    df_align["depth_imbalance"] = np.clip(df_align["depth_imbalance"], -1.0, 1.0)
    
    # Synthesize bid/ask depth values based on imbalance for env compatibilities
    df_align["bid_depth"] = 10000.0 * (1.0 + df_align["depth_imbalance"])
    df_align["ask_depth"] = 10000.0 * (1.0 - df_align["depth_imbalance"])
    
    # 5. Basis (Theory of Storage substitute)
    # Basis = Futures - Spot
    df_align["basis"] = df_align["futures_price"] - df_align["mid_price"]
    df_align["carry"] = df_align["fundingRate"] # perpetual swap funding rate represents cost of carry
    df_align["roll_yield"] = df_align["basis"] / df_align["mid_price"]
    
    # 6. Open Interest & Sentiment
    df_align["open_interest"] = df_align["openInterestAmount"]
    df_align["speculator_ratio"] = 0.3 * np.sin(np.linspace(0, 2 * np.pi, len(df_align))) # generic proxy
    df_align["sentiment"] = 0.15 * np.sign(df_align["log_returns"].rolling(window=10).mean().fillna(0.0))
    df_align["surprise"] = 0.0
    df_align["convenience_yield"] = 0.0
    
    # Drop temp cleanups
    df_align.drop(columns=["log_returns"], inplace=True, errors="ignore")
    
    print(f"[Data Pipeline] Aligned Real Data completed successfully! Shape: {df_align.shape}")
    return df_align
