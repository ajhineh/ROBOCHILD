import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def get_ffd_weights(d: float, thres: float = 1e-4) -> np.ndarray:
    """
    Generates weights for Fixed-width Window Fractional Differentiation (FFD).
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w)

def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """
    Applies Fixed-width Window Fractional Differentiation using fast NumPy convolution.
    """
    if d == 0.0:
        return series.copy()
        
    w = get_ffd_weights(d, thres)
    # Note: np.convolve uses reversed weights for standard convolution filter
    res = np.convolve(series.values, w[::-1], mode='valid')
    
    # Pad the beginning with NaNs to align with the original index
    pad_size = len(series) - len(res)
    res_full = np.empty(len(series))
    res_full[:pad_size] = np.nan
    res_full[pad_size:] = res
    
    return pd.Series(res_full, index=series.index)

def find_optimal_d(series: pd.Series, thres: float = 1e-4) -> float:
    """
    Finds the minimum fractional order d in [0.0, 1.0] that achieves 
    stationarity (95% confidence level in ADF test) while maximizing memory.
    """
    series_clean = series.ffill().bfill()
    for d in np.arange(0.0, 1.05, 0.05):
        try:
            diff_series = frac_diff_ffd(series_clean, d, thres).dropna()
            if len(diff_series) < 30:
                continue
            # Run Augmented Dickey-Fuller (ADF) test
            res = adfuller(diff_series, maxlag=1, regression='c', autolag=None)
            stat = res[0]
            critical_value_5pct = res[4]['5%']
            if stat < critical_value_5pct:
                return float(round(d, 2))
        except Exception:
            pass
    return 1.0

def get_latest_frac_diff(series_values: list, d: float, thres: float = 1e-4) -> float:
    """
    Computes the fractionally differentiated value of the most recent price in series_values.
    """
    if d == 0.0:
        return series_values[-1] if len(series_values) > 0 else 0.0
    w = get_ffd_weights(d, thres)
    n = len(w)
    if len(series_values) < n:
        # Pad by repeating the oldest available value to match the weight length
        pad_len = n - len(series_values)
        padded = [series_values[0]] * pad_len + list(series_values)
    else:
        padded = list(series_values[-n:])
    
    val = sum(w[i] * padded[-1-i] for i in range(n))
    return float(val)

