import numpy as np
import pandas as pd

MIN_WINDOW_DURATION_SEC = 20.0
MIN_RR_COUNT = 5

def clean_waveform_group(g):
    g = g.sort_values("Time").drop_duplicates("Time")
    g = g[g["Time"].notna() & (g["Time"] >= 0)]
    g = g[g["Time"].diff().fillna(1) > 0]
    g["Peak"] = pd.to_numeric(g["Peak"], errors="coerce")
    v = pd.to_numeric(g["Voltage"], errors="coerce")
    q_low, q_high = np.nanquantile(v, [0.005, 0.995])
    g["Voltage"] = v.clip(lower=q_low, upper=q_high)
    return g.dropna(subset=["Time", "Voltage", "Peak"] )

def safe_skew(x):
    return float(pd.Series(x).skew()) if len(x) >= 3 else np.nan

def safe_kurt(x):
    return float(pd.Series(x).kurt()) if len(x) >= 4 else np.nan

def rr_frequency_features(rr_ms):
    if len(rr_ms) < 4:
        return (np.nan, np.nan, np.nan, np.nan)
    rr_s = rr_ms / 1000.0
    t = np.cumsum(rr_s) - rr_s[0]
    if t[-1] <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    fs = 4.0
    t_uniform = np.arange(0, t[-1], 1 / fs)
    if len(t_uniform) < 8:
        return (np.nan, np.nan, np.nan, np.nan)
    rr_interp = np.interp(t_uniform, t, rr_ms)
    rr_detrended = rr_interp - np.nanmean(rr_interp)
    fft_vals = np.fft.rfft(rr_detrended)
    freqs = np.fft.rfftfreq(len(rr_detrended), d=1 / fs)
    psd = (np.abs(fft_vals) ** 2) / max(len(rr_detrended), 1)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    total_mask = (freqs >= 0.04) & (freqs <= 0.40)
    lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else np.nan
    hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else np.nan
    total_power = float(np.trapezoid(psd[total_mask], freqs[total_mask])) if np.any(total_mask) else np.nan
    lf_hf_ratio = (lf_power / hf_power) if (pd.notna(lf_power) and pd.notna(hf_power) and hf_power > 0) else np.nan
    return lf_power, hf_power, lf_hf_ratio, total_power

def extract_features_from_window(g, window_id=0, source_file="raw_file", window_start=None, window_end=None):
    g = clean_waveform_group(g)
    duration_sec = float(g["Time"].max() - g["Time"].min()) if len(g) else 0.0
    if len(g) < 20 or duration_sec < MIN_WINDOW_DURATION_SEC:
        return None
    peak_times = g.loc[g["Peak"] == 3, "Time"].values
    if len(peak_times) < 4:
        return None
    rr_raw = np.diff(peak_times) * 1000.0  # ms
    rr_ms = rr_raw[(rr_raw >= 300.0) & (rr_raw <= 2000.0)]
    rr_valid_ratio = float(len(rr_ms) / max(len(rr_raw), 1))
    if len(rr_ms) < MIN_RR_COUNT:
        return None
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2)) if len(rr_ms) >= 2 else np.nan
    pnn50 = float(np.mean(np.abs(np.diff(rr_ms)) > 50.0) * 100.0) if len(rr_ms) >= 2 else np.nan
    mean_rr_ms = float(np.mean(rr_ms))
    sdnn_ms = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan
    hr_bpm = 60000.0 / mean_rr_ms if mean_rr_ms > 0 else np.nan
    n_peaks = int(np.sum(g["Peak"] == 3))
    peak_rate = n_peaks / duration_sec if duration_sec > 0 else np.nan
    lf_power, hf_power, lf_hf_ratio, total_power = rr_frequency_features(rr_ms)
    label_val = g["label"].mode().iloc[0] if "label" in g.columns and not g["label"].isnull().all() else np.nan
    return {
        "source_file": source_file,
        "window_id": window_id,
        "window_start": window_start,
        "window_end": window_end,
        "label": label_val,
        "mean_rr_ms": mean_rr_ms,
        "median_rr_ms": float(np.median(rr_ms)),
        "sdnn_ms": sdnn_ms,
        "rmssd_ms": float(rmssd),
        "pnn50": float(pnn50),
        "heart_rate_bpm": float(hr_bpm),
        "n_peaks": n_peaks,
        "duration_sec": duration_sec,
        "peak_rate_per_sec": float(peak_rate),
        "rr_valid_ratio": rr_valid_ratio,
        "voltage_mean": float(g["Voltage"].mean()),
        "voltage_std": float(g["Voltage"].std(ddof=1)) if len(g) > 1 else np.nan,
        "voltage_min": float(g["Voltage"].min()),
        "voltage_max": float(g["Voltage"].max()),
        "voltage_skew": safe_skew(g["Voltage"].values),
        "voltage_kurtosis": safe_kurt(g["Voltage"].values),
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
        "total_power": total_power,
    }
