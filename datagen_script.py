
import os
import math
import numpy as np
import pandas as pd
import neurokit2 as nk

# --- Configuration ---
SAMPLING_RATE = 250      # Hz (Higher for quality)
OUTPUT_ROOT = "./data/raw/datagen/final_dataset"
RAW_DIR = os.path.join(OUTPUT_ROOT, "tier1_raw_samples")
FEATURE_DIR = os.path.join(OUTPUT_ROOT, "tier2_features_24h")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# --- Base Parameters (Male 25yo) ---
BASE_FALLBACK = {"heart_rate": 65.0, "low_frequency": 0.10, "high_frequency": 0.25}

# Level Transforms (0-5)
LEVEL_TRANSFORMS = {
    0: {"hr_mul": 1.00, "hr_std_mul": 1.00, "low_add": 0.00, "high_add": 0.00},
    1: {"hr_mul": 1.05, "hr_std_mul": 0.90, "low_add": 0.05, "high_add": -0.05},
    2: {"hr_mul": 1.15, "hr_std_mul": 0.80, "low_add": 0.10, "high_add": -0.10},
    3: {"hr_mul": 1.30, "hr_std_mul": 0.60, "low_add": 0.20, "high_add": -0.15},
    4: {"hr_mul": 1.50, "hr_std_mul": 0.40, "low_add": 0.30, "high_add": -0.20},
    5: {"hr_mul": 1.70, "hr_std_mul": 0.25, "low_add": 0.40, "high_add": -0.25},
}

ARTIFACT_CONFIG = {
    0: {'baseline_amp': 0.035, 'muscle_amp': 0.030, 'powerline_amp': 0.010, 'spike_prob': 0.001},
    1: {'baseline_amp': 0.020, 'muscle_amp': 0.010, 'powerline_amp': 0.005, 'spike_prob': 0.0005},
    2: {'baseline_amp': 0.055, 'muscle_amp': 0.040, 'powerline_amp': 0.018, 'spike_prob': 0.002},
    3: {'baseline_amp': 0.030, 'muscle_amp': 0.020, 'powerline_amp': 0.010, 'spike_prob': 0.001},
    4: {'baseline_amp': 0.035, 'muscle_amp': 0.025, 'powerline_amp': 0.012, 'spike_prob': 0.0015},
    5: {'baseline_amp': 0.040, 'muscle_amp': 0.035, 'powerline_amp': 0.015, 'spike_prob': 0.002}, 
}

def get_asian_hrv_baseline(age):
    age = max(19, min(80, age))
    hf = math.exp(-0.039 * age + 6.833)
    lf = math.exp(-0.047 * age + 7.197)
    sdnn = max(5.0, -0.502 * age + 53.907)
    return {"SDNN_base": sdnn, "HR_base": 65.0}

def get_stress_level_at_hour(hour, scenario_type):
    """Defined circadian patterns for 3 scenarios."""
    h = hour % 24
    
    if scenario_type == "Normal":
        # Sleep 0-6, Wake 6-8, Work 8-17, Relax 17-22, Sleep 22-24
        if 0 <= h < 6: return 0
        if 6 <= h < 8: return 1
        if 8 <= h < 12: return 2
        if 12 <= h < 13: return 1 # Lunch
        if 13 <= h < 17: return 2
        if 17 <= h < 22: return 1
        if 22 <= h < 24: return 0
        
    elif scenario_type == "Acute_Stress":
        # 10-12 High Stress, 16-17 Peak Stress
        if 0 <= h < 6: return 0
        if 6 <= h < 8: return 1
        if 8 <= h < 10: return 2
        if 10 <= h < 12: return 4 # High Stress
        if 12 <= h < 13: return 1
        if 13 <= h < 16: return 3
        if 16 <= h < 17: return 5 # PEAK
        if 17 <= h < 18: return 2 # Cooldown
        if 18 <= h < 22: return 1
        return 0
        
    elif scenario_type == "Chronic_Stress":
        # Poor sleep (L1), All day High Stress (L4-5)
        if 0 <= h < 6: return 1 # Poor sleep
        if 6 <= h < 8: return 2 # Morning anxiety
        if 8 <= h < 12: return 4
        if 12 <= h < 13: return 3 # Anxious Lunch
        if 13 <= h < 18: return 5 # Exhaustion
        if 18 <= h < 22: return 4 # Cannot relax
        return 3 # Insomnia
        
    return 0

def generate_noise(length, level):
    cfg = ARTIFACT_CONFIG.get(level, ARTIFACT_CONFIG[0])
    t = np.arange(length) / float(SAMPLING_RATE)
    rng = np.random.RandomState()
    
    baseline = cfg['baseline_amp'] * np.sin(2.0 * np.pi * 0.2 * t + rng.uniform(0, 2*np.pi))
    pl_freq = 50
    powerline = cfg['powerline_amp'] * np.sin(2.0 * np.pi * pl_freq * t + rng.uniform(0, 2*np.pi))
    noise = rng.normal(0.0, 1.0, size=length) * cfg['muscle_amp']
    
    if cfg.get('spike_prob', 0) > 0:
        spike_mask = rng.rand(length) < cfg['spike_prob']
        noise[spike_mask] += cfg['muscle_amp'] * 5.0 * rng.choice([-1, 1], size=np.sum(spike_mask))

    return baseline + powerline + noise

def generate_ecg_segment(duration_sec, level, age, gender="M", add_noise=False):
    # Determine parameters
    base = get_asian_hrv_baseline(age)
    tr = LEVEL_TRANSFORMS.get(level, LEVEL_TRANSFORMS[0])
    
    # Simple Gender adj
    g_mul = 1.05 if gender == 'F' else 1.0
    
    target_hr = base['HR_base'] * tr['hr_mul'] * g_mul
    target_sdnn = base['SDNN_base'] * tr['hr_std_mul']
    # convert SDNN to hr_std approximation
    target_hr_std = (target_sdnn * (target_hr**2)) / 60000.0
    
    lf = BASE_FALLBACK['low_frequency'] + tr['low_add']
    hf = BASE_FALLBACK['high_frequency'] + tr['high_add']
    
    try:
        ecg = nk.ecg_simulate(
            duration=duration_sec, 
            sampling_rate=SAMPLING_RATE, 
            heart_rate=target_hr, 
            heart_rate_std=target_hr_std,
            low_frequency=lf,
            high_frequency=hf
        )
        if add_noise:
            ecg += generate_noise(len(ecg), level)
        return ecg, target_hr, target_sdnn
    except:
        return np.zeros(int(duration_sec*SAMPLING_RATE)), target_hr, target_sdnn

def extract_features_from_segment(ecg_segment):
    try:
        # Fast extraction using neurokit
        # Clean first? If noisy yes.
        # For speed in simulation, we try raw peaks first.
        # If noise is high, clean is needed.
        cleaned = nk.ecg_clean(ecg_segment, sampling_rate=SAMPLING_RATE, method="neurokit")
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=SAMPLING_RATE)
        peaks = rpeaks['ECG_R_Peaks']
        
        if len(peaks) < 3: return None
        
        hrv_time = nk.hrv_time(peaks, sampling_rate=SAMPLING_RATE, show=False)
        # hrv_freq = nk.hrv_frequency(peaks, sampling_rate=SAMPLING_RATE, show=False) # Slow
        
        # Simplified return
        return {
            "HR": 60000/hrv_time['HRV_MeanNN'].values[0],
            "SDNN": hrv_time['HRV_SDNN'].values[0],
            "RMSSD": hrv_time['HRV_RMSSD'].values[0],
            "pNN50": hrv_time['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv_time else 0.0,
            # Skip freq domain for speed unless critical (can add if needed)
        }
    except:
        return None

# --- Main Generators ---

def generate_tier1_raw_samples():
    """Generate 5-minute Raw ECG samples for visualization."""
    print("Generating Tier 1: Raw Samples (5 mins)...")
    scenarios = [
        ("Normal_Morning", 2, "Normal"),
        ("Acute_Peak_Stress", 5, "Stress"),
        ("Chronic_Exhaustion", 4, "Stress")
    ]
    
    for name, level, _ in scenarios:
        # Clean
        ecg, _, _ = generate_ecg_segment(300, level, 25, "M", add_noise=False)
        pd.DataFrame({"Time": np.arange(len(ecg))/SAMPLING_RATE, "Voltage": ecg}).to_csv(
            os.path.join(RAW_DIR, f"Raw_{name}_Clean.csv"), index=False
        )
        # Noisy
        ecg_n, _, _ = generate_ecg_segment(300, level, 25, "M", add_noise=True)
        pd.DataFrame({"Time": np.arange(len(ecg_n))/SAMPLING_RATE, "Voltage": ecg_n}).to_csv(
            os.path.join(RAW_DIR, f"Raw_{name}_Noisy.csv"), index=False
        )
    print("Tier 1 Complete.")

def compute_features_for_level(level, age, gender="M", add_noise_jitter=False):
    """Compute HRV features directly from formulas (no ECG simulation needed).
    
    Features derived from level transforms + age baseline + realistic random variation.
    Includes: HR, SDNN, RMSSD, pNN50, LF, HF, LF_HF_Ratio
    """
    base = get_asian_hrv_baseline(age)
    tr = LEVEL_TRANSFORMS.get(level, LEVEL_TRANSFORMS[0])
    g_mul = 1.05 if gender == 'F' else 1.0
    rng = np.random.default_rng()
    
    # Target HR & SDNN
    target_hr   = base['HR_base'] * tr['hr_mul'] * g_mul
    target_sdnn = base['SDNN_base'] * tr['hr_std_mul']
    
    # RMSSD: ~1.2x SDNN at rest, decreases with stress
    rmssd_ratio   = 1.2 - (level * 0.12)   # L0:1.2  L5:0.6
    target_rmssd  = target_sdnn * rmssd_ratio
    
    # pNN50: age regression, scaled down by stress
    pnn50_base    = max(0.0, -0.650 * age + 53.852)
    pnn50_scale   = max(0.0, 1.0 - level * 0.18)  # L0:1.0  L5:0.10
    target_pnn50  = pnn50_base * pnn50_scale
    
    # LF (ms²): sympathetic power — increases with stress
    # Base LF ~1000 ms² at rest, rises with level
    lf_base   = BASE_FALLBACK['low_frequency']   # fraction (0.10)
    target_lf = (lf_base + tr['low_add']) * 10000  # scale to ms² range
    
    # HF (ms²): parasympathetic power — decreases with stress
    hf_base   = BASE_FALLBACK['high_frequency']  # fraction (0.25)
    target_hf = max(50.0, (hf_base + tr['high_add']) * 10000)
    
    # LF/HF ratio: key autonomic balance index
    target_lfhf = target_lf / target_hf
    
    # Physiological jitter
    jitter_pct = 0.05 if not add_noise_jitter else 0.12
    
    hr      = target_hr * (1.0 + rng.normal(0, jitter_pct))
    sdnn    = max(1.0,   target_sdnn * (1.0 + rng.normal(0, jitter_pct)))
    rmssd   = max(1.0,   target_rmssd * (1.0 + rng.normal(0, jitter_pct)))
    pnn50   = max(0.0,   min(100.0, target_pnn50 * (1.0 + rng.normal(0, jitter_pct * 1.5))))
    lf      = max(10.0,  target_lf * (1.0 + rng.normal(0, jitter_pct)))
    hf      = max(10.0,  target_hf * (1.0 + rng.normal(0, jitter_pct)))
    lf_hf   = max(0.01,  lf / hf)
    
    return {
        "HR_Target":    target_hr,
        "SDNN_Target":  target_sdnn,
        "HR_Extracted": hr,
        "SDNN_Extracted": sdnn,
        "RMSSD":        rmssd,
        "pNN50":        pnn50,
        "LF":           round(lf, 3),
        "HF":           round(hf, 3),
        "LF_HF_Ratio":  round(lf_hf, 4),
    }

def generate_tier2_features_24h():
    """Generate 24h Feature Datasets using formula-based computation (FAST)."""
    print("Generating Tier 2: 24h Feature Datasets (Formula-based)...")
    window_sec = 30
    total_sec = 24 * 3600
    steps = int(total_sec / window_sec)  # 2880 steps
    
    scenarios = ["Normal", "Acute_Stress", "Chronic_Stress"]
    noise_modes = [False, True]
    
    age = 25
    gender = "M"
    
    for sc in scenarios:
        for is_noisy in noise_modes:
            mode_str = "Noisy" if is_noisy else "Clean"
            print(f"  -> Scenario: {sc} ({mode_str})")
            
            data_rows = []
            
            for i in range(steps):
                current_time_sec = i * window_sec
                hour_float = current_time_sec / 3600.0
                
                level = get_stress_level_at_hour(hour_float, sc)
                feats = compute_features_for_level(level, age, gender, add_noise_jitter=is_noisy)
                
                row = {
                    "Time_Sec": current_time_sec,
                    "Hour": round(hour_float, 4),
                    "Stress_Level": level,
                    "Scenario": sc,
                    "Is_Noisy": is_noisy,
                    **feats
                }
                data_rows.append(row)
            
            # Save
            df = pd.DataFrame(data_rows)
            fname = f"Dataset_Features_24h_{sc}_{mode_str}.csv"
            df.to_csv(os.path.join(FEATURE_DIR, fname), index=False)
            print(f"    Saved {fname} ({len(df)} rows)")

if __name__ == "__main__":
    generate_tier1_raw_samples()
    generate_tier2_features_24h()
    print("All Generation Tasks Completed.")

