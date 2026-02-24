"""
preprocess.py
=============
End-to-End Preprocessing Pipeline for Stress Classification

1. ECG Signal Preprocessing (Cleaning & Peak Detection)
2. HRV Feature Preprocessing (Time & Frequency Domain Extraction)
3. ML Preprocessing (Feature Scaling & Label Alignment)

Data Input: dataset.h5, subject_metadata.csv, stress_ground_truth.csv
Data Output: ecg_features.csv (Final ML-ready dataset)
"""

import os
import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import welch

# ══════════════════════════════════════════════════════════════
#   1. CONFIGURATION
# ══════════════════════════════════════════════════════════════

CONFIG = {
    "data_dir":         "./data/research_dataset",
    "h5_path":          "./data/research_dataset/dataset.h5",
    "meta_path":        "./data/research_dataset/subject_metadata.csv",
    "gt_path":          "./data/research_dataset/stress_ground_truth.csv",
    "output_feat_csv":  "./data/research_dataset/ecg_features.csv",
    
    "sampling_rate":    250,        # Hz
    "window_size_sec":  300,        # 5 minutes
    "overlap_sec":      150,        # 50% overlap for better temporal resolution
}

# ══════════════════════════════════════════════════════════════
#   2. SIGNAL PREPROCESSING
# ══════════════════════════════════════════════════════════════

def preprocess_ecg_signal(ecg_raw, sr):
    """Clean ECG and detect R-peaks."""
    try:
        # Bandpass filter (0.5 - 45 Hz) + Notch filter (50Hz) handled by nk
        cleaned = nk.ecg_clean(ecg_raw, sampling_rate=sr, method="neurokit")
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sr)
        return cleaned, rpeaks['ECG_R_Peaks']
    except Exception as e:
        print(f"      [Signal Error] {e}")
        return None, None

# ══════════════════════════════════════════════════════════════
#   3. HRV FEATURE PREPROCESSING
# ══════════════════════════════════════════════════════════════

def extract_window_hrv(rri_s):
    """Compute HRV indices for a single window of RR intervals."""
    if len(rri_s) < 15: return None
    
    # Time Domain
    rri_ms = rri_s * 1000
    diff_rri = np.diff(rri_ms)
    
    hr    = 60000.0 / np.mean(rri_ms)
    sdnn  = float(np.std(rri_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff_rri**2)))
    pnn50 = float(100.0 * np.mean(np.abs(diff_rri) > 50))

    # Frequency Domain (Welch PSD)
    try:
        # Interpolate to 4Hz (equidistant sampling)
        rt = np.cumsum(rri_s) - rri_s[0]
        t_uni = np.arange(0, rt[-1], 0.25)
        rri_uni = np.interp(t_uni, rt, rri_ms)
        
        f, psd = welch(rri_uni, fs=4.0, nperseg=min(256, len(rri_uni)))
        lf_mask = (f >= 0.04) & (f < 0.15)
        hf_mask = (f >= 0.15) & (f < 0.40)
        
        # NumPy 2.x compatible trapezoidal integration
        lf = float(np.trapezoid(psd[lf_mask], f[lf_mask])) if any(lf_mask) else 0.0
        hf = float(np.trapezoid(psd[hf_mask], f[hf_mask])) if any(hf_mask) else 0.01
        lfhf = lf / max(hf, 1e-9)
    except:
        lf, hf, lfhf = 0.0, 0.0, 0.0

    return {
        "HR": round(hr, 2), "SDNN": round(sdnn, 2), "RMSSD": round(rmssd, 2),
        "pNN50": round(pnn50, 2), "LF": round(lf, 4), "HF": round(hf, 4), "LF_HF_Ratio": round(lfhf, 4)
    }

# ══════════════════════════════════════════════════════════════
#   4. ML PREPROCESSING (Alignment & Merging)
# ══════════════════════════════════════════════════════════════

def main():
    print(">>> Starting Preprocessing Pipeline...")
    
    if not os.path.exists(CONFIG["h5_path"]):
        print(f"Error: Dataset HDF5 not found at {CONFIG['h5_path']}")
        return

    # Load Ground Truth and Metadata
    print("Loading Ground Truth and Metadata...")
    gt_df = pd.read_csv(CONFIG["gt_path"])
    meta_df = pd.read_csv(CONFIG["meta_path"])
    
    # Subject-wise processing
    all_extracted_features = []
    
    with h5py.File(CONFIG["h5_path"], 'r') as f:
        subjects = list(f.keys())
        print(f"Found {len(subjects)} subjects. Processing...")

        for subj_id in subjects:
            print(f"  -> Subject: {subj_id}")
            
            # Load Raw ECG
            ecg_raw = f[subj_id]['ecg'][:]
            sr = CONFIG["sampling_rate"]
            total_sec = len(ecg_raw) / sr
            
            # 1. ECG Signal Preprocessing
            print(f"     Stage 1: Signal Cleansing & Peak Detection...")
            cleaned, peaks = preprocess_ecg_signal(ecg_raw, sr)
            if peaks is None: continue
            
            # Convert peaks to RRI
            rri_s = np.diff(peaks) / sr
            peak_times = peaks[1:] / sr
            
            # 2. HRV Feature Preprocessing (Windowing)
            print(f"     Stage 2: HRV Feature Extraction...")
            win_size = CONFIG["window_size_sec"]
            step_size = win_size - CONFIG["overlap_sec"]
            
            subj_meta = meta_df[meta_df['subject_id'] == subj_id].iloc[0]
            subj_gt   = gt_df[gt_df['subject_id'] == subj_id]
            
            for t_start in range(0, int(total_sec) - win_size, step_size):
                t_end = t_start + win_size
                mask = (peak_times >= t_start) & (peak_times < t_end)
                win_rri = rri_s[mask]
                
                hrv_feats = extract_window_hrv(win_rri)
                if hrv_feats:
                    # 3. ML Preprocessing: Label Assignment
                    # Find dominant stress label in this window from ground truth
                    win_gt = subj_gt[(subj_gt['time_sec'] >= t_start) & (subj_gt['time_sec'] < t_end)]
                    if not win_gt.empty:
                        stress_label = int(win_gt['stress_label'].mode()[0])
                    else:
                        stress_label = 0 # Default if unknown

                    # Add row
                    row = {
                        "subject_id": subj_id,
                        "age": int(subj_meta['age']),
                        "gender": subj_meta['gender'],
                        "scenario": subj_meta['scenario'],
                        "is_noisy": bool(subj_meta['noise_flag']),
                        "window_start_sec": t_start,
                        "stress_label": stress_label, # Attached here for training convenience
                        **hrv_feats
                    }
                    all_extracted_features.append(row)
            
            print(f"     Finished {subj_id}.")

    # Save to CSV
    final_df = pd.DataFrame(all_extracted_features)
    final_df.to_csv(CONFIG["output_feat_csv"], index=False)
    
    print("\n>>> Preprocessing Complete.")
    print(f"Final ML-Ready Features Saved: {CONFIG['output_feat_csv']}")
    print(f"Total Window Samples: {len(final_df)}")
    print("\nSummary Counts (Stress Labels):")
    print(final_df['stress_label'].value_counts())

if __name__ == "__main__":
    main()
