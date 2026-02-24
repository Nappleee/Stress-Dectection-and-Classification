"""
datagen_script.py
=================
Block-based Multi-Subject ECG Simulation Framework (HDF5 Output)

1. Scenario Engine: Time-based activities with physio modulation (shifts).
2. Block-based Generation: 60-second chunks to minimize memory & allow scaling.
3. HDF5 Storage: Continuous binary storage (float32) for raw signals.
4. Separation of Labels: Ground truth saved separately for evaluation.
"""

import os
import math
import random
import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d

# ══════════════════════════════════════════════════════════════
#   1. CONFIGURATION
# ══════════════════════════════════════════════════════════════

CONFIG = {
    "duration_hours":      8,          # 07:00 - 15:00
    "start_time":          "07:00",
    "sampling_rate":       250,        # Hz
    "block_sec":           60,         # Process in 1-min windows
    "output_dir":          "./data/research_dataset",
    "h5_filename":         "dataset.h5",
    "gt_filename":         "stress_ground_truth.csv",
    "meta_filename":       "subject_metadata.csv",

    "subjects_count":      5,
    "age_range":           (19, 69),
}

# ══════════════════════════════════════════════════════════════
#   2. SCENARIO ENGINE (Physiological Modulation)
# ══════════════════════════════════════════════════════════════

# Modulation values: (HR_shift_bpm, SDNN_shift_pct, stress_label)
# SDNN_shift_pct: negative means reduction
ACTIVITIES = {
    "commute":        (10, -0.15, 1),
    "normal_work":    (0,   0.00, 0),
    "meeting":        (15, -0.25, 2),
    "lunch":          (-5,  0.10, 0),
    "deadline":       (20, -0.35, 3),
    "break":          (-2,  0.05, 0),
    "physical_load":  (25, -0.10, 2), # Higher HR, less SDNN drop than mental stress
    "emergency":      (35, -0.50, 3)
}

SCENARIOS = {
    "office_worker": [
        ("07:00", "07:30", "commute"),
        ("07:30", "09:00", "normal_work"),
        ("09:00", "11:00", "meeting"),
        ("11:00", "12:00", "lunch"),
        ("12:00", "14:00", "normal_work"),
        ("14:00", "15:00", "deadline")
    ],
    "physical_worker": [
        ("07:00", "08:00", "commute"),
        ("08:00", "10:00", "physical_load"),
        ("10:15", "10:30", "break"),
        ("10:30", "12:00", "physical_load"),
        ("12:00", "13:00", "lunch"),
        ("13:00", "15:00", "physical_load")
    ],
    "high_pressure": [
        ("07:00", "08:00", "commute"),
        ("08:00", "10:00", "deadline"),
        ("10:00", "11:00", "emergency"),
        ("11:00", "12:00", "lunch"),
        ("12:00", "14:00", "meeting"),
        ("14:00", "15:00", "deadline")
    ]
}

# ══════════════════════════════════════════════════════════════
#   3. CORE MODELS
# ══════════════════════════════════════════════════════════════

def get_hrv_baseline(age: int) -> dict:
    """Age-dependent baseline based on Asian regression studies."""
    age = float(np.clip(age, 19, 69))
    noise = lambda: 1.0 + random.uniform(-0.05, 0.05)
    
    sdnn  = (-0.502 * age + 53.907) * noise()
    hf    = math.exp(-0.039 * age + 6.833) * noise()
    lf    = math.exp(-0.047 * age + 7.197) * noise()
    
    total_pow = lf + hf
    return {
        "HR":       float(np.clip(72 - 0.1 * (age-25) + random.uniform(-5, 5), 50, 90)),
        "SDNN":     max(10.0, sdnn),
        "LF_ratio": lf / total_pow,
        "HF_ratio": hf / total_pow,
    }

def get_activity_at_sec(sec: int, scenario_list: list):
    """Find activity modulation for current time."""
    ref_min = int(CONFIG["start_time"].split(":")[0]) * 60
    current_min = ref_min + (sec // 60)
    
    for (t_s, t_e, act) in scenario_list:
        s_m = int(t_s.split(":")[0])*60 + int(t_s.split(":")[1])
        e_m = int(t_e.split(":")[0])*60 + int(t_e.split(":")[1])
        if s_m <= current_min < e_m:
            return ACTIVITIES[act]
    return ACTIVITIES["normal_work"]

def generate_rr_block(hr_target, sdnn_target, baseline, duration_sec):
    """Generate RR internal for a block with spectral balance."""
    rng = np.random.default_rng()
    rr_list, t = [], 0.0
    
    lf_ratio = baseline["LF_ratio"]
    hf_ratio = baseline["HF_ratio"]
    
    while t < duration_sec:
        rr_mean = 60.0 / hr_target
        sdnn_s  = sdnn_target / 1000.0
        
        # Spec oscillations
        lf = sdnn_s * lf_ratio * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2*np.pi))
        hf = sdnn_s * hf_ratio * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2*np.pi))
        ns = rng.normal(0, sdnn_s * 0.05)
        
        rr = float(np.clip(rr_mean + lf + hf + ns, 0.4, 1.5))
        rr_list.append(rr)
        t += rr
    return np.array(rr_list, dtype=np.float32)

def build_ecg_block(rr_intervals: np.ndarray, sr: int, noise_lvl: float):
    """Neurokit simulation for a short block."""
    rri_ms = (rr_intervals * 1000).astype(int)
    try:
        ecg = nk.ecg_simulate(rri=rri_ms, sampling_rate=sr, noise=noise_lvl)
        return ecg.astype(np.float32)
    except:
        return np.random.normal(0, noise_lvl, int(np.sum(rr_intervals) * sr)).astype(np.float32)

# ══════════════════════════════════════════════════════════════
#   4. MAIN SIMULATION
# ══════════════════════════════════════════════════════════════

def main():
    print(">>> Starting Block-based ECG Generation (HDF5)")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    h5_path   = os.path.join(CONFIG["output_dir"], CONFIG["h5_filename"])
    gt_path   = os.path.join(CONFIG["output_dir"], CONFIG["gt_filename"])
    meta_path = os.path.join(CONFIG["output_dir"], CONFIG["meta_filename"])

    # Prepare Metadata & GT lists
    meta_data = []
    gt_rows   = []

    with h5py.File(h5_path, 'w') as f:
        sr = CONFIG["sampling_rate"]
        duration_sec = CONFIG["duration_hours"] * 3600
        n_blocks = duration_sec // CONFIG["block_sec"]

        for s_idx in range(CONFIG["subjects_count"]):
            subj_id = f"Subject_{s_idx+1:02d}"
            age     = random.randint(*CONFIG["age_range"])
            gender  = random.choice(["M", "F"])
            sc_name = random.choice(list(SCENARIOS.keys()))
            
            print(f"  [{subj_id}] age={age}, {sc_name}")
            
            baseline = get_hrv_baseline(age)
            is_noisy = (s_idx % 2 == 1)
            
            meta_data.append({
                "subject_id": subj_id, 
                "age": age, 
                "gender": gender, 
                "scenario": sc_name,
                "baseline_HR": round(baseline["HR"], 1),
                "baseline_SDNN": round(baseline["SDNN"], 2),
                "noise_flag": is_noisy
            })
            grp = f.create_group(subj_id)
            
            # Since sampling is continuous, total samples = duration * sr
            total_samples = duration_sec * sr
            dset_ecg  = grp.create_dataset("ecg", (total_samples,), dtype='f4', compression="gzip")
            dset_time = grp.create_dataset("time", (total_samples,), dtype='f4')
            
            current_ptr = 0
            
            noise_val = 0.01 * (3.0 if is_noisy else 1.0)
            
            for b in range(n_blocks):
                t_base_sec = b * CONFIG["block_sec"]
                
                # Get modulation
                hr_shift, sd_shift_pct, label = get_activity_at_sec(t_base_sec, SCENARIOS[sc_name])
                
                hr_target   = baseline["HR"] + hr_shift
                sdnn_target = baseline["SDNN"] * (1.0 + sd_shift_pct)
                
                # Generate Block
                rr = generate_rr_block(hr_target, sdnn_target, baseline, CONFIG["block_sec"])
                ecg_block = build_ecg_block(rr, sr, noise_val)
                
                # Trim or pad to fit block_sec exactly to keep time alignment simple
                expected_samples = CONFIG["block_sec"] * sr
                if len(ecg_block) > expected_samples:
                    ecg_block = ecg_block[:expected_samples]
                elif len(ecg_block) < expected_samples:
                    ecg_block = np.pad(ecg_block, (0, expected_samples - len(ecg_block)), 'constant')
                
                # Write to HDF5
                dset_ecg[current_ptr : current_ptr + expected_samples] = ecg_block
                dset_time[current_ptr : current_ptr + expected_samples] = np.arange(current_ptr, current_ptr + expected_samples) / sr
                
                # Store Ground Truth per block (optional: can be per-second)
                gt_rows.append({"subject_id": subj_id, "time_sec": t_base_sec, "stress_label": label})
                
                current_ptr += expected_samples
            
    # Save CSVs
    pd.DataFrame(meta_data).to_csv(meta_path, index=False)
    pd.DataFrame(gt_rows).to_csv(gt_path, index=False)
    
    print(f"\nSimulation Complete.")
    print(f"HDF5 Dataset: {h5_path}")
    print(f"Metadata   : {meta_path}")
    print(f"Ground Truth: {gt_path}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
