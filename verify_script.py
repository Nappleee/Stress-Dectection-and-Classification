
import os
import numpy as np
import pandas as pd
import glob
from scipy import stats

# Configuration
DATA_ROOT = "./data/raw/datagen/final_dataset"
TIER1_DIR = os.path.join(DATA_ROOT, "tier1_raw_samples")
TIER2_DIR = os.path.join(DATA_ROOT, "tier2_features_24h")

def verify_tier1_raw():
    print("\n>>> Verifying Tier 1: Raw Samples <<<")
    files = glob.glob(os.path.join(TIER1_DIR, "*.csv"))
    if not files:
        print("  No Tier 1 files found.")
        return

    for f in files:
        fname = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            duration = df['Time'].iloc[-1]
            sr = len(df) / duration if duration > 0 else 0
            has_nan = df['Voltage'].isna().any()
            print(f"  {fname}: {duration:.1f}s, ~{int(sr)}Hz, NaNs: {has_nan}")
        except Exception as e:
            print(f"  {fname}: Error reading file - {e}")

def verify_tier2_features():
    print("\n>>> Verifying Tier 2: 24h Feature Datasets <<<")
    files = glob.glob(os.path.join(TIER2_DIR, "*.csv"))
    if not files:
        print("  No Tier 2 files found.")
        return

    for f in files:
        fname = os.path.basename(f)
        print(f"  Analysing {fname}...")
        df = pd.read_csv(f)
        
        # 1. Monotonicity Check (Stress Level vs HR/SDNN)
        # Group by level and check trends
        print("    [Trends by Level]")
        grouped = df.groupby("Stress_Level")[["HR_Extracted", "SDNN_Extracted"]].mean()
        print(grouped)
        
        # Check if HR increases with level
        hr_increasing = grouped["HR_Extracted"].is_monotonic_increasing
        # Check if SDNN decreases (roughly)
        sdnn_decreasing = grouped["SDNN_Extracted"].is_monotonic_decreasing
        
        print(f"    -> HR Increases with Stress: {hr_increasing}")
        print(f"    -> SDNN Decreases with Stress: {sdnn_decreasing}")
        
        # 2. Circadian Check (Day vs Night) for Normal/Chronic
        # Night: 00-06, Day: 08-18
        night_df = df[(df['Hour'] >= 0) & (df['Hour'] < 6)]
        day_df = df[(df['Hour'] >= 8) & (df['Hour'] < 18)]
        
        if not night_df.empty and not day_df.empty:
            night_hr = night_df['HR_Extracted'].mean()
            day_hr = day_df['HR_Extracted'].mean()
            print(f"    [Circadian] Night HR: {night_hr:.1f} vs Day HR: {day_hr:.1f}")
            if day_hr > night_hr:
                print("    -> Circadian Rhythm Detected (Day > Night).")
            else:
                print("    -> WARNING: Day HR not higher than Night HR.")
                
        # 3. Peak Detection in Acute
        if "Acute" in fname:
            peak_stress = df[df['Stress_Level'] == 5]
            if not peak_stress.empty:
                print(f"    [Acute] Found {len(peak_stress)} windows of Peak Stress (Level 5).")
            else:
                print("    [Acute] WARNING: No Level 5 windows found.")

        print("-" * 30)

if __name__ == "__main__":
    if os.path.exists(TIER1_DIR):
        verify_tier1_raw()
    else:
        print(f"Tier 1 Dir not found: {TIER1_DIR}")
        
    if os.path.exists(TIER2_DIR):
        verify_tier2_features()
    else:
        print(f"Tier 2 Dir not found: {TIER2_DIR}")
