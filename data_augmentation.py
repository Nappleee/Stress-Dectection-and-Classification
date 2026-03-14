import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob

def add_gaussian_noise(df, column='Voltage', noise_factor=0.02):
    """
    Add Gaussian noise to a column
    noise_factor: fraction of the column's std to use as noise std
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in dataframe")
        return df
    
    signal_std = df[column].std()
    noise_std = noise_factor * signal_std
    
    noise = np.random.normal(0, noise_std, size=len(df))
    df[column] = df[column] + noise
    
    print(f"Added noise to '{column}': std={noise_std:.4f} (factor={noise_factor})")
    return df

def process_sequences(sequences_dir='sequences_multi', output_dir='data_split'):
    """
    Load sequences, add noise, split into train/val/test, and save
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all sequence files
    sequence_files = sorted(glob.glob(f'{sequences_dir}/sequence_*.csv'))
    
    if len(sequence_files) == 0:
        print(f"No sequence files found in {sequences_dir}")
        return
    
    print(f"Found {len(sequence_files)} sequences in {sequences_dir}")
    
    # Split sequences into train/val/test (70/20/10)
    train_files, temp_files = train_test_split(sequence_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=1/3, random_state=42)
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test sequences")
    
    # Process each split
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        if len(files) == 0:
            print(f"Warning: No files for {split_name} split")
            continue
        
        print(f"\nProcessing {split_name} split ({len(files)} sequences)...")
        
        # Load and combine sequences
        split_data = []
        for file in files:
            df = pd.read_csv(file)
            # Add noise only to train and val, not test (to keep test clean)
            if split_name in ['train', 'val']:
                df = add_gaussian_noise(df, column='Voltage', noise_factor=0.02)
            split_data.append(df)
        
        combined_df = pd.concat(split_data, ignore_index=True)
        
        # Save to CSV
        output_file = f"{output_dir}/{split_name}.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"Saved {split_name}.csv: {len(combined_df)} rows, {combined_df['Time'].max()/60:.1f} min")
        
        # Show label distribution
        if 'Label' in combined_df.columns:
            print(f"Label distribution: {combined_df['Label'].value_counts().sort_index().to_dict()}")

if __name__ == "__main__":
    print("🚀 Data Augmentation and Splitting")
    print("=" * 50)
    
    # Process multi-label sequences
    process_sequences(sequences_dir='sequences_multi', output_dir='data_split')
    
    print("\n✅ Data augmentation and splitting complete!")