import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import glob

def add_gaussian_noise(df, noise_std=0.02):
    """
    Add Gaussian noise to Voltage column.
    noise_std: fixed std dev for noise.
    """
    if 'Voltage' not in df.columns:
        return df
    
    noise = np.random.normal(0, noise_std, size=len(df))
    df = df.copy()
    df['Voltage'] += noise
    return df

def process_sequences(input_dir='data/sequences_by_hour', output_dir='data/sequences_noisy', noise_std=0.02):
    """
    Add noise to all sequences and save noisy versions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sequence_files = glob.glob(f'{input_dir}/hour_*/v*.csv')
    print(f"Found {len(sequence_files)} sequences to process")
    
    for file_path in sequence_files:
        df = pd.read_csv(file_path)
        df = df.astype({
            'Time': float,
            'Voltage': float,
            'Peak': int,
            'Label': int
        })
        df_noisy = add_gaussian_noise(df, noise_std)
        
        # Keep only essential columns
        columns_to_keep = ['Time', 'Voltage', 'Peak', 'Label']
        df_noisy = df_noisy[columns_to_keep]
        
        # Output path
        rel_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df_noisy.to_csv(output_path, index=False)
        print(f"Processed {rel_path} → {output_path}")

def split_sequences(input_dir='data/sequences_noisy', output_dir='data/ml_splits', test_size=0.2, val_size=0.2):
    """
    Split all sequences into train/val/test sets.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sequence_files = glob.glob(f'{input_dir}/hour_*/v*.csv')
    print(f"Splitting {len(sequence_files)} sequences")
    
    # Collect all data
    all_data = []
    for file_path in sequence_files:
        df = pd.read_csv(file_path)
        df = df.astype({
            'Time': float,
            'Voltage': float,
            'Peak': int,
            'Label': int
        })
        # Add sequence identifier
        seq_id = os.path.basename(file_path).replace('.csv', '')
        hour = os.path.basename(os.path.dirname(file_path))
        df['Sequence_ID'] = f"{hour}_{seq_id}"
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total combined data: {len(combined_df)} rows")
    
    # Split by sequences to avoid data leakage
    sequence_ids = combined_df['Sequence_ID'].unique()
    print(f"Unique sequences: {len(sequence_ids)}")
    
    # First split: train + (val+test)
    train_seqs, temp_seqs = train_test_split(sequence_ids, test_size=(test_size + val_size), random_state=42)
    
    # Second split: val and test
    val_seqs, test_seqs = train_test_split(temp_seqs, test_size=test_size/(test_size + val_size), random_state=42)
    
    print(f"Train sequences: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
    
    # Filter data
    train_df = combined_df[combined_df['Sequence_ID'].isin(train_seqs)].drop(columns=['Sequence_ID'])
    val_df = combined_df[combined_df['Sequence_ID'].isin(val_seqs)].drop(columns=['Sequence_ID'])
    test_df = combined_df[combined_df['Sequence_ID'].isin(test_seqs)].drop(columns=['Sequence_ID'])
    
    # Save
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    print(f"Saved train.csv: {len(train_df)} rows")
    print(f"Saved val.csv: {len(val_df)} rows")
    print(f"Saved test.csv: {len(test_df)} rows")

if __name__ == "__main__":
    print("Starting data augmentation...")
    
    # Add noise
    process_sequences(noise_std=0.02)
    
    print("✅ Noisy sequences created!")
    # Split can be done separately if needed