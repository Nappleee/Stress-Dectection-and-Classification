import pandas as pd
import numpy as np
import os
import random
from typing import List, Dict, Union

class ECGAdvancedConcatenator:
    """
    Advanced ECG Concatenator with proper column handling
    """
    
    def __init__(self, csv_label_file: str, data_dir: str):
        """Initialize concatenator"""
        self.csv_label_file = csv_label_file
        self.data_dir = data_dir
        self.file_label_map = {}
        self.label_data = {}
        
        # Validate files
        if not os.path.exists(csv_label_file):
            raise FileNotFoundError(f"❌ CSV file not found: {csv_label_file}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")
        
        self._load_label_mapping()
    
    def _load_label_mapping(self):
        """Load label mapping with proper column cleanup"""
        print("📂 Loading label mapping...")
        df_labels = pd.read_csv(self.csv_label_file)
        
        # ✅ FIX: Strip spaces from ALL column names
        df_labels.columns = df_labels.columns.str.strip()
        
        print(f"✅ Cleaned columns: {list(df_labels.columns)}\n")
        
        for idx, row in df_labels.iterrows():
            filename = row['File'].strip()
            label = row['Label']  # Now clean!
            self.file_label_map[filename] = label
        
        print(f"✅ Loaded {len(self.file_label_map)} files\n")
        
        # Display sample
        print(f"Sample mapping:")
        for i, (fname, lbl) in enumerate(list(self.file_label_map.items())[:3]):
            print(f"  {fname} → Label {lbl}")
        print()
    
    def _load_label_files(self, label: int):
        """Load files of a specific label"""
        if label in self.label_data:
            return
        
        print(f"📂 Loading Label {label} files...")
        
        label_files = [f for f in self.file_label_map.keys() 
                       if self.file_label_map[f] == label]
        label_files = sorted(label_files)
        
        if len(label_files) == 0:
            print(f"⚠️  No files found for Label {label}!")
            return
        
        label_data = {}
        for filename in label_files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                # Also clean data file columns
                df.columns = df.columns.str.strip()
                label_data[filename] = df
                print(f"  ✅ {filename}: {len(df)} rows")
            except FileNotFoundError:
                print(f"  ❌ {filename}: NOT FOUND!")
            except Exception as e:
                print(f"  ❌ {filename}: ERROR - {str(e)}")
        
        self.label_data[label] = label_data
        print()
    
    def _get_duration_from_dataframe(self, df: pd.DataFrame) -> float:
        """Calculate duration from Time column"""
        if 'Time' in df.columns:
            return df['Time'].max() - df['Time'].min()
        else:
            return len(df)
    
    def concatenate_preserve_time(self, 
                                  label: int, 
                                  duration_minutes: float,
                                  random_order: bool = True) -> pd.DataFrame:
        """
        Concatenate files while preserving TIME column (no reset)
        """
        
        self._load_label_files(label)
        
        if label not in self.label_data or len(self.label_data[label]) == 0:
            raise ValueError(f"No data found for Label {label}")
        
        target_duration = duration_minutes * 60
        label_files = list(self.label_data[label].keys())
        
        print("=" * 80)
        print(f"🔗 CONCATENATING {duration_minutes:.0f}min LABEL {label}")
        print("=" * 80)
        print(f"Target: {target_duration:.0f}s ({duration_minutes:.1f} min)\n")
        
        if random_order:
            shuffled_files = label_files.copy()
            random.shuffle(shuffled_files)
        else:
            shuffled_files = sorted(label_files)
        
        concatenated_data = []
        concatenated_files = []
        current_duration = 0
        global_time_offset = 0
        
        print(f"🔄 Concatenating files...")
        print("-" * 80)
        
        for file_idx, filename in enumerate(shuffled_files):
            if current_duration >= target_duration:
                break
            
            df = self.label_data[label][filename].copy()
            file_duration = self._get_duration_from_dataframe(df)
            
            # Preserve TIME: adjust so it's continuous
            if 'Time' in df.columns:
                original_time_min = df['Time'].min()
                original_time_max = df['Time'].max()
                
                # Shift TIME to continue from previous offset
                df['Time'] = df['Time'] - original_time_min + global_time_offset
                
                # Update offset for next file
                global_time_offset += (original_time_max - original_time_min)
            
            # Add metadata
            df['Segment_Index'] = file_idx
            df['Segment_File'] = filename
            
            concatenated_data.append(df)
            concatenated_files.append(filename)
            current_duration += file_duration
            
            print(f"  {filename:30s}: +{file_duration:8.2f}s | Time: {df['Time'].min():8.2f}s → {df['Time'].max():8.2f}s")
        
        print("-" * 80)
        
        # Combine all
        combined_df = pd.concat(concatenated_data, ignore_index=True)
        
        print(f"\n✅ Concatenation Complete!")
        print(f"   Files: {len(concatenated_files)}")
        print(f"   Total rows: {len(combined_df)}")
        print(f"   Total duration: {combined_df['Time'].max():.2f}s ({combined_df['Time'].max()/60:.2f} min)")
        print(f"   Time range: {combined_df['Time'].min():.2f}s → {combined_df['Time'].max():.2f}s\n")
        
        combined_df['Label'] = label
        combined_df['Sequence_ID'] = 0
        
        # Filter columns to keep only essential ones
        columns_to_keep = ['Time', 'Voltage', 'Peak', 'Label']
        combined_df = combined_df[columns_to_keep]
        
        # Trim to target duration if exceeded
        if combined_df['Time'].max() > target_duration:
            combined_df = combined_df[combined_df['Time'] <= target_duration].copy()
            print(f"   Trimmed to: {combined_df['Time'].max():.2f}s ({combined_df['Time'].max()/60:.2f} min)\n")
        
        return combined_df
    
    def create_multi_label_sequences(self,
                                    label_configs: Dict[int, int],
                                    num_sequences: int,
                                    target_duration_minutes: float = 30.0,
                                    random_order: bool = True) -> List[pd.DataFrame]:
        """
        Create multiple sequences by concatenating different labels
        Example: {0: 10, 1: 10, 2: 5} = 10 files label0 + 10 files label1 + 5 files label2
        """
        
        target_duration = target_duration_minutes * 60
        
        print("=" * 80)
        print(f"🔗 CREATING {num_sequences} MULTI-LABEL SEQUENCES")
        print("=" * 80)
        print(f"Label config: {label_configs}")
        print(f"Sequences: {num_sequences}")
        print(f"Target duration: {target_duration:.0f}s ({target_duration_minutes:.1f} min)\n")
        
        # Load all label files
        for label in label_configs.keys():
            self._load_label_files(label)
        
        sequences = []
        
        for seq_num in range(num_sequences):
            print(f"\n📌 Sequence {seq_num + 1}/{num_sequences}")
            print("-" * 80)
            
            sequence_data = []
            global_time_offset = 0
            current_label_in_seq = 0
            
            # Process each label in order
            for label in sorted(label_configs.keys()):
                num_files = label_configs[label]
                
                label_files = list(self.label_data[label].keys())
                selected_files = random.sample(label_files, min(num_files, len(label_files)))
                
                print(f"  Label {label}: {len(selected_files)} files")
                
                for file_idx, filename in enumerate(selected_files):
                    df = self.label_data[label][filename].copy()
                    
                    # Preserve TIME
                    if 'Time' in df.columns:
                        original_time_min = df['Time'].min()
                        original_time_max = df['Time'].max()
                        
                        df['Time'] = df['Time'] - original_time_min + global_time_offset
                        
                        global_time_offset += (original_time_max - original_time_min)
                    
                    # Add metadata
                    df['Segment_Index'] = file_idx
                    df['Segment_File'] = filename
                    df['Label'] = label
                    df['Label_Sequence'] = current_label_in_seq
                    
                    sequence_data.append(df)
                
                current_label_in_seq += 1
            
            # Combine sequence
            if sequence_data:
                combined_seq = pd.concat(sequence_data, ignore_index=True)
                combined_seq['Sequence_ID'] = seq_num
                
                # Filter columns to keep only essential ones
                columns_to_keep = ['Time', 'Voltage', 'Peak', 'Label']
                combined_seq = combined_seq[columns_to_keep]
                
                # Trim to target duration if exceeded
                if combined_seq['Time'].max() > target_duration:
                    combined_seq = combined_seq[combined_seq['Time'] <= target_duration].copy()
                    print(f"  Trimmed to: {combined_seq['Time'].max():.0f}s ({combined_seq['Time'].max()/60:.1f} min)")
                
                sequences.append(combined_seq)
                
                print(f"  ✅ Rows: {len(combined_seq)} | Time: {combined_seq['Time'].min():.0f}s → {combined_seq['Time'].max():.0f}s ({combined_seq['Time'].max()/60:.1f} min)")
        
        print(f"\n✅ Created {len(sequences)} sequences!\n")
        
        return sequences
    
    def display_sequence_info(self, df: pd.DataFrame, seq_id: int = None):
        """Display sequence information"""
        print("=" * 80)
        print(f"📊 SEQUENCE INFO{f' #{seq_id}' if seq_id is not None else ''}")
        print("=" * 80)
        
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if 'Time' in df.columns:
            print(f"\nTime: {df['Time'].min():.2f}s → {df['Time'].max():.2f}s ({df['Time'].max()/60:.2f} min)")
            is_monotonic = df['Time'].is_monotonic_increasing
            print(f"Time preserved: {'✅ YES' if is_monotonic else '❌ NO'}")
        
        if 'Label' in df.columns:
            print(f"\nLabel distribution:")
            print(df['Label'].value_counts().sort_index())
        
        if 'HR(bpm)' in df.columns:
            print(f"\nHR: {df['HR(bpm)'].min():.1f} - {df['HR(bpm)'].max():.1f} bpm (avg: {df['HR(bpm)'].mean():.1f})")
        
        if 'Segment_File' in df.columns:
            print(f"\nFiles used: {df['Segment_File'].nunique()}")
        
        print()
    
    def save_sequences(self, sequences: List[pd.DataFrame], output_dir: str = 'sequences'):
        """Save all sequences to CSV files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"💾 Saving {len(sequences)} sequences to {output_dir}/")
        print("-" * 80)
        
        for seq_id, df in enumerate(sequences):
            filename = f"{output_dir}/sequence_{seq_id:03d}.csv"
            df.to_csv(filename, index=False)
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f"  ✅ sequence_{seq_id:03d}.csv ({len(df)} rows, {file_size:.2f} MB)")
        
        print(f"\n✅ Saved {len(sequences)} sequences!\n")
        
        return [f"{output_dir}/sequence_{i:03d}.csv" for i in range(len(sequences))]
    
    def combine_sequences(self, sequences: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine all sequences into one dataframe"""
        print("🔗 Combining all sequences...")
        
        combined = pd.concat(sequences, ignore_index=True)
        
        print(f"✅ Combined: {len(combined)} rows\n")
        
        return combined
    
    def create_hourly_sequences(self, hour_label_map: Dict[int, Union[int, Dict[int, int]]], num_variations: int = 2, output_base_dir: str = 'data/sequences_by_hour'):
        """
        Create hourly sequences from 6 AM to 11 PM (hours 6-23).
        hour_label_map: {hour: label} where label is int for single label or dict for multi-label config {label: num_files}
        num_variations: Number of variations per hour.
        """
        hours = list(range(6, 24))  # 6 AM to 11 PM
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)
        
        global_time_offset = 0  # To keep time continuous across all sequences
        
        for hour in hours:
            hour_dir = f"{output_base_dir}/hour_{hour:02d}"
            os.makedirs(hour_dir, exist_ok=True)
            
            config = hour_label_map.get(hour, 0)
            print(f"Creating {num_variations} variations for hour {hour} (Config: {config})")
            
            for var in range(1, num_variations + 1):
                if isinstance(config, int):
                    # Single label
                    seq_df = self.concatenate_preserve_time(label=config, duration_minutes=60, random_order=True)
                elif isinstance(config, dict):
                    # Multi-label
                    sequences = self.create_multi_label_sequences(
                        label_configs=config, 
                        num_sequences=1, 
                        target_duration_minutes=60, 
                        random_order=True
                    )
                    seq_df = sequences[0] if sequences else pd.DataFrame()
                else:
                    print(f"Invalid config for hour {hour}: {config}")
                    continue
                
                if seq_df.empty:
                    print(f"  Skipping hour {hour} v{var}: no data")
                    continue
                
                # Shift Time to be continuous globally
                if 'Time' in seq_df.columns:
                    seq_df['Time'] += global_time_offset
                    global_time_offset = seq_df['Time'].max()
                
                seq_df['Sequence_ID'] = f"hour_{hour}_v{var}"
                
                filename = f"{hour_dir}/v{var}.csv"
                seq_df.to_csv(filename, index=False)
                print(f"  Saved {filename}: {len(seq_df)} rows, Time: {seq_df['Time'].min():.0f}s → {seq_df['Time'].max():.0f}s")
        
        print(f"\n✅ Created hourly sequences in {output_base_dir}/\n")
    
    def save_to_csv(self, df: pd.DataFrame, output_file: str):
        """Save dataframe to CSV file"""
        df.to_csv(output_file, index=False)
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print("=" * 80)
        print(f"✅ SAVED: {output_file}")
        print("=" * 80)
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"File size: {file_size_mb:.2f} MB\n")

# ============================================================
# MAIN - USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    
    try:
        print("=" * 80)
        print("🚀 ECG CONCATENATOR - STRESS CLASSIFICATION")
        print("=" * 80 + "\n")
        
        # ========== INITIALIZE ==========
        concat = ECGAdvancedConcatenator(
            csv_label_file='data/hrv_features_label.csv',
            data_dir='data/raw_gen'
        )
        
        # ========== HOURLY SEQUENCES ==========
        print("\n" + "=" * 80)
        print("CREATING HOURLY SEQUENCES (6 AM - 11 PM)")
        print("=" * 80)
        
        # Define stress levels by hour
        hour_label_map = {
            # Low stress: label 0
            6: 0, 7: 0, 8: 0, 12: 0, 13: 0, 22: 0, 23: 0,
            # Medium stress: mix 0 & 1
            9: {0: 10, 1: 5}, 10: {0: 10, 1: 5}, 11: {0: 10, 1: 5}, 
            14: {0: 10, 1: 5}, 15: {0: 10, 1: 5}, 16: {0: 10, 1: 5},
            # High stress: mix 1 & 2
            17: {1: 5, 2: 10}, 18: {1: 5, 2: 10}, 19: {1: 5, 2: 10}, 
            20: {1: 5, 2: 10}, 21: {1: 5, 2: 10}
        }
        
        concat.create_hourly_sequences(
            hour_label_map=hour_label_map,
            num_variations=2,
            output_base_dir='data/sequences_by_hour'
        )
        
        print("✅ HOURLY SEQUENCES CREATED!\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()