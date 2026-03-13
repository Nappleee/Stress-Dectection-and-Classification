import pandas as pd
import numpy as np
import os
import random
from typing import List, Dict, Tuple

class ECGAdvancedConcatenator:
    """
    Advanced ECG Concatenator với:
    1. Preserve TIME khi nối files
    2. Nối sequences từ các label khác nhau
    """
    
    def __init__(self, csv_label_file: str, data_dir: str):
        """Initialize concatenator"""
        self.csv_label_file = csv_label_file
        self.data_dir = data_dir
        self.file_label_map = {}
        self.label_data = {}
        
        self._load_label_mapping()
    
    def _load_label_mapping(self):
        """Load label mapping từ CSV"""
        print("📂 Loading label mapping...")
        df_labels = pd.read_csv(self.csv_label_file)
        
        for idx, row in df_labels.iterrows():
            filename = row['File']
            label = row['Label']
            self.file_label_map[filename] = label
        
        print(f"✅ Loaded {len(self.file_label_map)} files\n")
    
    def _load_label_files(self, label: int):
        """Load tất cả files của một label"""
        if label in self.label_data:
            return
        
        print(f"📂 Loading Label {label} files...")
        
        label_files = [f for f in self.file_label_map.keys() 
                       if self.file_label_map[f] == label]
        label_files = sorted(label_files)
        
        label_data = {}
        for filename in label_files:
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            label_data[filename] = df
            print(f"  ✅ {filename}: {len(df)} rows")
        
        self.label_data[label] = label_data
        print()
    
    def _get_duration_from_dataframe(self, df: pd.DataFrame) -> float:
        """Calculate duration from dataframe"""
        if 'Time' in df.columns:
            return df['Time'].max() - df['Time'].min()
        else:
            return len(df)
    
    # ============================================================
    # METHOD 1: Concatenate files bảo tồn TIME
    # ============================================================
    
    def concatenate_preserve_time(self, 
                                  label: int, 
                                  duration_minutes: float,
                                  random_order: bool = True) -> pd.DataFrame:
        """
        Nối files của một label NHƯNG bảo tồn TIME column
        (TIME không reset về 0, liên tục tăng)
        
        Args:
            label (int): Label number
            duration_minutes (float): Target duration in minutes
            random_order (bool): Shuffle files
            
        Returns:
            pd.DataFrame: Concatenated data with preserved TIME
        """
        
        self._load_label_files(label)
        
        target_duration = duration_minutes * 60
        label_files = list(self.label_data[label].keys())
        
        print("=" * 80)
        print(f"🔗 CONCATENATING {duration_minutes:.0f}min LABEL {label} (PRESERVE TIME)")
        print("=" * 80)
        print(f"Target duration: {target_duration:.0f}s\n")
        
        # Shuffle if needed
        if random_order:
            shuffled_files = label_files.copy()
            random.shuffle(shuffled_files)
        else:
            shuffled_files = sorted(label_files)
        
        # Concatenate with TIME adjustment
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
            
            # PRESERVE TIME: Adjust TIME column
            if 'Time' in df.columns:
                # Get original time range
                original_time_min = df['Time'].min()
                original_time_max = df['Time'].max()
                
                # Shift TIME to be continuous
                df['Time'] = df['Time'] - original_time_min + global_time_offset
                
                # Update offset for next file
                global_time_offset += (original_time_max - original_time_min)
            
            # Add metadata
            df['Segment_Index'] = file_idx
            df['Segment_File'] = filename
            
            concatenated_data.append(df)
            concatenated_files.append(filename)
            current_duration += file_duration
            
            print(f"  {filename:25s}: +{file_duration:8.2f}s | Time: {df['Time'].min():8.2f}s → {df['Time'].max():8.2f}s")
        
        print("-" * 80)
        
        # Combine all dataframes
        combined_df = pd.concat(concatenated_data, ignore_index=True)
        
        print(f"\n✅ Concatenation Complete!")
        print(f"   Files: {len(concatenated_files)}")
        print(f"   Total rows: {len(combined_df)}")
        print(f"   Total duration: {combined_df['Time'].max():.2f}s ({combined_df['Time'].max()/60:.2f} min)")
        print(f"   Time preserved: {combined_df['Time'].min():.2f}s → {combined_df['Time'].max():.2f}s\n")
        
        # Add final metadata
        combined_df['Label'] = label
        combined_df['Sequence_ID'] = 0
        
        return combined_df
    
    # ============================================================
    # METHOD 2: Concatenate sequences từ multiple labels
    # ============================================================
    
    def create_multi_label_sequences(self,
                                    label_configs: Dict[int, int],
                                    num_sequences: int,
                                    random_order: bool = True) -> List[pd.DataFrame]:
        """
        Tạo multiple sequences bằng cách nối từng segment từ các label khác nhau
        
        Ví dụ:
        - Create 10 sequences
        - Mỗi sequence: 10 records từ Label 0 + 10 records từ Label 1 + 5 records từ Label 2
        - TIME liên tục (không reset)
        - Label column chỉ ra stress level của từng phần
        
        Args:
            label_configs (Dict[int, int]): {label: num_files_per_sequence}
                                           Example: {0: 10, 1: 10, 2: 5}
            num_sequences (int): Số sequences cần tạo
            random_order (bool): Shuffle files
            
        Returns:
            List[pd.DataFrame]: List of sequences
        """
        
        print("=" * 80)
        print(f"🔗 CREATING {num_sequences} MULTI-LABEL SEQUENCES")
        print("=" * 80)
        print(f"Label config: {label_configs}")
        print(f"Number of sequences: {num_sequences}\n")
        
        # Load all label files
        for label in label_configs.keys():
            self._load_label_files(label)
        
        sequences = []
        
        for seq_num in range(num_sequences):
            print(f"\n📌 Creating Sequence {seq_num + 1}/{num_sequences}")
            print("-" * 80)
            
            sequence_data = []
            global_time_offset = 0
            current_label_in_seq = 0
            
            # Process each label in order
            for label in sorted(label_configs.keys()):
                num_files = label_configs[label]
                
                # Get random files from this label
                label_files = list(self.label_data[label].keys())
                selected_files = random.sample(label_files, min(num_files, len(label_files)))
                
                print(f"  Label {label}: {num_files} files")
                
                for file_idx, filename in enumerate(selected_files):
                    df = self.label_data[label][filename].copy()
                    
                    # PRESERVE TIME
                    if 'Time' in df.columns:
                        original_time_min = df['Time'].min()
                        original_time_max = df['Time'].max()
                        
                        df['Time'] = df['Time'] - original_time_min + global_time_offset
                        
                        global_time_offset += (original_time_max - original_time_min)
                    
                    # Add metadata
                    df['Segment_Index'] = file_idx
                    df['Segment_File'] = filename
                    df['Label'] = label  # Stress level
                    df['Label_Sequence'] = current_label_in_seq
                    
                    sequence_data.append(df)
                
                current_label_in_seq += 1
            
            # Combine sequence
            if sequence_data:
                combined_seq = pd.concat(sequence_data, ignore_index=True)
                combined_seq['Sequence_ID'] = seq_num
                sequences.append(combined_seq)
                
                print(f"  ✅ Rows: {len(combined_seq)} | Time: {combined_seq['Time'].min():.0f}s → {combined_seq['Time'].max():.0f}s ({combined_seq['Time'].max()/60:.1f} min)")
        
        print(f"\n✅ Created {len(sequences)} sequences!\n")
        
        return sequences
    
    # ============================================================
    # UTILITIES
    # ============================================================
    
    def display_sequence_info(self, df: pd.DataFrame, seq_id: int = None):
        """Display detailed info về sequence"""
        print("=" * 80)
        print(f"📊 SEQUENCE INFO{f' #{seq_id}' if seq_id else ''}")
        print("=" * 80)
        
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if 'Time' in df.columns:
            print(f"\nTime range: {df['Time'].min():.2f}s → {df['Time'].max():.2f}s ({df['Time'].max()/60:.2f} min)")
            print(f"Time preserved: {'✅ YES' if df['Time'].min() >= 0 and df['Time'].is_monotonic_increasing else '❌ NO'}")
        
        if 'Label' in df.columns:
            print(f"\nLabel distribution:")
            print(df['Label'].value_counts().sort_index())
        
        if 'HR(bpm)' in df.columns:
            print(f"\nHR range: {df['HR(bpm)'].min():.1f} - {df['HR(bpm)'].max():.1f} bpm (avg: {df['HR(bpm)'].mean():.1f})")
        
        if 'Segment_File' in df.columns:
            print(f"\nFiles used: {df['Segment_File'].nunique()}")
            print(f"  {df['Segment_File'].unique()}")
        
        print()
    
    def save_sequences(self, sequences: List[pd.DataFrame], output_dir: str = 'sequences'):
        """Save tất cả sequences"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"💾 Saving {len(sequences)} sequences to {output_dir}/")
        
        output_files = []
        
        for seq_id, df in enumerate(sequences):
            filename = f"{output_dir}/sequence_{seq_id:03d}.csv"
            df.to_csv(filename, index=False)
            output_files.append(filename)
            print(f"  ✅ {filename} ({len(df)} rows)")
        
        print(f"\n✅ Saved {len(sequences)} sequences\n")
        
        return output_files
    
    def combine_sequences(self, sequences: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine tất cả sequences vào 1 dataframe"""
        print("🔗 Combining all sequences...")
        
        combined = pd.concat(sequences, ignore_index=True)
        
        print(f"✅ Combined: {len(combined)} rows\n")
        
        return combined


# ============================================================
# USAGE EXAMPLES
# ============================================================

if __name__ == "__main__":
    
    # Initialize
    concat = ECGAdvancedConcatenator(
        csv_label_file='data/hrv_features_label.csv',
        data_dir='data/raw_gen'
    )
    
    # ========== EXAMPLE 1: Preserve TIME khi nối files ==========
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Concatenate 30min Label 0 with PRESERVED TIME")
    print("=" * 80)
    
    df_preserved = concat.concatenate_preserve_time(
        label=0,
        duration_minutes=30,
        random_order=True
    )
    
    concat.display_sequence_info(df_preserved, seq_id=1)
    df_preserved.to_csv('sequence_label0_preserved_time.csv', index=False)
    
    # ========== EXAMPLE 2: Tạo 5 sequences từ Label 0 riêng ==========
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Create 5 sequences từ Label 0 only")
    print("=" * 80)
    
    sequences_label0 = concat.create_multi_label_sequences(
        label_configs={0: 6},  # 6 files từ label 0 per sequence
        num_sequences=5,
        random_order=True
    )
    
    for idx, seq in enumerate(sequences_label0):
        concat.display_sequence_info(seq, seq_id=idx)
    
    concat.save_sequences(sequences_label0, output_dir='sequences_label0')
    
    # ========== EXAMPLE 3: Create 10 sequences từ Label 1 ==========
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Create 10 sequences từ Label 1 only")
    print("=" * 80)
    
    sequences_label1 = concat.create_multi_label_sequences(
        label_configs={1: 5},  # 5 files từ label 1 per sequence
        num_sequences=10,
        random_order=True
    )
    
    for idx, seq in enumerate(sequences_label1[:3]):  # Show first 3
        concat.display_sequence_info(seq, seq_id=idx)
    print(f"... ({len(sequences_label1) - 3} more sequences)")
    
    concat.save_sequences(sequences_label1, output_dir='sequences_label1')
    
    # ========== EXAMPLE 4: Create sequences nối từ Label 0 + Label 1 ==========
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Create 10 sequences: Label0 + Label1 (MIX)")
    print("=" * 80)
    
    sequences_mixed = concat.create_multi_label_sequences(
        label_configs={0: 10, 1: 10},  # 10 files label 0, then 10 files label 1
        num_sequences=10,
        random_order=True
    )
    
    for idx, seq in enumerate(sequences_mixed[:2]):  # Show first 2
        concat.display_sequence_info(seq, seq_id=idx)
    print(f"... ({len(sequences_mixed) - 2} more sequences)")
    
    concat.save_sequences(sequences_mixed, output_dir='sequences_mixed_0_1')
    
    # ========== EXAMPLE 5: Create sequences từ TẤT CẢ 3 labels ==========
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Create 5 sequences: Label0 + Label1 + Label2 (FULL)")
    print("=" * 80)
    
    sequences_full = concat.create_multi_label_sequences(
        label_configs={0: 5, 1: 5, 2: 3},  # Mix từ 3 labels
        num_sequences=5,
        random_order=True
    )
    
    for idx, seq in enumerate(sequences_full):
        concat.display_sequence_info(seq, seq_id=idx)
    
    concat.save_sequences(sequences_full, output_dir='sequences_full_mix')
    
    # ========== EXAMPLE 6: Combine tất cả sequences ==========
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Combine all sequences")
    print("=" * 80)
    
    all_sequences = sequences_label0 + sequences_label1 + sequences_mixed + sequences_full
    combined_all = concat.combine_sequences(all_sequences)
    
    print(f"Total rows: {len(combined_all)}")
    print(f"Label distribution:")
    print(combined_all['Label'].value_counts().sort_index())
    
    combined_all.to_csv('all_sequences_combined.csv', index=False)