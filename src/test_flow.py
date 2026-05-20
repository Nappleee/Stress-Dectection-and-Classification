import pandas as pd
import numpy as np
from pathlib import Path
from data_prep import standardize_features_df, find_label_column
from sklearn.model_selection import train_test_split
from custom_rf import run_random_forest, grid_search_custom_rf

def main():
    feature_file = Path("data/features/5/features_windowed_balanced_256s_id5.csv")
    if not feature_file.exists():
        print(f"File not found: {feature_file}")
        print("Hãy tạo file features mới từ concateFile.ipynb trước.")
        return

    print("=== 1. Data Preparation ===")
    df = pd.read_csv(feature_file)
    print(f"Original shape: {df.shape}")
    
    df = standardize_features_df(df)
    label_col = find_label_column(df)
    print(f"Shape after standardization: {df.shape}")
    print(f"Target label column: {label_col}")
    print("Class distribution:")
    print(df[label_col].value_counts())

    y = df[label_col].values
    
    # Drop non-feature columns
    X_df = df.drop(columns=["source_file", "window_id", label_col], errors="ignore")
    drop_cols = ["window_start", "window_end", "duration_sec", "n_peaks"]
    X_df = X_df.drop(columns=drop_cols, errors="ignore")
    
    X = X_df.values
    feature_names = X_df.columns.tolist()
    print(f"\nFeature matrix X shape: {X.shape}")
    
    print("\n=== 2. Train/Test Split (70/30) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    print("\n=== 3. Custom Random Forest Modeling with Grid Search ===")
    
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
    
    _, best_params, _ = grid_search_custom_rf(
        X_train, y_train, param_grid, n_splits=5
    )
    
    print("\n=== 4. Evaluating Best Model on Test Set ===")
    rf, y_pred = run_random_forest(
        X_train, y_train, 
        X_test, y_test, 
        n_estimators=best_params["n_estimators"], 
        max_depth=best_params["max_depth"], 
        random_state=best_params.get("random_state", 42),
        max_features=best_params.get("max_features"),
        max_leaf_nodes=best_params.get("max_leaf_nodes"),
        min_samples_leaf=best_params.get("min_samples_leaf", 1),
    )

if __name__ == "__main__":
    main()
