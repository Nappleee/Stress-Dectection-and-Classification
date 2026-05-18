import nbformat
import sys

def modify_notebook_1(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    new_cells = []
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            if 'def clean_waveform_group' in source or 'def safe_skew' in source or 'def rr_frequency_features' in source or 'def extract_features_from_window' in source:
                continue 
        new_cells.append(cell)
    
    import_cell = nbformat.v4.new_code_cell("from src.features import extract_features_from_window")
    new_cells.insert(2, import_cell) 
    
    nb.cells = new_cells
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def modify_notebook_2(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    new_cells = []
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            if 'def find_label_column' in source or 'def standardize_features_df' in source:
                continue
            if 'def pick_best_group_split' in source:
                pass
            if 'def make_pipeline' in source:
                pass
            if 'def gini' in source or 'class Node' in source or 'class DecisionTree' in source or 'class RandomForest' in source:
                continue
            if 'def run_random_forest' in source:
                continue
        new_cells.append(cell)
    
    nb.cells = new_cells
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    modify_notebook_1(r'C:\Users\buck\Napplee\StressClassification\train_part1_features_ml.ipynb')
    print("Modified notebook 1")