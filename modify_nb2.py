import nbformat
import re

def modify_notebook_2(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    new_cells = []
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            # 1) find_label_column / standardize_features_df
            if 'def find_label_column' in source:
                # We replace everything from LABEL_CANDIDATES down to the 'if not FEATURE_FILE.exists():'
                # with just an import.
                new_src = re.sub(
                    r'LABEL_CANDIDATES.*?def standardize_features_df.*?\s+return out\n+',
                    'from src.data_prep import standardize_features_df, find_label_column\n\n',
                    source,
                    flags=re.DOTALL
                )
                cell.source = new_src
                
            # 2) pick_best_group_split
            if 'def pick_best_group_split' in source:
                new_src = re.sub(
                    r'def pick_best_group_split\(.*?return best\n+',
                    'from src.model_utils import pick_best_group_split\n\n',
                    cell.source,
                    flags=re.DOTALL
                )
                cell.source = new_src
                
            # 3) make_pipeline
            if 'def make_pipeline' in source:
                new_src = re.sub(
                    r'def make_pipeline\(model\):\n\s+return Pipeline\(\[\n\s+\("preprocessor", preprocessor\),\n\s+\("model", model\),\n\s+\]\)\n',
                    'from src.model_utils import make_pipeline\n\n',
                    cell.source,
                    flags=re.DOTALL
                )
                cell.source = new_src
                
            # 4) gini, Node, DecisionTree, RandomForest
            if 'def gini' in source and 'class Node' in source:
                continue # Skip this cell entirely
                
            # 5) run_random_forest
            if 'def run_random_forest' in source:
                cell.source = 'from src.custom_rf import run_random_forest\n'
                
        new_cells.append(cell)
        
    nb.cells = new_cells
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    modify_notebook_2(r'C:\Users\buck\Napplee\StressClassification\train_part2_modeling_ml.ipynb')
    print("Modified notebook 2")
