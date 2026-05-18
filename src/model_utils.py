import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

def pick_best_group_split(X_in, y_in, groups_in, test_size=0.2, n_trials=40, random_state=42):
    best = None
    best_score = -1.0
    y_all = pd.Series(y_in)
    global_dist = y_all.value_counts(normalize=True).sort_index()
    for seed in range(random_state, random_state + n_trials):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(X_in, y_in, groups=groups_in))
        y_te = pd.Series(y_in[te_idx])
        dist_te = y_te.value_counts(normalize=True).sort_index()
        # Bo qua split neu thieu lop trong test
        if dist_te.shape[0] < global_dist.shape[0]:
            continue
        # Score cao neu phan bo test gan phan bo global va min class ratio cao
        aligned = pd.concat([global_dist, dist_te], axis=1).fillna(0.0)
        aligned.columns = ["global", "test"]
        l1_gap = (aligned["global"] - aligned["test"]).abs().sum()
        min_ratio = float(dist_te.min())
        score = (1.0 - l1_gap) + min_ratio
        if score > best_score:
            best_score = score
            best = (tr_idx, te_idx, seed, dist_te)
    return best

def make_pipeline(model, preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
