import numpy as np
import math
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold, ParameterGrid
from joblib import Parallel, delayed

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf

class DecisionTree:
    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        max_leaf_nodes=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self._leaf_count = 0
        self.root = self._grow_tree(X, y)

    def _best_split(self, X, y):
        best_gini = float("inf")
        split_idx, split_thr = None, None

        features = np.random.choice(
            self.n_features,
            self.max_features if self.max_features else self.n_features,
            replace=False,
        )

        for feature in features:
            feat_vals = X[:, feature]
            thresholds = np.unique(feat_vals)
            # Threshold binning optimization
            if len(thresholds) > 20:
                thresholds = np.percentile(feat_vals, np.linspace(0, 100, 22)[1:-1])
                
            for thr in thresholds:
                left = y[feat_vals <= thr]
                right = y[feat_vals > thr]

                if len(left) == 0 or len(right) == 0:
                    continue
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                g = (len(left) * gini(left) + len(right) * gini(right)) / len(y)

                if g < best_gini:
                    best_gini = g
                    split_idx = feature
                    split_thr = thr

        return split_idx, split_thr

    def _make_leaf(self, y):
        if self.max_leaf_nodes is None or self._leaf_count < self.max_leaf_nodes:
            self._leaf_count += 1
        return Node(value=np.bincount(y).argmax())

    def _grow_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_labels = len(np.unique(y))

        # stopping conditions
        min_samples_leaf = max(1, int(self.min_samples_leaf))
        min_samples_split = max(int(self.min_samples_split), 2 * min_samples_leaf)
        max_depth_reached = (self.max_depth is not None) and (depth >= self.max_depth)
        max_leaves_reached = (self.max_leaf_nodes is not None) and (self._leaf_count >= self.max_leaf_nodes)
        if (
            max_depth_reached
            or max_leaves_reached
            or num_labels == 1
            or num_samples < min_samples_split
        ):
            return self._make_leaf(y)

        feat, thr = self._best_split(X, y)
        if feat is None:
            return self._make_leaf(y)

        left_idx = X[:, feat] <= thr
        right_idx = X[:, feat] > thr

        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feat, thr, left, right)

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict(x, self.root) for x in X])

class RandomForest:
    def __init__(
        self,
        n_estimators,
        max_depth,
        random_state=None,
        max_features=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return max(1, int(math.sqrt(n_features)))
        if isinstance(self.max_features, str):
            key = self.max_features.lower()
            if key == "sqrt":
                return max(1, int(math.sqrt(n_features)))
            if key == "log2":
                return max(1, int(math.log2(n_features)))
            return n_features
        if isinstance(self.max_features, float):
            if 0 < self.max_features <= 1:
                return max(1, int(n_features * self.max_features))
            return n_features
        if isinstance(self.max_features, int):
            return max(1, min(n_features, int(self.max_features)))
        return n_features

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.trees = []
        self.classes_ = np.unique(y)
        n_samples = len(y)
        max_features = self._resolve_max_features(X.shape[1])

        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random

        def train_single_tree(seed):
            rng_tree = np.random.RandomState(seed)
            idx = rng_tree.choice(n_samples, n_samples, replace=True)
            X_sample = X[idx]
            y_sample = y[idx]

            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample, y_sample)
            return tree

        # Train trees in parallel
        seeds = rng.randint(0, 2**31, size=self.n_estimators)
        self.trees = Parallel(n_jobs=-1)(
            delayed(train_single_tree)(seed) for seed in seeds
        )

    def predict(self, X):
        X = np.asarray(X)
        # Predict in parallel
        preds = np.array(Parallel(n_jobs=-1, prefer="threads")(
            delayed(tree.predict)(X) for tree in self.trees
        ))
        return np.array([
            np.bincount(preds[:, i]).argmax()
            for i in range(X.shape[0])
        ])

    def predict_proba(self, X):
        X = np.asarray(X)
        # Predict in parallel
        preds = np.array(Parallel(n_jobs=-1, prefer="threads")(
            delayed(tree.predict)(X) for tree in self.trees
        ))
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i in range(X.shape[0]):
            votes = preds[:, i]
            for v in votes:
                proba[i, class_to_idx[v]] += 1.0
        proba /= max(len(self.trees), 1)
        return proba

def run_random_forest(X_train, y_train, X_test, y_test,
                      n_estimators=50,
                      max_depth=10,
                      random_state=42,
                      max_features=None,
                      max_leaf_nodes=None,
                      min_samples_leaf=1):
   

    rf = RandomForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
    )

    import time
    start = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = rf.predict(X_test)
    test_time = time.time() - start

    y_true = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro_val = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    r_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted_val = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    
    roc_auc = np.nan
    pr_auc = np.nan
    try:
        y_proba = rf.predict_proba(X_test)
        classes = np.unique(y_true)
        if len(classes) == 2:
            pos_idx = 1 if y_proba.shape[1] > 1 else 0
            score_pos = y_proba[:, pos_idx]
            roc_auc = roc_auc_score(y_true, score_pos)
            pr_auc = average_precision_score(y_true, score_pos)
        else:
            y_bin = label_binarize(y_true, classes=classes)
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            pr_auc = average_precision_score(y_bin, y_proba, average="weighted")
    except Exception:
        pass

    print("\n===== Random Forest (Scratch) =====")
    print(
        "n_estimators={0}, max_depth={1}, random_state={2}, max_features={3}, max_leaf_nodes={4}, min_samples_leaf={5}".format(
            n_estimators,
            max_depth,
            random_state,
            max_features,
            max_leaf_nodes,
            min_samples_leaf,
        )
    )
    print(f"Train time: {train_time:.4f}s | Test time: {test_time:.4f}s")
    print(
        f"ACC={acc:.4f} | BAL_ACC={bal_acc:.4f} | "
        f"Precision_macro={p_macro:.4f} | Recall_macro={r_macro:.4f} | "
        f"Precision_w={p_weighted:.4f} | Recall_w={r_weighted:.4f} | "
        f"F1_macro={f1_macro_val:.4f} | F1_weighted={f1_weighted_val:.4f} | "
        f"MCC={mcc:.4f} | Kappa={kappa:.4f} | "
        f"ROC_AUC={roc_auc:.4f} | PR_AUC={pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return rf, y_pred

def grid_search_custom_rf(X, y, param_grid, n_splits=5):
    
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = list(ParameterGrid(param_grid))
    
    best_score = -1
    best_params = None
    
    print(f"Starting Grid Search with {len(grid)} parameter combinations over {n_splits} folds...")
    
    for params in grid:
        fold_scores = []
        for train_idx, val_idx in kf.split(X_arr, y_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]
            
            rf = RandomForest(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            
            f1_macro_val = f1_score(y_val, y_pred, average="macro", zero_division=0)
            fold_scores.append(f1_macro_val)
            
        avg_score = np.mean(fold_scores)
        print(f"Params: {params} | CV F1_macro: {avg_score:.4f} (folds: {[round(s, 4) for s in fold_scores]})")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            
    print(f"\nBest Params: {best_params} | Best CV F1_macro: {best_score:.4f}")
    
    print("Training best model on all data...")
    best_model = RandomForest(**best_params)
    best_model.fit(X_arr, y_arr)
        
    return best_model, best_params, best_score
