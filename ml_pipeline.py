"""
ml_pipeline.py — Tự Implement Random Forest, Gradient Boosting, SVM
=====================================================================
Phân loại stress từ HRV features mà không dùng sklearn/xgboost.

Cấu trúc file:
  1. DecisionTree        — CART, Gini Impurity
  2. RandomForest        — Bagging + Feature Subsampling
  3. GradientBoosting    — Simplified XGBoost (Newton boosting)
  4. SVM                 — Kernel RBF, dual coordinate ascent
  5. Evaluation helpers  — Accuracy, f1, Confusion Matrix
  6. Pipeline            — Train / Test / Report
"""

import os
import glob
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# ══════════════════════════════════════════════════════════════
#   1. DECISION TREE (CART)
# ══════════════════════════════════════════════════════════════

class _Node:
    """Nút cây: hoặc là nút trong (có ngưỡng chia) hoặc là lá (giá trị dự đoán)."""
    def __init__(self):
        self.feature_idx = None   # Chỉ số feature để chia
        self.threshold   = None   # Ngưỡng chia
        self.left        = None   # Cây con trái (X[f] <= threshold)
        self.right       = None   # Cây con phải (X[f] > threshold)
        self.value       = None   # Giá trị lá (nhãn dự đoán)


class DecisionTree:
    """
    Cây quyết định CART dùng Gini Impurity.

    Tham số:
        max_depth        : Độ sâu tối đa của cây
        min_samples_split: Số mẫu tối thiểu để tiếp tục chia
        n_features       : Số feature ngẫu nhiên xem xét ở mỗi nút (None = dùng tất cả)
    """
    def __init__(self, max_depth=10, min_samples_split=5, n_features=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.n_features        = n_features
        self.root              = None

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.root = self._grow(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    # ── Nội bộ ────────────────────────────────────────────
    def _gini(self, y):
        """
        Gini Impurity = 1 - Σ p_k²
        Bằng 0 khi tất cả nhãn giống nhau (tinh khiết nhất).
        """
        n = len(y)
        if n == 0:
            return 0.0
        counts = Counter(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def _best_split(self, X, y):
        """Tìm feature và ngưỡng tốt nhất bằng cách giảm Gini nhiều nhất."""
        n, n_feat = X.shape
        n_feat_try = self.n_features or n_feat
        feat_indices = random.sample(range(n_feat), min(n_feat_try, n_feat))

        best_gain = -1.0
        best_feat = best_thresh = None

        gini_parent = self._gini(y)

        for fi in feat_indices:
            col     = X[:, fi]
            sorted_vals = np.unique(col)
            # Chỉ thử tối đa 20 midpoints để tránh O(n) threshold loop
            midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
            thresholds = midpoints if len(midpoints) <= 20 else midpoints[np.linspace(0, len(midpoints)-1, 20, dtype=int)]

            for thresh in thresholds:
                left_mask  = col <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_l, y_r = y[left_mask], y[right_mask]
                gini_split = (len(y_l) / n) * self._gini(y_l) + \
                             (len(y_r) / n) * self._gini(y_r)

                gain = gini_parent - gini_split
                if gain > best_gain:
                    best_gain  = gain
                    best_feat  = fi
                    best_thresh = thresh

        return best_feat, best_thresh

    def _grow(self, X, y, depth):
        node = _Node()

        # Điều kiện dừng: đã đạt độ sâu max, hoặc quá ít mẫu, hoặc thuần nhãn
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(set(y)) == 1):
            node.value = Counter(y).most_common(1)[0][0]
            return node

        best_feat, best_thresh = self._best_split(X, y)

        if best_feat is None:    # Không tìm được split
            node.value = Counter(y).most_common(1)[0][0]
            return node

        node.feature_idx = best_feat
        node.threshold   = best_thresh

        mask = X[:, best_feat] <= best_thresh
        node.left  = self._grow(X[mask],  y[mask],  depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def _traverse(self, x, node):
        if node.value is not None:   # Nút lá
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# ══════════════════════════════════════════════════════════════
#   2. RANDOM FOREST
# ══════════════════════════════════════════════════════════════

class RandomForest:
    """
    Random Forest = Bagging nhiều Decision Tree độc lập.

    Mỗi cây được train trên một bootstrap sample (lấy ngẫu nhiên có hoàn lại)
    và chỉ xem xét √(n_features) feature ở mỗi lần chia → giảm tương quan giữa cây.
    Dự đoán cuối = majority vote của tất cả cây.

    Tham số:
        n_estimators : Số cây
        max_depth    : Độ sâu tối đa mỗi cây
        min_samples_split : Số mẫu tối thiểu để chia
    """
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.trees             = []

    def fit(self, X, y):
        n, n_feat = X.shape
        n_feat_per_tree = max(1, int(math.sqrt(n_feat)))   # √p features mỗi cây
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sample (lấy n mẫu có hoàn lại)
            indices = np.random.choice(n, size=n, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_feat_per_tree
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        # Lấy dự đoán từ tất cả cây, rồi vote
        all_preds = np.array([t.predict(X) for t in self.trees])   # (n_trees, n_samples)
        final = []
        for col in all_preds.T:
            final.append(Counter(col).most_common(1)[0][0])
        return np.array(final)


# ══════════════════════════════════════════════════════════════
#   3. GRADIENT BOOSTING (Simplified XGBoost)
# ══════════════════════════════════════════════════════════════

class GradientBoosting:
    """
    Gradient Boosting cho phân loại đa lớp.

    Ý tưởng cốt lõi:
      - Dùng softmax để tính xác suất mỗi lớp.
      - Mỗi vòng lặp: tính gradient (residual), fit một cây nhỏ vào gradient đó.
      - Cộng dần các cây lại → mỗi cây "sửa lỗi" của tập cây trước.

    Tham số:
        n_estimators : Số vòng boosting
        learning_rate: Bước học (shrinkage)
        max_depth    : Độ sâu cây yếu (thường 3-5)
    """
    def __init__(self, n_estimators=80, learning_rate=0.1, max_depth=4):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.trees_        = []   # trees_[m][k] = cây cho lớp k ở vòng m
        self.F0_           = None # Giá trị khởi đầu log-odds
        self.classes_      = None

    def _softmax(self, F):
        """F: (n, K) → probabilities (n, K)."""
        e = np.exp(F - F.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes_  = np.unique(y)
        K              = len(self.classes_)
        n              = len(y)
        # One-hot encode
        Y = np.zeros((n, K))
        for i, c in enumerate(self.classes_):
            Y[:, i] = (y == c).astype(float)

        # Khởi đầu: log(prior)
        prior      = Y.mean(axis=0) + 1e-9
        self.F0_   = np.log(prior)              # (K,)
        F          = np.tile(self.F0_, (n, 1))  # (n, K) — dự đoán hiện tại

        self.trees_ = []

        for _ in range(self.n_estimators):
            P    = self._softmax(F)          # Xác suất hiện tại (n, K)
            grad = Y - P                     # Pseudo-residuals = true - pred
            trees_this_round = []

            for k in range(K):
                r_k  = grad[:, k]            # Residual cho lớp k
                tree = DecisionTree(max_depth=self.max_depth, min_samples_split=3)
                tree.fit(X, r_k > 0)         # Phân loại residual dương/âm (đơn giản)
                leaf_pred = tree.predict(X).astype(float) * 2.0 - 1.0  # → {-1, +1}
                F[:, k] += self.learning_rate * leaf_pred
                trees_this_round.append(tree)

            self.trees_.append(trees_this_round)

    def predict(self, X):
        n = len(X)
        F = np.tile(self.F0_, (n, 1))

        for trees_round in self.trees_:
            for k, tree in enumerate(trees_round):
                leaf_pred = tree.predict(X).astype(float) * 2.0 - 1.0
                F[:, k] += self.learning_rate * leaf_pred

        proba = self._softmax(F)
        idx   = np.argmax(proba, axis=1)
        return self.classes_[idx]


# ══════════════════════════════════════════════════════════════
#   4. SVM (Kernel RBF — Dual Coordinate Ascent)
# ══════════════════════════════════════════════════════════════

class SVMBinary:
    """
    Binary SVM với RBF Kernel dùng Dual Coordinate Ascent (simplified SMO).

    Bài toán dual:
        max  Σα_i - ½ Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
        s.t. 0 ≤ α_i ≤ C, Σ α_i y_i = 0

    Kernel RBF: K(x, z) = exp(-γ ||x-z||²)

    Tham số:
        C     : Penalty (cân bằng margin và lỗi)
        gamma : Bandwidth kernel RBF
        tol   : Dừng khi Σ(cập nhật) < tol
        max_iter: Số vòng lặp tối đa
    """
    def __init__(self, C=1.0, gamma=0.5, tol=1e-3, max_iter=200):
        self.C        = C
        self.gamma    = gamma
        self.tol      = tol
        self.max_iter = max_iter

    def _rbf(self, x1, x2):
        diff = x1 - x2
        return math.exp(-self.gamma * float(np.dot(diff, diff)))

    def _kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                v = self._rbf(X[i], X[j])
                K[i, j] = v
                K[j, i] = v
        return K

    def fit(self, X, y):
        """y phải là {-1, +1}."""
        n     = len(X)
        alpha = np.zeros(n)
        b     = 0.0
        K     = self._kernel_matrix(X)

        for iteration in range(self.max_iter):
            num_changed = 0

            for i in range(n):
                # Tính error cho mẫu i
                E_i = float(np.dot(alpha * y, K[:, i])) + b - y[i]

                # Kiểm tra điều kiện KKT
                if (y[i] * E_i < -self.tol and alpha[i] < self.C) or \
                   (y[i] * E_i >  self.tol and alpha[i] > 0):

                    # Chọn j ≠ i ngẫu nhiên (single SMO pass)
                    j = random.choice([k for k in range(n) if k != i])
                    E_j = float(np.dot(alpha * y, K[:, j])) + b - y[j]

                    # Tính bounds L, H cho alpha[j]
                    if y[i] == y[j]:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    else:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])

                    if L >= H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Cập nhật alpha[j]
                    alpha_j_new = alpha[j] - y[j] * (E_i - E_j) / eta
                    alpha_j_new = np.clip(alpha_j_new, L, H)

                    if abs(alpha_j_new - alpha[j]) < 1e-5:
                        continue

                    alpha_i_new = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new)

                    # Cập nhật bias b
                    b1 = b - E_i - y[i] * (alpha_i_new - alpha[i]) * K[i, i] \
                                 - y[j] * (alpha_j_new - alpha[j]) * K[i, j]
                    b2 = b - E_j - y[i] * (alpha_i_new - alpha[i]) * K[i, j] \
                                 - y[j] * (alpha_j_new - alpha[j]) * K[j, j]

                    alpha[i] = alpha_i_new
                    alpha[j] = alpha_j_new

                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed += 1

            if num_changed == 0:
                break

        self.alpha_     = alpha
        self.b_         = b
        self.X_train_   = X
        self.y_train_   = y
        # Lưu support vectors để dự đoán nhanh hơn
        sv_mask         = alpha > 1e-5
        self.sv_alpha_  = alpha[sv_mask]
        self.sv_y_      = y[sv_mask]
        self.sv_X_      = X[sv_mask]

    def _decision(self, x):
        val = sum(self.sv_alpha_[k] * self.sv_y_[k] * self._rbf(self.sv_X_[k], x)
                  for k in range(len(self.sv_y_)))
        return val + self.b_

    def predict_one(self, x):
        return 1 if self._decision(x) >= 0 else -1


class SVMOneVsRest:
    """
    Multiclass SVM: One-vs-Rest — mỗi lớp có một binary SVM riêng.
    Dự đoán = lớp có decision value cao nhất.

    Tham số:
        C, gamma, tol, max_iter: chuyển thẳng vào SVMBinary
        max_train: giới hạn số mẫu train để giảm thời gian (RBF kernel O(n²))
    """
    def __init__(self, C=1.0, gamma=0.5, tol=1e-3, max_iter=200, max_train=500):
        self.svms      = {}
        self.C         = C
        self.gamma     = gamma
        self.tol       = tol
        self.max_iter  = max_iter
        self.max_train = max_train

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # Subsample nếu quá nhiều mẫu (giới hạn do O(n²) kernel)
        if len(X) > self.max_train:
            idx = np.random.choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]

        for cls in self.classes_:
            y_bin = np.where(y == cls, 1, -1).astype(float)
            svm   = SVMBinary(C=self.C, gamma=self.gamma,
                              tol=self.tol, max_iter=self.max_iter)
            svm.fit(X, y_bin)
            self.svms[cls] = svm

    def predict(self, X):
        # Trả về lớp có decision value cao nhất
        scores = {cls: [] for cls in self.classes_}
        for x in X:
            for cls in self.classes_:
                scores[cls].append(self.svms[cls]._decision(x))

        preds = []
        for i in range(len(X)):
            best_cls = max(self.classes_, key=lambda c: scores[c][i])
            preds.append(best_cls)
        return np.array(preds)


# ══════════════════════════════════════════════════════════════
#   5. EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════

def accuracy_score_custom(y_true, y_pred):
    return float(np.sum(y_true == y_pred)) / len(y_true)

def confusion_matrix_custom(y_true, y_pred):
    labels = sorted(set(list(y_true) + list(y_pred)))
    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_idx[t], label_idx[p]] += 1
    return cm, labels

def f1_score_custom(y_true, y_pred, average="macro"):
    labels = sorted(set(list(y_true) + list(y_pred)))
    f1s = []
    for lbl in labels:
        tp = np.sum((y_pred == lbl) & (y_true == lbl))
        fp = np.sum((y_pred == lbl) & (y_true != lbl))
        fn = np.sum((y_pred != lbl) & (y_true == lbl))
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(3, len(labels))))
    img = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(img, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#   6. PIPELINE
# ══════════════════════════════════════════════════════════════

DATA_DIR     = "./data/research_dataset"
REPORT_DIR   = "./results"
FEATURE_PATH = f"{DATA_DIR}/ecg_features.csv"

# ─── Feature columns dùng cho training ───────────────────────
FEATURE_COLS = ["HR", "SDNN", "RMSSD", "pNN50", "LF_HF_Ratio"]

os.makedirs(REPORT_DIR, exist_ok=True)


def normalize(X_train, X_test):
    """Min-Max normalization theo train set."""
    mn = X_train.min(axis=0)
    mx = X_train.max(axis=0)
    rng = mx - mn + 1e-9
    return (X_train - mn) / rng, (X_test - mn) / rng


def subsample_df(df, frac=1.0, seed=42):
    """Lấy frac% dòng ngẫu nhiên."""
    if frac >= 1.0: return df
    return df.sample(frac=frac, random_state=seed).reset_index(drop=True)


def run_experiment(strategy_name, train_df, test_df, results, train_frac=1.0, test_frac=1.0):
    """Chạy cả 2 tasks × 3 models cho một chiến lược."""
    train_df = subsample_df(train_df, frac=train_frac)
    test_df  = subsample_df(test_df,  frac=test_frac)
    print(f"  [Data] Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    for task in ["binary", "multiclass"]:
        # Chuẩn bị nhãn
        if task == "binary":
            # 0=Low, 1=Medium/High/VeryHigh
            y_train = (train_df["stress_label"] > 0).astype(int).values
            y_test  = (test_df["stress_label"] > 0).astype(int).values
        else:
            y_train = train_df["stress_label"].values
            y_test  = test_df["stress_label"].values

        X_train = train_df[FEATURE_COLS].values.astype(float)
        X_test  = test_df[FEATURE_COLS].values.astype(float)
        X_train_n, X_test_n = normalize(X_train, X_test)

        models = {
            "RandomForest":      RandomForest(n_estimators=30, max_depth=6),
            "GradientBoosting":  GradientBoosting(n_estimators=30, learning_rate=0.1, max_depth=3),
            "SVM_RBF":           SVMOneVsRest(C=1.0, gamma=0.5, max_iter=100, max_train=500),
        }

        for mname, model in models.items():
            print(f"  [{strategy_name}] [{task}] [{mname}] — training...")
            try:
                model.fit(X_train_n, y_train)
                y_pred = model.predict(X_test_n)

                acc = accuracy_score_custom(y_test, y_pred)
                f1  = f1_score_custom(y_test, y_pred, average="macro")
                cm, lbls = confusion_matrix_custom(y_test, y_pred)

                print(f"    Accuracy : {acc:.4f}, F1: {f1:.4f}")

                # Lưu ảnh confusion matrix
                safe = strategy_name.replace(" ", "_").replace("→", "to").replace("+", "plus")
                cm_path = os.path.join(REPORT_DIR, f"cm_{safe}_{task}_{mname}.png")
                plot_confusion_matrix(cm, lbls,
                                      f"{strategy_name}\n{task} | {mname}\nAcc={acc:.2f} F1={f1:.2f}",
                                      cm_path)

                results.append({
                    "Strategy":    strategy_name,
                    "Task":        task,
                    "Model":       mname,
                    "Accuracy":    round(acc, 4),
                    "F1_Score":    round(f1, 4),
                })
            except Exception as e:
                print(f"    ERROR: {e}")


def plot_summary(df_results):
    """Vẽ bar chart so sánh accuracy và F1 theo model × task."""
    for metric in ["Accuracy", "F1_Score"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, task in zip(axes, ["binary", "multiclass"]):
            sub = df_results[df_results["Task"] == task]
            if sub.empty: continue
            models    = sub["Model"].unique()
            strategies = sub["Strategy"].unique()
            x = np.arange(len(models))
            width = 0.35

            for i, strat in enumerate(strategies):
                vals = [sub[(sub["Model"] == m) & (sub["Strategy"] == strat)][metric].mean()
                        for m in models]
                ax.bar(x + i * width, vals, width, label=strat)

            ax.set_title(f"{metric} — {task}")
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(models, rotation=10)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel(metric)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.4)

        plt.suptitle(f"Model Comparison — {metric}", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, f"summary_{metric}.png"), dpi=100)
        plt.close(fig)


def main():
    print("=" * 60)
    print("  ML Pipeline — Stress Classification (Research Dataset)")
    print("=" * 60)

    if not os.path.exists(FEATURE_PATH):
        print(f"Error: Features file {FEATURE_PATH} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(FEATURE_PATH)
    print(f"Loaded {len(df)} windowed samples.")
    results = []

    # Strategy A: Train on Clean, Test on Noisy
    print("\nStrategy A: Clean → Noisy (Robustness)")
    train_A = df[df['is_noisy'] == False]
    test_A  = df[df['is_noisy'] == True]
    run_experiment("A: Clean→Noisy", train_A, test_A, results, train_frac=1.0, test_frac=1.0)

    # Strategy B: Train Office/Physical, Test HighPressure
    print("\nStrategy B: Normal/Active → HighPressure (Generalization)")
    train_B = df[df['scenario'].isin(['office_worker', 'physical_worker'])]
    test_B  = df[df['scenario'] == 'high_pressure']
    run_experiment("B: Gen→Stress", train_B, test_B, results, train_frac=1.0, test_frac=1.0)

    # Save results
    df_res = pd.DataFrame(results)
    report_path = os.path.join(REPORT_DIR, "ml_results_summary.csv")
    df_res.to_csv(report_path, index=False)

    plot_summary(df_res)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if not df_res.empty:
        print(df_res.groupby(["Strategy", "Task"])["Accuracy"].mean())
    print(f"\n✅ Results: {REPORT_DIR}/")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()
