from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from collections import Counter, defaultdict
import random


class FoldsGeneratorBase:
    def __init__(self, fold_num, seed):
        self.fold_num = fold_num
        self.seed = seed

    def run(self, data):
        raise Exception("Implement please.")

    def return_flag(self):
        return f"{self.fold_num}_{self.seed}"


class FoldsGenerator(FoldsGeneratorBase):
    def run(self, data):
        kf = KFold(n_splits=self.fold_num,
                   random_state=self.seed, shuffle=True)
        return kf.split(data)

    def return_flag(self):
        return f"kfold_{super().return_flag()}"


class StratifiedFoldsGenerator(FoldsGeneratorBase):
    def __init__(self, fold_num, seed, target_col):
        super().__init__(fold_num, seed)
        self.target_col = target_col

    def run(self, data):
        kf = StratifiedKFold(n_splits=self.fold_num,
                             random_state=self.seed, shuffle=True)
        return kf.split(data, data[self.target_col])

    def return_flag(self):
        return f"stratified_{self.target_col}_{super().return_flag()}"


class GroupKFoldsGenerator(FoldsGeneratorBase):
    def __init__(self, fold_num, seed, key_col):
        super().__init__(fold_num, seed)
        self.key_col = key_col

    def run(self, data):
        kf = self._MyGroupKFold(n_splits=self.fold_num,
                                random_state=self.seed, shuffle=True)
        return kf.split(data, group=data[self.key_col])

    def return_flag(self):
        return f"group_{self.key_col}_{super().return_flag()}"

    class _MyGroupKFold:
        """
        GroupKFold with random shuffle with a sklearn-like structure
        """

        def __init__(self, n_splits, shuffle, random_state):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def get_n_splits(self, X=None, y=None, group=None):
            return self.n_splits

        def split(self, X=None, y=None, group=None):
            kf = KFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            unique_ids = group.unique()
            for tr_group_idx, va_group_idx in kf.split(unique_ids):
                # split group
                tr_group, va_group = unique_ids[tr_group_idx], unique_ids[va_group_idx]
                train_idx = np.where(group.isin(tr_group))[0]
                val_idx = np.where(group.isin(va_group))[0]
                yield train_idx, val_idx


class StratifiedGroupKFold(FoldsGeneratorBase):
    def __init__(self, fold_num, seed, target_col, key_col):
        super().__init__(fold_num, seed)
        self.key_col = key_col
        self.target_col = target_col

    def run(self, data):
        y = data[self.target_col]
        X = data.drop(self.target_col, axis=1)
        return self._stratified_group_k_fold(X=X, y=y, groups=X[self.key_col], k=self.fold_num, seed=self.seed)

    def return_flag(self):
        return f"stratified_group_{self.key_col}_{self.target_col}_{super().return_flag()}"

    @staticmethod
    def _stratified_group_k_fold(X, y, groups, k, seed=None):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(k):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(k):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(
                groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(
                groups) if g in test_groups]

            yield train_indices, test_indices
