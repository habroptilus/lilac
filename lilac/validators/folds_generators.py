from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
import numpy as np
from collections import Counter, defaultdict
import random
import sys
import pandas as pd


class FoldsGeneratorBase:
    def __init__(self, fold_num):
        self.fold_num = fold_num

    def run(self, data):
        raise Exception("Implement please.")

    def return_flag(self):
        return self.fold_num


class FoldsGenerator(FoldsGeneratorBase):
    def __init__(self, fold_num, seed):
        self.seed = seed
        super().__init__(fold_num)

    def run(self, data):
        kf = KFold(n_splits=self.fold_num,
                   random_state=self.seed, shuffle=True)
        return kf.split(data)

    def return_flag(self):
        return f"kfold_{self.seed}_{super().return_flag()}"


class StratifiedFoldsGenerator(FoldsGeneratorBase):
    def __init__(self, fold_num, seed, target_col):
        super().__init__(fold_num)
        self.seed = seed
        self.target_col = target_col

    def run(self, data):
        kf = StratifiedKFold(n_splits=self.fold_num,
                             random_state=self.seed, shuffle=True)
        return kf.split(data, data[self.target_col])

    def return_flag(self):
        return f"stratified_{self.target_col}_{self.seed}_{super().return_flag()}"


class GroupKFoldsGenerator(FoldsGeneratorBase):
    """sklearnのGroupKfoldはseedを指定できないので自前実装したもの."""

    def __init__(self, fold_num, seed, key_col):
        super().__init__(fold_num)
        self.key_col = key_col
        self.seed = seed

    def run(self, data):
        kf = self._MyGroupKFold(n_splits=self.fold_num,
                                random_state=self.seed, shuffle=True)
        return kf.split(data, group=data[self.key_col])

    def return_flag(self):
        return f"group_{self.key_col}_{self.seed}_{super().return_flag()}"

    class _MyGroupKFold:
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
        super().__init__(fold_num)
        self.key_col = key_col
        self.target_col = target_col
        self.seed = seed

    def run(self, data):
        y = data[self.target_col].astype(int)
        X = data.drop(self.target_col, axis=1)
        return self._stratified_group_k_fold(X=X, y=y, groups=X[self.key_col], k=self.fold_num, seed=self.seed)

    def return_flag(self):
        return f"stratified_group_{self.key_col}_{self.target_col}_{self.seed}_{super().return_flag()}"

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


class TimeSeriesKfold(FoldsGeneratorBase):
    """ts_columnに沿った時系列でCVをきる.(ex, train:0~10, valid: 11)
    [waring]: 時系列的に一番古いfoldのデータがvalidに入ることがないため、古いデータに関してout of foldのpredictionが手に入らないという問題がある.
    一応evaluatorはwarning出して動く仕様にしたが、oof_predがないデータがあるためアンサンブルはできない.
    """

    def __init__(self, fold_num, ts_column, clipping=False):
        self.splitter = self._MovingWindowKFold(
            ts_column=ts_column, clipping=clipping, n_splits=fold_num)
        super().__init__(fold_num)
        self.ts_column = ts_column
        self.clipping = clipping

    def run(self, data):
        return self.splitter.split(data)

    def return_flag(self):
        return f"time_series_{self.ts_column}_{self.clipping}_{super().return_flag()}"

    class _MovingWindowKFold(TimeSeriesSplit):
        """時系列情報が含まれるカラムでソートした iloc を返す KFold"""

        def __init__(self, ts_column, clipping=False, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 時系列データのカラムの名前
            self.ts_column = ts_column
            # 得られる添字のリストの長さを過去最小の Fold に揃えるフラグ
            self.clipping = clipping

        def split(self, X, *args, **kwargs):
            # 渡されるデータは DataFrame を仮定する
            assert isinstance(X, pd.DataFrame)

            # clipping が有効なときの長さの初期値
            train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize

            # 時系列のカラムを取り出す
            ts = X[self.ts_column]
            # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
            ts_df = ts.reset_index()
            # 時系列でソートする
            sorted_ts_df = ts_df.sort_values(by=self.ts_column)
            # スーパークラスのメソッドで添字を計算する
            for train_index, test_index in super().split(sorted_ts_df, *args, **kwargs):
                # 添字を元々の DataFrame の iloc として使える値に変換する
                train_iloc_index = sorted_ts_df.iloc[train_index].index
                test_iloc_index = sorted_ts_df.iloc[test_index].index

                if self.clipping:
                    # TimeSeriesSplit.split() で返される Fold の大きさが徐々に大きくなることを仮定している
                    train_fold_min_len = min(
                        train_fold_min_len, len(train_iloc_index))
                    test_fold_min_len = min(
                        test_fold_min_len, len(test_iloc_index))

                yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])


class TimeSeriesSimpleKfold(FoldsGeneratorBase):
    """時系列の順序は気にせず時間的な近さごとにfoldを作る."""

    def __init__(self, fold_num, ts_column):
        self.splitter = self._InnerSplitter(
            ts_column=ts_column, n_splits=fold_num)
        super().__init__(fold_num)
        self.ts_column = ts_column

    def run(self, data):
        return self.splitter.split(data)

    class _InnerSplitter(KFold):
        def __init__(self, ts_column, *args, **kwargs):
            super().__init__(shuffle=False, *args, **kwargs)
            self.ts_column = ts_column

        def split(self, X, *args, **kwargs):

            assert isinstance(X, pd.DataFrame)

            # 時系列のカラムを取り出す
            ts = X[self.ts_column]
            # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
            ts_df = ts.reset_index()
            # 時系列でソートする
            sorted_ts_df = ts_df.sort_values(by=self.ts_column)
            # スーパークラスのメソッドで添字を計算する
            for train_index, test_index in super().split(sorted_ts_df, *args, **kwargs):
                train_iloc_index = sorted_ts_df.iloc[train_index].index
                test_iloc_index = sorted_ts_df.iloc[test_index].index
                yield list(train_iloc_index), list(test_iloc_index)
