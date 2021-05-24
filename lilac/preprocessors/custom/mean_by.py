import pandas as pd


def get_group_mean(data, key_col, target_cols):
    hoge = pd.DataFrame()
    hoge[[f"{e}MeanBy{key_col}" for e in target_cols]] = data.groupby(key_col).mean()[
        target_cols]
    return hoge


def add_mean_by_cols(train, group_mean, key_col):
    added_train = pd.merge(train, group_mean, on=key_col, how="left")
    return added_train


def add_diff_ratio(data, key_col, target_cols):
    data = data.copy()
    for target_col in target_cols:
        data[f"{target_col}DiffMeanBy{key_col}"] = data[target_col].values - \
            data[f"{target_col}MeanBy{key_col}"].values
        data[f"{target_col}RatioMeanBy{key_col}"] = data[target_col].values / \
            data[f"{target_col}MeanBy{key_col}"]
        data[f"{target_col}DiffRatioMeanBy{key_col}"] = data[f"{target_col}DiffMeanBy{key_col}"].values / \
            data[f"{target_col}MeanBy{key_col}"]
    return data
