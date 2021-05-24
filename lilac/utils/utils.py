import numpy as np
import matplotlib.pyplot as plt
from slack import WebClient
import json


def remove_records_with_null(df, null_allowed_cols=None):
    """null_allowed_cols以外のカラムにnullがあるrecordを除去する."""
    df = df.copy()
    cols = list(df.columns)
    if null_allowed_cols is not None:
        for col in null_allowed_cols:
            if col in cols:
                cols.remove(col)
    return df.dropna(subset=cols)


def null_category_columns_remove(df):
    """欠損のあるカラムとobject型のカラムを削除する"""
    df = df.dropna(how='any', axis=1)
    object_cols = df.select_dtypes(include=[object]).columns
    df = df.drop(object_cols, axis=1)
    return df


def plot_feature_importance(df, path, max_n=20):
    # 特徴量数(説明変数の個数)
    n = len(df)
    df_plot = df.sort_values('importance')
    df_plot = df_plot.iloc[max(n-max_n, 0):]  # 上位max_n個だけ表示
    n_features = len(df_plot)
    f_importance_plot = df_plot['importance'].values  # 特徴量重要度の取得
    plt.barh(range(n_features), f_importance_plot, align='center')
    cols_plot = df_plot.index            # 特徴量の取得
    plt.yticks(np.arange(n_features), cols_plot)      # x軸,y軸の値の設定
    plt.xlabel('Importance')                  # x軸のタイトル
    plt.ylabel('Feature')
    plt.savefig(path, bbox_inches='tight')


class SlackApi:
    def __init__(self, token, app_name, channel):
        self.app_name = app_name
        self.channel = channel
        self.client = WebClient(token)

    def send_message(self, message):
        self.client.chat_postMessage(channel=self.channel,
                                     text=message, username=self.app_name)

    def upload_file(self, file_path):
        self.client.files_upload(
            channels=self.channel,
            file=file_path
        )


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
