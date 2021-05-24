import pandas as pd


class FeaturesBase:
    """特徴量生成の基底クラス.継承して_fitや_transformを実装する.

    (tips: 依存する特徴量クラスの出力カラムをまとめて指定する方法)
    target_colsという名前で適用するカラム名を指定するようなクラスの場合、
    依存する特徴量クラスの出力カラムを数字で指定することもできる
    dependenciesのリストに対応したインデックス番号のリストをtarget_colsに指定すると、runするときにresolveしてくれるようにした.
    Kmeans,PCAなどで便利かも.

    (tips: testでは特徴量を計算しない場合)
    fit_transformだけ実装し、_transformをNoneを返す実装にするとできる.
    modelの入力にはできないが、依存クラスとしては使用できる.
    """
    dependencies = []

    def __init__(self, features_dir=None):
        """features_dirを指定しない場合、save,load機能がオフになる."""
        self.features_dir = features_dir
        self._dependencies_cols = []

    def run(self, train_data, test_data):
        train = train_data.copy()
        test = test_data.copy()
        res_train = None
        res_test = None
        if self.features_dir:
            train_path = self.features_dir / self.return_flag() / "train.ftr"
            test_path = self.features_dir / self.return_flag() / "test.ftr"
            if train_path.exists():
                # あるなら読み込む
                print(f"Loading {self.return_flag()} (train)...")
                res_train = pd.read_feather(train_path)
            if test_path.exists():
                print(f"Loading {self.return_flag()} (test)...")
                res_test = pd.read_feather(test_path)

        if res_train is not None:
            return res_train, res_test

        # 依存クラスがあるなら実行して追加
        for settings in self.dependencies:
            feature_gen = settings["features"](
                features_dir=self.features_dir, **settings["params"])
            _train, _test = feature_gen.run(train_data, test_data)
            self._dependencies_cols.append(list(_train.columns))
            train = pd.concat([train, _train], axis=1)
            if _test is not None:
                test = pd.concat([test, _test], axis=1)

        # 本体実行
        print(f"Generating {self.return_flag()}...")
        res_train = self.fit_transform(train)
        res_test = self.transform(test)
        if self.features_dir:
            train_path = self.features_dir / self.return_flag() / "train.ftr"
            train_path.parent.mkdir(exist_ok=True)
            res_train.to_feather(train_path)

            if res_test is not None:
                test_path = self.features_dir / self.return_flag() / "test.ftr"
                res_test.to_feather(test_path)
        return res_train, res_test

    def fit(self, df):
        df = df.copy()
        self._resolve_target_cols()
        self._fit(df)
        return self

    def _fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        return self._transform(df)

    def _transform(self, df):
        raise Exception("Not implemented error.")

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def return_flag(self):
        return self.__class__.__name__

    def _resolve_target_cols(self):
        if hasattr(self, 'target_cols') and (type(self.target_cols[0]) == int):
            # flatten
            self.target_cols = sum([self._dependencies_cols[i]
                                    for i in self.target_cols], [])


class TrainOnlyFeatureBase(FeaturesBase):
    """trainしか作らない特徴量はこちらを継承する."""

    def fit_transform(self, df):
        df = df.copy()
        return self._fit_transform(df)

    def _fit_transform(self):
        raise Exception("Implement _fit_transform method in TrainOnlyFeature.")

    def _transform(self, df):
        return None
