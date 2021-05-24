from lilac.features.features_base import FeaturesBase
from xfeat import ArithmeticCombinations


class ArithmeticCombiFeatures(FeaturesBase):
    """数値特徴量の四則演算を指定したcolumns全組み合わせ計算する."""
    operator_str_dict = {"+": "sum", "-": "diff", "*": "prod", "/": "div"}

    def __init__(self,  target_cols, operator, features_dir=None):
        self.model = ArithmeticCombinations(
            input_cols=target_cols,
            drop_origin=True,
            operator=operator,
            output_suffix=f"_{self.operator_str_dict[operator]}",
            r=2
        )
        self.operator = operator
        self.target_cols = target_cols
        super().__init__(features_dir)

    def _transform(self, df):
        return self.model.transform(df[self.target_cols])

    def return_flag(self):
        """継承したらそのクラスの名前をflagにする."""
        if self.__class__.__name__ == "ArithmeticConbiFeatures":
            return f"{super().return_flag()}_{self.operator_str_dict[self.operator]}_" + "_".join(self.target_cols)
        else:
            return super().return_flag()
