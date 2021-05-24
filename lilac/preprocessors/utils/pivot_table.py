import pandas as pd


class PivotTable:
    """pivot tableを使って展開する."""

    def __init__(self,  key_col, emb_col_name, values_cols):
        self.key_col = key_col
        self.emb_col_name = emb_col_name
        self.values_cols = values_cols

    def transform(self, df):
        temp_list = []
        for values_col in self.values_cols:
            _df = pd.pivot_table(
                df, index=self.key_col, columns=self.emb_col_name, values=values_col).add_prefix(f"{values_col}_")
            temp_list.append(_df)

        return pd.concat(temp_list, axis=1)

    def return_flag(self):
        return f"{self.key_col}_{self.emb_col_name}_" + "_".join(self.values_cols)
