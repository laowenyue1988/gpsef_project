# gpsef/utils/industry_utils.py
import pandas as pd

class IndustryNeutralizer:
    """行业中性化处理器"""
    def __init__(self, data_provider):
        self.data_provider = data_provider

    def neutralize(self, df, factor_cols, date_col='trade_date'):
        """
        对特征 DataFrame 进行行业中性化。
        df 必须包含 'ts_code' 和 date_col 列。
        factor_cols: 需要中性化的因子列名列表。
        返回：中性化后的 DataFrame（原列被修改）。
        """
        df = df.copy()
        df['industry'] = df['ts_code'].apply(self.data_provider.get_industry)
        for col in factor_cols:
            industry_mean = df.groupby([date_col, 'industry'])[col].transform('mean')
            df[col] = df[col] - industry_mean
        df.drop(columns=['industry'], inplace=True)
        return df