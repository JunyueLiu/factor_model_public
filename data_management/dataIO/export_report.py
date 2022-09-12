import os.path

import pandas as pd

from data_management.dataIO.component_data import get_industry_info, get_industry_component, IndustryCategory, \
    get_index_component, IndexTicker, get_index_weights
from data_management.dataIO.fundamental_data import get_stock_info
from data_management.dataIO.trade_data import get_trade, TradeTable
from data_management.pandas_utils.cache import panel_df_join
from data_management.dataIO import market_data
from factor_zoo.industry import industry_category


def generate_selection_excel(selections: pd.Series, data_input_config: str, save_path=None):
    industry_info = get_industry_info(IndustryCategory.sw_l1, config_path=data_input_config)
    sw_l1 = get_industry_component(IndustryCategory.sw_l1, config_path=data_input_config)
    sec_info = get_stock_info(config_path=data_input_config)
    csi_300 = get_index_component(IndexTicker.csi300, config_path=data_input_config)
    zz_500 = get_index_component(IndexTicker.zz500, config_path=data_input_config)
    index_weights = get_index_weights(IndexTicker.csi300, config_path=data_input_config)
    # daily_data = market_data.get_bars(cols=('close',), adjust=False, eod_time_adjust=False, config_path=data_input_config)

    daily_basic = get_trade(TradeTable.daily_basic, config_path=data_input_config,
                            cols=['circ_mv', 'pe_ttm', 'ps_ttm', 'pb'])

    if selections.index.nlevels != 2:
        raise ValueError

    selections.index.names = ['date', 'code']

    sec_info.index.names = ['code']
    cat = industry_category(sw_l1).astype(str)
    industry_info.index.names = ['industry_code']
    industry_info.index = industry_info.index.astype(str)

    index_weights = index_weights['weight'].unstack().fillna(0)

    stock_df = selections.to_frame().join(sec_info['display_name'])
    stock_df = panel_df_join(stock_df, cat.to_frame())
    stock_df = stock_df[~stock_df[selections.name].isna()]
    stock_df = panel_df_join(stock_df, daily_basic)
    stock_df = stock_df[~stock_df[selections.name].isna()]

    index_weights = index_weights.fillna(0).reindex(index_weights.index
                                                    .union(stock_df.index.get_level_values(0).drop_duplicates()))\
        .fillna(method='ffill')
    index_weights = index_weights.stack()
    index_weights = index_weights[index_weights > 0]
    stock_df = panel_df_join(stock_df, index_weights.to_frame('csi300_weight'))
    stock_df = stock_df[~stock_df[selections.name].isna()]
    stock_df['csi300_weight'] = stock_df['csi300_weight'] / 100
    stock_df['weight_diff'] = stock_df[selections.name] - stock_df['csi300_weight']

    stock_df = stock_df.set_index('industry_code', append=True)
    stock_df = stock_df.join(industry_info['name']).reset_index(level=2)
    stock_df = stock_df.rename(columns={'name': 'sw1_name'})

    def _func2(x, u_dict):
        date = x.index.get_level_values(0)[0]
        comp = set(u_dict.get(date, []))
        if len(comp) > 0:
            return pd.DataFrame([c in comp for c in x.index.get_level_values(1)], index=x.index)
        else:
            return pd.DataFrame(False, index=x.index)


    stock_df['is_csi300_component'] = stock_df.groupby(level=0).apply(lambda x: _func2(x, csi_300))
    stock_df['is_zz500_component'] = stock_df.groupby(level=0).apply(lambda x: _func2(x, zz_500))
    stock_df['circ_mv'] = stock_df['circ_mv'] / 10_000
    stock_df = stock_df[['display_name', 'industry_code', 'sw1_name', 'circ_mv', 'pb', 'pe_ttm', 'ps_ttm',
                         'is_csi300_component', 'is_zz500_component', 'csi300_weight', 'weight_diff', selections.name]]
    stock_df = stock_df.sort_values(['date', selections.name], ascending=False)
    path = str(selections.name) + '.xlsx'
    if save_path is not None:
        path = os.path.join(save_path, path)

    writer = pd.ExcelWriter(path)
    stock_df.to_excel(writer, freeze_panes=(1, 1))

    # Auto-adjust columns' width
    for column in stock_df:
        column_width = max(stock_df[column].astype(str).map(len).max(), len(column))
        col_idx = stock_df.columns.get_loc(column)
        writer.sheets['Sheet1'].set_column(col_idx + 1, col_idx + 1, column_width + 2)
    writer.sheets['Sheet1'].set_column(0, 0, 22)
    writer.sheets['Sheet1'].set_column(1, 1, 11)

    # three_color_scale(writer.sheets['Sheet1'], 2, len(stock_df.columns), len(stock_df) + 1, len(stock_df.columns))

    writer.save()
    #
    # stock_df.to_excel(path, freeze_panes=(1, 1))


