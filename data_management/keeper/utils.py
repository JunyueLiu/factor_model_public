from typing import Union, Optional, Dict, List

import pandas as pd

from factor_zoo.industry import industry_category


def add_stock_info(factor_data: Union[pd.DataFrame, pd.Series],
                   stock_info: pd.DataFrame) -> pd.DataFrame:
    """

    :param factor_data:
    :param stock_info:
    :return:
    """
    if isinstance(factor_data, pd.Series):
        factor_data = factor_data.to_frame(factor_data.name)
    stock_info.index.names = [factor_data.index.names[1]]
    factor_data = factor_data.join(stock_info[['display_name', 'start_date']])
    factor_data['listed_days'] = factor_data.index.get_level_values(0) - factor_data['start_date']
    factor_data['listed_days'] = factor_data['listed_days'].dt.days
    factor_data = factor_data.rename(columns={'start_date': 'listed_date'})
    return factor_data


def add_industry_info(factor_data: Union[pd.DataFrame, pd.Series],
                      industry_dict: Dict,
                      industry_info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """

    :param factor_data:
    :param industry_dict:
    :param industry_info:
    :return:
    """
    if isinstance(factor_data, pd.Series):
        factor_data = factor_data.to_frame(factor_data.name)

    cat = industry_category(industry_dict).astype(str)
    factor_data = factor_data.join(cat)
    if industry_info is not None:
        industry_info.index.names = ['industry_code']
        factor_data = factor_data.set_index('industry_code', append=True)
        industry_info.index = industry_info.index.astype(str)
        factor_data = factor_data.join(industry_info['name']).reset_index(level=2)
        factor_data = factor_data.rename(columns={'name': 'industry_name'})
    return factor_data


def add_daily_basic(factor_data: Union[pd.DataFrame, pd.Series],
                    daily_basic: pd.DataFrame) -> pd.DataFrame:
    """

    :param factor_data:
    :param daily_basic:
    :return:
    """
    if isinstance(factor_data, pd.Series):
        factor_data = factor_data.to_frame(factor_data.name)
    factor_data = factor_data. \
        join(daily_basic[['close', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
                          'dv_ttm', 'total_share', 'float_share', 'free_share',
                          'total_mv', 'circ_mv'
                          ]])
    return factor_data


def add_component(factor_data: Union[pd.DataFrame, pd.Series],
                  component: Dict[str, Dict[pd.Timestamp, List[str]]]) -> pd.DataFrame:
    """

    :param factor_data:
    :param component:
    :return:
    """
    def _func2(x, u_dict):
        date = x.index.get_level_values(0)[0]
        comp = set(u_dict.get(date, []))
        if len(comp) > 0:
            return pd.Series([c in comp for c in x.index.get_level_values(1)], index=x.index)
        else:
            return pd.Series(False, index=x.index)

    if isinstance(factor_data, pd.Series):
        factor_data = factor_data.to_frame(factor_data.name)

    for universe_name, u_dict in component.items():
        factor_data['is_{}_component'.format(universe_name)] = \
            factor_data.groupby(level=0)[factor_data.columns[0]].apply(lambda x: _func2(x, u_dict))

    return factor_data
