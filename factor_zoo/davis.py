from typing import Optional, Union, List, Tuple

import pandas as pd


def tf_davis_double_kill(income_statement: pd.DataFrame, trading_dates,
                         *,
                         fin_forecast: Optional[pd.DataFrame] = None,
                         quick_fin: Optional[pd.DataFrame] = None,
                         income_statement_net_profit_name='np_parent_company_owners',
                         freq: Union[str, Union[List[int], Tuple[int, int]]] = 'D', lookback: int = 60,
                         start=None,
                         end=None,
                         return_with_yoy_growth_rate=False
                         ):
    """
        The implementation is not shown in open source version
        """


def tf_davis_double_kill_with_low_value(income_statement: pd.DataFrame,
                                        daily_basic: pd.DataFrame,
                                        trading_dates,
                                        fin_forecast: Optional[pd.DataFrame] = None,
                                        quick_fin: Optional[pd.DataFrame] = None,
                                        income_statement_net_profit_name='np_parent_company_owners',
                                        freq: Union[str, Union[List[int], Tuple[int, int]]] = 'D', lookback: int = 60,
                                        start=None,
                                        end=None,
                                        pe_threshold=50
                                        ):
    """
        The implementation is not shown in open source version
        """


def tf_davis_double_kill_with_low_value_continuable_growth(income_statement: pd.DataFrame,
                                                           daily_basic: pd.DataFrame,
                                                           trading_dates,
                                                           fin_forecast: Optional[pd.DataFrame] = None,
                                                           quick_fin: Optional[pd.DataFrame] = None,
                                                           income_statement_net_profit_name='np_parent_company_owners',
                                                           freq: Union[str, Union[List[int], Tuple[int, int]]] = 'D',
                                                           lookback: int = 60,
                                                           start=None,
                                                           end=None,
                                                           pe_threshold: float = 50,
                                                           yoy_growth_lower: float = 0.2,
                                                           yoy_growth_upper: float = 1,
                                                           num_stocks: int = 25
                                                           ):
    """
        The implementation is not shown in open source version
        """
