import datetime
import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from itertools import combinations
from xlsxwriter.utility import xl_rowcol_to_cell

from factor_testing.performance import quantize_factor, quantile_backtesting, factor_backtesting, cal_ic, \
    information_analysis, factor_ols_regression, factor_quantile_regression, backtesting_metric, aggregate_returns, \
    factors_correlation, Newey_West_t_statistics
from factor_testing.utils import load_single_category_factors, calculate_forward_returns, \
    get_forward_returns_columns
from factor_testing.xlsx_utils import two_color_heatmap, three_color_scale, t_statistics_heatmap, corr_matrix_heatmap, \
    insert_chart, ChartType, insert_NewlyWest_chart, insert_chart2, insert_chart3
from factor_zoo.factor_transform import cap_industry_neutralize
from factor_zoo.industry import industry_category
from factor_zoo.utils import load_pickle, market_filter_in, combine_market_with_fundamental


def single_factor_report(merge_data: pd.DataFrame, factor_name, quantiles=None, bins=5, demeaned=True,
                         result_save_path='../single_factor_report', daily_market=None,
                         daily_basic: Optional[pd.DataFrame] = None,
                         stock_info: Optional[pd.DataFrame] = None,
                         industry_dict: Optional[Dict] = None,
                         industry_info: Optional[pd.DataFrame] = None,
                         top_portfolio=None,
                         benchmark: Optional[pd.DataFrame] = None,
                         component: Optional[Dict[str, Dict[pd.Timestamp, List[str]]]] = None,
                         sparkline: bool = False,
                         ):
    """

    :param merge_data:
    :param factor_name:
    :param quantiles:
    :param bins:
    :param demeaned:
    :param result_save_path:
    :param daily_market:
    :param daily_basic:
    :param stock_info:
    :param industry_dict:
    :param industry_info:
    :param top_portfolio:
    :param benchmark:
    :param component:
    :return:
    """
    if benchmark is not None:
        if demeaned:
            raise ValueError('Cannot set benchmark and demeaned at the same time.')

        for c in merge_data.columns:
            if 'excess' in c:
                raise ValueError('future return is Excess return, Cannot work with benchmark at the same time.')

    factor_data = merge_data.copy()
    dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(result_save_path, '{}_{}_test.xlsx'.format(factor_name, dt))
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')

    # factor statistics
    summary = factor_data.groupby(level=0)[factor_name].agg(['min', 'max', 'mean', 'std', 'count', 'skew'])
    summary['cover_rate'] = factor_data[factor_name].groupby(level=0).aggregate(lambda x: 1 - x.isnull().sum() / len(x))
    summary['positive_rate'] = \
        factor_data[factor_name].groupby(level=0).aggregate(lambda x: (x > 0).sum() / len(x))
    if component:
        def _func(x, u_dict):
            date = x.index.get_level_values(0)[0]
            comp = set(u_dict.get(date, []))
            if len(comp) > 0:
                return len(set(x.dropna().index.get_level_values(1)).intersection(comp)) / len(comp)
            else:
                return np.nan

        for universe_name, u_dict in component.items():
            summary['{}_cover_rate'.format(universe_name)] = \
                factor_data[factor_name].groupby(level=0).agg(lambda x: _func(x, u_dict))

    summary.to_excel(writer, sheet_name='factor summary statistics', )
    # dropna
    factor_data = factor_data.dropna(axis=0)
    # find the rebalance freq
    rb_date = factor_data.index.get_level_values(0).drop_duplicates().sort_values()
    rb_days = (rb_date[1:] - rb_date[:-1]).median().days
    if rb_days <= 1:
        period = 250
    elif rb_days <= 7:
        period = 52
    elif rb_days <= 17:
        period = 26
    elif rb_days <= 35:
        period = 12
    elif rb_days <= 100:
        period = 4
    elif rb_days <= 260:
        period = 2
    else:
        period = 1

    # factor IC, OLS, t-test, quantile regression
    # todo ICIR
    cross_section_ic = cal_ic(factor_data, factor_name=factor_name)
    information = information_analysis(cross_section_ic)
    ols = factor_ols_regression(factor_data, factor_name)
    report = pd.concat([information, ols])
    for q in [0.5]:
        quantile_reg = factor_quantile_regression(factor_data, factor_name, q)
        report = report.append(quantile_reg)

    report.to_excel(writer, sheet_name='IC OLS QuantileReg')

    # quantile
    cum_ret, mean_ret, count_stat, std_error_ret = quantile_backtesting(factor_data, factor_name,
                                                                        bins=bins, demeaned=demeaned)
    factor_cum = factor_backtesting(factor_data, factor_name, demeaned)
    newey_west = Newey_West_t_statistics(mean_ret)

    cum_ic = cross_section_ic.cumsum()
    cum_ic.to_excel(writer, sheet_name='Cumulative IC')
    factor_data['group'] = quantize_factor(factor_data, factor_name, quantiles=quantiles, bins=bins)
    group_ic = cal_ic(factor_data, factor_name=factor_name, by_group=True)
    group_ic = group_ic.unstack(-1)
    group_ic = group_ic.cumsum()
    group_ic.to_excel(writer, sheet_name='Cumulative IC', startrow=0, startcol=3)

    # backtesting statistics

    backtesting_res = backtesting_metric(mean_ret, cum_ret, period)
    quarter_ret = aggregate_returns(mean_ret, 'quarter')  # type: pd.DataFrame
    year_ret = aggregate_returns(mean_ret, 'year')  # type: pd.DataFrame

    # turnover
    to_replace = [i for i in range(2, bins)]
    factor_data['long_short'] = factor_data['group'].replace(to_replace, 0).replace(1, -1).replace(bins, 1)
    factor_data['long'] = factor_data['long_short'].replace(-1, 0)
    group_change = \
        factor_data.groupby(level=1)[['long_short', 'long']].apply(lambda x: (x.shift(1) - x).abs())
    holding_num = factor_data.groupby(level=0)[['long_short', 'long']].agg(lambda x: (x != 0).sum()).shift(1)
    turnover = group_change.groupby(level=0).sum()
    turnover_ratio = turnover / holding_num
    turnover_ratio_heatmap = turnover_ratio.groupby(pd.Grouper(freq='M')).sum()
    turnover_ratio_heatmap['year'] = turnover_ratio_heatmap.index.year
    turnover_ratio_heatmap['month'] = turnover_ratio_heatmap.index.month
    turnover_ratio_heatmap = turnover_ratio_heatmap.set_index(['year', 'month']).unstack()  # type: pd.DataFrame

    factor_data = factor_data.drop(columns=['long', 'long_short'])

    # return heatmap
    heatmap = mean_ret.copy().iloc[:, [-2, -1]].groupby(pd.Grouper(freq='M')).sum()
    heatmap['year'] = heatmap.index.year
    heatmap['month'] = heatmap.index.month
    heatmap = heatmap.set_index(['year', 'month']).unstack()

    backtesting_res.to_excel(writer, sheet_name='Quantile backtesting', startrow=24, startcol=len(cum_ret.columns) + 2)

    factor_cum.to_excel(writer, sheet_name='factor backtesting', )
    cum_ret.to_excel(writer, sheet_name='Quantile backtesting')
    mean_ret.to_excel(writer, sheet_name='Sorted Portfolios Analysis', )
    quarter_ret.to_excel(writer, sheet_name='Sorted Portfolios Analysis', startrow=0,
                         startcol=2 * len(mean_ret.columns) + 2)
    year_ret.to_excel(writer, sheet_name='Sorted Portfolios Analysis', startrow=0,
                      startcol=3 * len(mean_ret.columns) + 3)
    newey_west.to_excel(writer, sheet_name='Sorted Portfolios Analysis', startrow=0, startcol=len(mean_ret.columns) + 1)
    # backtesting_statistics
    heatmap.to_excel(writer, sheet_name='Monthly Ret Heatmap')
    count_stat.to_excel(writer, sheet_name='Num Stock each Portfolios')
    turnover_ratio.to_excel(writer, sheet_name='Turnover Analysis')
    turnover_ratio_heatmap.to_excel(writer, sheet_name='Turnover Analysis', startrow=0, startcol=4)
    # Access the XlsxWriter workbook and worksheet objects from the dataframe.
    workbook = writer.book

    # ----------------------------------------------
    # plot
    # -----------------------------------------------
    # todo summary page

    # --------------------- Cumulative IC page --------------------------
    worksheet = writer.sheets['Cumulative IC']
    location = xl_rowcol_to_cell(1, len(cum_ic.columns) + len(group_ic.columns) + 3)
    insert_chart(workbook, worksheet, ChartType.Line, 'Cumulative IC', 1, 1, len(cum_ic), 1,
                 'Cumulative IC', location, 'time', 'Cumulative IC',
                 'Cumulative IC Results',
                 2, 1.5)
    num_cols = len(group_ic.columns)
    cols = [_ + 4 for _ in range(num_cols)]
    location = xl_rowcol_to_cell(24, len(cum_ic.columns) + len(group_ic.columns) + 3)
    insert_chart(workbook, worksheet, ChartType.Line, 'Cumulative IC',
                 [3] * num_cols, cols,
                 [3 + len(group_ic)] * num_cols, cols,
                 ['Group {}'.format(c[1]) for c in group_ic.columns],
                 location, 'time', 'Cumulative IC', 'Group Cumulative IC Results',
                 2, 1.5
                 )
    # --------------------- factor backtesting page --------------------------
    worksheet = writer.sheets['factor backtesting']
    insert_chart(workbook, worksheet, ChartType.Line, 'factor backtesting', 1, 1, len(cum_ret), 1,
                 'factor backteset', 'D2', 'time', 'net value', 'Factor Backtest Results',
                 3, 2)

    # --------------------- Quantile backtesting page --------------------------
    worksheet = writer.sheets['Quantile backtesting']
    # chart = workbook.add_chart({'type': 'line'})
    num_cols = len(cum_ret.columns)
    cols = [_ + 1 for _ in range(num_cols)]
    location = xl_rowcol_to_cell(1, num_cols + 2)
    insert_chart(workbook, worksheet, ChartType.Line, 'Quantile backtesting',
                 [3] * num_cols, cols,
                 [3 + len(cum_ret)] * num_cols, cols,
                 ['{}-Portfolio'.format(c[1]) for c in cum_ret.columns],
                 location, 'time', 'net value', 'Quantile Backtest Results',
                 2, 1.5
                 )

    # --------------------- Sorted Portfolios Analysis backtesting page --------------------------
    worksheet = writer.sheets['Sorted Portfolios Analysis']
    num_cols = len(newey_west.columns)
    location = xl_rowcol_to_cell(9, len(mean_ret.columns) + 2)
    insert_NewlyWest_chart(workbook, worksheet, 'Sorted Portfolios Analysis',
                           3, (len(mean_ret.columns) + 2, len(mean_ret.columns) + num_cols + 1), location,
                           'group', 'Avg Return',
                           'Group Return Results',
                           1.2, 1.2, {'type': 'polynomial', 'order': 3, }
                           )

    num_cols = len(mean_ret.columns)
    cols = [_ + 1 for _ in range(num_cols)]
    location = xl_rowcol_to_cell(27, num_cols + 2)
    insert_chart(workbook, worksheet, ChartType.Column,
                 'Sorted Portfolios Analysis',
                 [3] * num_cols, cols,
                 [len(mean_ret) + 2] * num_cols, cols,
                 ['{}-Portfolio'.format(c[1]) for c in mean_ret.columns],
                 location, 'group', 'Monthly Return', 'Group Monthly Return Results',
                 4, 2)

    cols = [_ + 2 * num_cols + 3 for _ in range(num_cols)]
    location = xl_rowcol_to_cell(57, num_cols + 2)
    insert_chart(workbook, worksheet, ChartType.Column,
                 'Sorted Portfolios Analysis',
                 [3] * num_cols, cols,
                 [len(quarter_ret) + 2] * num_cols, cols,
                 ['{}-Portfolio'.format(c[1]) for c in quarter_ret.columns],
                 location, 'group', 'Quarterly Return', 'Group Quarterly Return Results',
                 4, 2)

    cols = [_ + 3 * num_cols + 4 for _ in range(num_cols)]
    location = xl_rowcol_to_cell(87, num_cols + 2)
    insert_chart(workbook, worksheet, ChartType.Column,
                 'Sorted Portfolios Analysis',
                 [3] * num_cols, cols,
                 [len(year_ret) + 2] * num_cols, cols,
                 ['{}-Portfolio'.format(c[1]) for c in year_ret.columns],
                 location, 'group', 'Yearly Return', 'Group Yearly Return Results',
                 4, 2)

    # todo double sort with cap
    # --- color --

    # todo color
    worksheet = writer.sheets['Quantile backtesting']
    for i in range(30, 36):
        three_color_scale(worksheet, i, len(cum_ret.columns) + 2, i,
                          len(cum_ret.columns) + len(backtesting_res.columns) + 3)
    for i in range(37, 45):
        three_color_scale(worksheet, i, len(cum_ret.columns) + 2, i,
                          len(cum_ret.columns) + len(backtesting_res.columns) + 3)

    worksheet = writer.sheets['Sorted Portfolios Analysis']
    for i in range(3, len(mean_ret) + 3):
        three_color_scale(worksheet, i, 1, i, len(mean_ret.columns))
    three_color_scale(worksheet, 3, len(mean_ret.columns) + 2, 3, len(newey_west.columns) + len(mean_ret.columns) + 1)
    t_statistics_heatmap(workbook, worksheet, 4, len(mean_ret.columns) + 2, 4,
                         len(newey_west.columns) + len(mean_ret.columns) + 1)

    worksheet = writer.sheets['Monthly Ret Heatmap']
    for i in range(len(heatmap)):
        three_color_scale(worksheet, 4 + i, 1, 4 + i, 12)
        three_color_scale(worksheet, 4 + i, 13, 4 + i, 24)

    worksheet = writer.sheets['Turnover Analysis']
    for i in range(len(turnover_ratio_heatmap)):
        three_color_scale(worksheet, 3 + i, 5, 3 + i, 16)
        three_color_scale(worksheet, 3 + i, 17, 3 + i, 28)

    # todo stock list
    forward_ret_cols = get_forward_returns_columns(factor_data.columns)
    factor_data = factor_data. \
        reset_index(level=0).sort_values(['date', factor_name]).set_index('date', append=True).swaplevel(0, 1)
    factor_data['rank'] = factor_data.groupby(level=0)[factor_name].rank(method='min', ascending=False)
    # info Chinese name industry daily_basic ...
    if stock_info is not None:
        stock_info.index.names = [factor_data.index.names[1]]
        factor_data = factor_data.join(stock_info[['display_name', 'start_date']])
        factor_data['listed_days'] = factor_data.index.get_level_values(0) - factor_data['start_date']
        factor_data = factor_data.rename(columns={'start_date': 'listed_date'})

    if industry_dict:
        cat = industry_category(industry_dict).astype(str)
        factor_data = factor_data.join(cat)
        if industry_info is not None:
            industry_info.index.names = ['industry_code']
            factor_data = factor_data.set_index('industry_code', append=True)
            industry_info.index = industry_info.index.astype(str)
            factor_data = factor_data.join(industry_info['name']).reset_index(level=2)
            factor_data = factor_data.rename(columns={'name': 'industry_name'})

    if component:
        def _func2(x, u_dict):
            date = x.index.get_level_values(0)[0]
            comp = set(u_dict.get(date, []))
            if len(comp) > 0:
                return pd.Series([c in comp for c in x.index.get_level_values(1)],
                                 index=x.index.get_level_values(1))
            else:
                return pd.Series(False, index=x.index.get_level_values(1))

        for universe_name, u_dict in component.items():
            factor_data['is_{}_component'.format(universe_name)] = \
                factor_data[factor_name].groupby(level=0).apply(lambda x: _func2(x, u_dict))

        # todo universe comparison analysis
        # only consider first, last and long-short quantile
        info_uni = information.copy()
        bt_uni = backtesting_res.copy()
        bt_uni = bt_uni.iloc[:, [0, -2, -1]]
        bt_uni = bt_uni.rename(columns=lambda x: 'Full', level=0)
        cum_ret_uni = cum_ret.copy()
        cum_ret_uni = cum_ret_uni.iloc[:, [0, -2, -1]]
        cum_ret_uni = cum_ret_uni.rename(columns=lambda x: 'Full', level=0)

        for universe_name, u_dict in component.items():
            component_factor_data = factor_data[factor_data['is_{}_component'.format(universe_name)]]
            ic_comp = cal_ic(component_factor_data, factor_name=factor_name)
            information_comp = information_analysis(ic_comp)
            information_comp = information_comp.rename(columns={k: '{}_{}'.format(universe_name, k)
                                                                for k in information_comp.columns})
            info_uni = pd.concat([info_uni, information_comp], axis=1)
            cum_ret_comp, mean_ret_comp, count_stat_comp, std_error_ret_comp = \
                quantile_backtesting(component_factor_data,
                                     factor_name,
                                     bins=bins, demeaned=demeaned)
            mean_ret_comp = mean_ret_comp.iloc[:, [0, -2, -1]]
            mean_ret_comp = mean_ret_comp.rename(columns=lambda x: universe_name, level=0)
            cum_ret_comp = cum_ret_comp.iloc[:, [0, -2, -1]]
            cum_ret_comp = cum_ret_comp.rename(columns=lambda x: universe_name, level=0)
            backtesting_comp = backtesting_metric(mean_ret_comp, cum_ret_comp, period)
            backtesting_comp = backtesting_comp.rename(columns=lambda x: universe_name, level=0)
            bt_uni = pd.concat([bt_uni, backtesting_comp], axis=1)
            cum_ret_uni = pd.concat([cum_ret_uni, cum_ret_comp], axis=1)
        cum_ret_uni.to_excel(writer, sheet_name='Universe Study')
        bt_uni.to_excel(writer, sheet_name='Universe Study', startrow=24,
                        startcol=len(cum_ret_uni.columns) + 2)
        info_uni.to_excel(writer, sheet_name='Universe Study', startrow=24,
                          startcol=len(cum_ret_uni.columns) + len(bt_uni.columns) + 3)

        worksheet = writer.sheets['Universe Study']
        insert_chart2(workbook, worksheet, ChartType.Line,
                      'Universe Study',
                      ('Universe Study', 3, 0, len(cum_ret_uni) + 2, 0),
                      [3] * 3,
                      [i + 1 for i in range(len(cum_ret_uni.columns)) if i % 3 == 0],
                      [len(cum_ret_uni) + 2] * 3,
                      [i + 1 for i in range(len(cum_ret_uni.columns)) if i % 3 == 0],
                      cum_ret_uni.columns.get_level_values(0).drop_duplicates().to_list(),
                      xl_rowcol_to_cell(0, 11),
                      'time', 'net value', 'Group 1', 2, 1.5
                      )

        insert_chart2(workbook, worksheet, ChartType.Line,
                      'Universe Study',
                      ('Universe Study', 3, 0, len(cum_ret_uni) + 2, 0),
                      [3] * 3,
                      [i + 1 for i in range(len(cum_ret_uni.columns)) if i % 3 == 1],
                      [len(cum_ret_uni) + 2] * 3,
                      [i + 1 for i in range(len(cum_ret_uni.columns)) if i % 3 == 1],
                      cum_ret_uni.columns.get_level_values(0).drop_duplicates().to_list(),
                      xl_rowcol_to_cell(0, 26),
                      'time', 'net value', 'Group {}'.format(bins), 2, 1.5
                      )

        insert_chart2(workbook, worksheet, ChartType.Line,
                      'Universe Study',
                      ('Universe Study', 3, 0, len(cum_ret_uni) + 2, 0),
                      [3] * 3,
                      [i + 1 for i in range(len(cum_ret_uni.columns)) if i % 3 == 2],
                      [len(cum_ret_uni) + 2] * 3,
                      [i + 1 for i in range(len(cum_ret_uni.columns)) if i % 3 == 2],
                      cum_ret_uni.columns.get_level_values(0).drop_duplicates().to_list(),
                      xl_rowcol_to_cell(0, 41),
                      'time', 'net value', 'Group {}-1'.format(bins), 2, 1.5
                      )

    if daily_basic is not None:
        factor_data = factor_data.join(daily_basic[['close', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
                                                    'dv_ttm', 'total_share', 'float_share', 'free_share',
                                                    'total_mv', 'circ_mv'
                                                    ]])
        if industry_dict is not None:
            neutralized_factor_name = 'neutralized_' + factor_name
            factor_data[neutralized_factor_name] = cap_industry_neutralize(
                factor_data[[factor_name, 'industry_code', 'circ_mv']],
                factor_name, 'industry_code', 'circ_mv', 'winsorize_norm',
                winsorize_limits=0.5 / bins
            )
            industry_factor_data = factor_data[[factor_name, neutralized_factor_name,
                                                'industry_code', 'circ_mv'
                                                ] + forward_ret_cols]

            ic_ind = cal_ic(industry_factor_data, factor_name=neutralized_factor_name)
            information_ind = information_analysis(ic_ind)
            information_ind = information_ind.rename(columns={k: 'neu_factor_' + k for k in information_ind.columns})
            information_ind = pd.concat([information, information_ind], axis=1)
            cum_ret2, mean_ret2, count_stat2, std_error_ret2 = quantile_backtesting(industry_factor_data,
                                                                                    neutralized_factor_name,
                                                                                    bins=bins, demeaned=demeaned)

            backtesting_res2 = backtesting_metric(mean_ret2, cum_ret2, period)

            backtesting_res2.to_excel(writer, sheet_name='Ind Neu', startrow=24,
                                      startcol=len(cum_ret2.columns) + 2)
            information_ind.to_excel(writer, sheet_name='Ind Neu', startrow=24,
                                     startcol=len(cum_ret2.columns) + len(backtesting_res2.columns) + 3)

            cum_ret2.to_excel(writer, sheet_name='Ind Neu')

            # todo plot
            worksheet = writer.sheets['Ind Neu']
            insert_chart3(workbook, worksheet, ChartType.Line, ['Quantile backtesting', 'Ind Neu'],
                          ('Ind Neu', 3, 0, len(cum_ret2) + 2, 0),
                          [3, 3], [1, 1], [2 + len(cum_ret2), 2 + len(cum_ret2)],
                          [1, 1], ['Origin', 'Ind Neu'],
                          xl_rowcol_to_cell(0, 11), 'time', 'net_value',
                          'Group 1', 2, 1.5
                          )
            insert_chart3(workbook, worksheet, ChartType.Line, ['Quantile backtesting', 'Ind Neu'],
                          ('Ind Neu', 3, 0, len(cum_ret2) + 2, 0),
                          [3, 3], [bins, bins], [2 + len(cum_ret2), 2 + len(cum_ret2)],
                          [bins, bins], ['Origin', 'Ind Neu'],
                          xl_rowcol_to_cell(0, 26), 'time', 'net_value',
                          'Group {}'.format(bins), 2, 1.5
                          )
            insert_chart3(workbook, worksheet, ChartType.Line, ['Quantile backtesting', 'Ind Neu'],
                          ('Ind Neu', 3, 0, len(cum_ret2) + 2, 0),
                          [3, 3], [1 + bins, 1 + bins], [2 + len(cum_ret2), 2 + len(cum_ret2)],
                          [1 + bins, 1 + bins], ['Origin', 'Ind Neu'],
                          xl_rowcol_to_cell(0, 41), 'time', 'net_value',
                          'Group {}-1'.format(bins), 2, 1.5
                          )

    factor_data.to_excel(writer, sheet_name='raw_factor', merge_cells=False)
    worksheet = writer.sheets['raw_factor']
    worksheet.autofilter(0, 0, len(factor_data) + 1, len(factor_data.columns) + 1)
    start = xl_rowcol_to_cell(1, factor_data.columns.get_loc(forward_ret_cols[0]) + 2)
    end = xl_rowcol_to_cell(1 + len(factor_data), factor_data.columns.get_loc(forward_ret_cols[0]) + 2)
    worksheet.conditional_format('{}:{}'.format(start, end),
                                 {'type': 'data_bar',
                                  'bar_negative_color': '#FF2D2D',
                                  'min_value': -0.3,
                                  'max_value': 0.3,
                                  'bar_color': '#00BB00'
                                  })

    if daily_market is not None:
        if sparkline:
            close = daily_market['adj_close'].unstack()
            exit_dates = factor_data.index.get_level_values(0).drop_duplicates().to_frame().shift(-1)['date'].fillna(
                close.index[-1]).to_dict()
            keep = pd.DataFrame(np.empty_like(close.values).fill(np.nan), index=close.index, columns=close.columns)

            def _func(keep_, close_, t, c):
                keep_.loc[t:exit_dates[t], c] = \
                    close_.loc[t:exit_dates[t], c]

            with ThreadPoolExecutor(max_workers=10) as pool:
                for t, c in factor_data.index:
                    pool.submit(_func, keep, close, t, c)
            pool.shutdown()
            keep = keep.dropna(axis=1, how='all')
            keep = keep.dropna(axis=0, how='all')  # type: pd.DataFrame
            keep.to_excel(writer, sheet_name='adj_close_prices')
            col = len(factor_data.columns) + 2

            def _add_sparkline(row, ticker, start_date, row_data):
                if row_data['group'] not in [1, bins]:
                    return
                s = keep.index.get_loc(start_date) + 1
                e = keep.index.get_loc(exit_dates[start_date]) + 1
                c = keep.columns.get_loc(ticker) + 1
                s = xl_rowcol_to_cell(s, c)
                e = xl_rowcol_to_cell(e, c)
                range_str = "'adj_close_prices'!{}:{}".format(s, e)
                setting = {'range': range_str,
                           'high_point': True,
                           'low_point': True,
                           'series_color': '#00BB00' if row_data[forward_ret_cols[0]] > 0 else '#FF2D2D',
                           'weight': 1.25
                           }
                worksheet.add_sparkline(row, col, setting)

            with ThreadPoolExecutor(max_workers=10) as pool:
                for i, (index, row_data) in enumerate(factor_data.iterrows(), 1):
                    pool.submit(_add_sparkline, i, index[1], index[0], row_data)

    if top_portfolio:
        pass


    # todo daily backtesting

    writer.save()


def multi_factor_report(factor_data: pd.DataFrame,
                        factor_names: list,
                        market_data: pd.DataFrame,
                        benchmark_data: pd.DataFrame or None = None,
                        industry_index_data: pd.DataFrame or None = None,
                        quantiles=None, bins=5,
                        result_save_path='../multi_factors_report',
                        market_filter_dict: dict or tuple = None,
                        industry_dict=None,
                        stock_selection_num: int or tuple or None = None
                        ):
    """

    :param factor_data:
    :param factor_names:
    :param market_data:
    :param benchmark_data:
    :param industry_index_data:
    :param quantiles:
    :param bins:
    :param result_save_path:
    :param market_filter_dict:
    :param industry_dict:
    :param stock_selection_num:
    :return:
    """
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    save_file_name = os.path.join(result_save_path, '_'.join(factor_names) + '_test.xlsx')
    # if os.path.exists(save_file_name):
    #     raise ValueError('{} exists'.format(save_file_name))

    writer = pd.ExcelWriter(save_file_name, engine='xlsxwriter')

    factors = factor_data[factor_names]
    pearson_corr, spearman_corr = factors_correlation(factors)
    # todo rolling corr

    pearson_corr.to_excel(writer, sheet_name='Correlation')
    spearman_corr.to_excel(writer, sheet_name='Correlation', startrow=len(pearson_corr) + 1)
    # color the correlation matrix
    worksheet = writer.sheets['Correlation']
    corr_matrix_heatmap(worksheet, 1, 1, len(pearson_corr), len(pearson_corr))
    corr_matrix_heatmap(worksheet, len(pearson_corr) + 1, 1, len(pearson_corr) + 1 + len(spearman_corr),
                        len(pearson_corr))

    # market = market_filter_in(market, [zz_500, csi_300])
    returns = calculate_forward_returns(market_data, [1], price_key='close')
    market = market_data.join(returns).dropna()
    data = combine_market_with_fundamental(market, factors)
    data = data.replace([-np.inf, np.inf], np.nan)
    data = data.dropna()

    universe_selected_data = None
    if market_filter_dict:
        universe_selected_data = dict()
        for k, v in market_filter_dict.items():
            universe_selected_data[k] = market_filter_in(data, v)
        # cumulative version
        for i in range(2, len(market_filter_dict) + 1):
            combine_universe = combinations(market_filter_dict.keys(), i)
            for u in combine_universe:
                key = '+'.join(u)
                universe_selected_data[key] = market_filter_in(data, [market_filter_dict[k] for k in u])

    # --------------------- factor return comparison ---------------------
    factor_backtest = pd.DataFrame()
    idx = data.index.get_level_values(0).drop_duplicates().sort_values()
    start = idx[0]
    end = idx[-1]

    if benchmark_data:
        pass

    for factor_name in factor_names:
        factor_cum = factor_backtesting(data, factor_name, equal_weight=True)
        factor_backtest['All_' + factor_name] = factor_cum['forward_1_period_net_value']
        if market_filter_dict:
            for key, value in universe_selected_data.items():
                factor_cum = factor_backtesting(value, factor_name, equal_weight=True)
                factor_backtest[key + '_' + factor_name] = factor_cum['forward_1_period_net_value']

    if stock_selection_num:
        for factor_name in factor_names:
            for num in stock_selection_num:
                factor_cum = factor_backtesting(data, factor_name, equal_weight=True, selection_num=num)
                factor_backtest['All_{}_{}'.format(num, factor_name)] = factor_cum['forward_1_period_net_value']
                if market_filter_dict is not None:
                    for key, value in universe_selected_data.items():
                        factor_cum = factor_backtesting(value, factor_name, equal_weight=True, selection_num=num)
                        factor_backtest['{}_{}_{}'.format(key, num, factor_name)] = factor_cum[
                            'forward_1_period_net_value']

    factor_backtest = factor_backtest.fillna(method='ffill').fillna(1)
    factor_backtest.to_excel(writer, sheet_name='factor_backtesting', startcol=len(factor_backtest.columns) + 1)
    bt_returns = factor_backtest.pct_change()
    bt_metric = backtesting_metric(bt_returns, factor_backtest)
    bt_metric.to_excel(writer, sheet_name='factor_backtesting', startrow=30)

    returns_corr = bt_returns.corr()
    returns_corr.to_excel(writer, sheet_name='factor_backtesting', startrow=len(bt_metric) + 32)
    workbook = writer.book
    # --------------------- factor_backtesting page --------------------------
    worksheet = writer.sheets['factor_backtesting']
    chart = workbook.add_chart({'type': 'line'})

    # Configure the series of the chart from the dataframe data.

    #     [sheetname, first_row, first_col, last_row, last_col]
    for i in range(len(factor_backtest.columns)):
        chart.add_series({
            'name': factor_backtest.columns[i],
            'categories': ['factor_backtesting', 1, len(factor_backtest.columns) + 1, len(factor_backtest),
                           len(factor_backtest.columns) + 1],
            'values': ['factor_backtesting', 1, i + len(factor_backtest.columns) + 2, len(factor_backtest),
                       i + len(factor_backtest.columns) + 2],
        })

    # Configure the chart axes.
    chart.set_x_axis({'name': 'time'})

    chart.set_y_axis({'name': 'net value',
                      'major_gridlines': {'visible': False},
                      'min': factor_backtest.min().min() * 0.9,
                      'max': factor_backtest.max().max() * 1.1})
    chart.set_title({'name': 'Factor Backtest Results'})
    chart.set_size({'x_scale': 3, 'y_scale': 2})
    # Insert the chart into the worksheet.
    worksheet.insert_chart('A1', chart)

    # universe study
    i = 0
    for name in factor_names:
        columns_list = [c for c in bt_metric.columns if name in c]
        mean_var_metric = bt_metric.loc[['Compounded Ann Growth Rate', 'Ret Vol',
                                         'Ret Skew', 'Ret Kurt', 'sharpe', 'sortino'], columns_list]
        mean_var_metric.to_excel(writer, sheet_name='universe study', startrow=i, startcol=30)

        # todo plot
        worksheet = writer.sheets['universe study']
        chart = workbook.add_chart({'type': 'scatter'})

        # Configure the series of the chart from the dataframe data.

        #     [sheetname, first_row, first_col, last_row, last_col]
        for j in range(len(mean_var_metric.columns)):
            chart.add_series({
                'name': mean_var_metric.columns[j].replace('_{}'.format(name), ''),
                'categories': ['universe study', i + 2, 31 + j, i + 2, 31 + j],
                'values': ['universe study', i + 1, 31 + j, i + 1, 31 + j],
                'data_labels': {'series_name': True},
            })

        # Configure the chart axes.
        chart.set_x_axis({'name': 'Volatility',
                          'major_gridlines': {
                              'visible': True,
                              'line': {'dash_type': 'dash'}
                          },
                          })
        chart.set_y_axis({'name': 'Returns',
                          'major_gridlines': {
                              'visible': True,
                              'line': {'dash_type': 'dash'}
                          },
                          # 'major_gridlines': {'visible': False},
                          })
        chart.set_title({'name': '{} universe comparison'.format(name)})
        chart.set_size({'x_scale': 2, 'y_scale': 2})
        # Insert the chart into the worksheet.
        worksheet.insert_chart(4 * i, 0, chart)
        i += len(mean_var_metric) + 2

    # predictive variable study

    # pca study
    if len(factor_names) > 5:
        pass

    # return path study
    bt_returns = bt_returns.sort_index(axis=1)  # type: pd.DataFrame
    bt_returns.to_excel(writer, sheet_name='return_path', startcol=30)
    worksheet = writer.sheets['return_path']
    universe = list(set([c.split('_')[0] for c in bt_returns.columns]))

    for k in universe:
        columns_list = [c for c in bt_returns.columns if k == c.split('_')[0]]
        print(columns_list)

        # chart = workbook.add_chart({'type': 'line'})

    # industry
    if industry_dict:
        pass
    # factor normalized study
    writer.save()


if __name__ == '__main__':
    # trading_dates =
    factors = load_single_category_factors('../data/factors/profit', )
    names = factors.columns.to_list()
    # factors = pd.read_parquet('../data/factors/tmp.parquet')
    factors = factors.sort_index()
    factors = factors.loc[pd.to_datetime('2019-01-01'):]
    market = pd.read_parquet('../data/all_M_data.parquet')
    market = market.loc[pd.to_datetime('2019-01-01'):]

    zz_500 = load_pickle('../data/universe/000905.XSHG.pickle')
    csi_300 = load_pickle('../data/universe/000300.XSHG.pickle')

    # market = market_filter_in(market, [zz_500, csi_300])
    # returns = calculate_forward_returns(market, [1], price_key='close')
    # market = market.join(returns).dropna()
    # data = combine_market_with_fundamental(market, factors)
    # data = data.replace([-np.inf, np.inf], np.nan)
    market_filter_dict = {'csi300': csi_300, 'zz500': zz_500}

    multi_factor_report(factors, names, market, market_filter_dict=market_filter_dict,
                        stock_selection_num=(10, 20, 30))
