import os
from typing import List, Union, Optional

import pandas as pd
import quantstats as qs
from plotly import graph_objs as go

from backtesting.Account import Account
from data_management.dataIO import market_data, index_data
from data_management.dataIO.component_data import get_industry_info, IndustryCategory, get_industry_component
from data_management.dataIO.exotic_data import get_exotic, Exotic
from data_management.dataIO.export_report import generate_selection_excel
from data_management.dataIO.fundamental_data import get_stock_info
from data_management.dataIO.index_data import IndexTicker
from factor_zoo.industry import industry_category
from graph.backtest_graph import net_value_line, timing_contour
from graph.bar_component import candlestick
from graph.indicator_component import volume, buy_marker_graph, sell_marker_graph, announcement_graph
from graph.stock_graph import stick_and_volume


class BacktestAnalyst:

    def __init__(self, account: Account, data_config_path: str,
                 analysis_name="Backtest analysis"
                 ):
        self.account = account
        self.data_config_path = data_config_path
        self.market_data = market_data.get_bars(config_path=self.data_config_path, eod_time_adjust=False,
                                                add_limit=False, verbose=0)
        self.securities_info = get_stock_info(config_path=data_config_path)
        self.announcement = get_exotic(Exotic.announcement, config_path=data_config_path, verbose=0)
        self.sw1_info = get_industry_info(IndustryCategory.sw_l1, config_path=data_config_path)
        self.sw_l1 = get_industry_component(IndustryCategory.sw_l1, config_path=data_config_path)
        self.analysis_name = analysis_name

    def get_trade_records_plot(self, code: str, start=None, end=None):
        bars = self.market_data.loc[:, code, :]
        records = self.account.get_full_trade_records()
        records = records[records['code'] == code]
        if start is None:
            start = records['date'].iloc[0] - pd.Timedelta(days=20)

        if end is None:
            end = records['date'].iloc[-1] + pd.Timedelta(days=20)

        #           date       code  trade_price  ...  trade_id  trade_cum_cashflow     cost
        # 1523 2018-08-08  603885.SH     37.10940  ...       1.0       -1.539817e+06  37.1094
        # 1538 2018-08-15  603885.SH     36.72954  ...       1.0       -1.576191e+04      inf
        # 2407 2018-12-26  603885.SH     36.08670  ...       2.0       -1.351952e+06  36.0867
        # 2433 2018-12-28  603885.SH     36.61266  ...       2.0        1.970457e+04     -inf

        bars = bars.loc[pd.to_datetime(start): pd.to_datetime(end)]
        bars = bars.droplevel(1)

        candlestick_plot = candlestick(bars, ohlc_key=['adj_open', 'adj_high', 'adj_low', 'adj_close'], symbol=code,
                                       increasing_fillcolor='#02DF82',
                                       increasing_line_color='#02DF82',
                                       decreasing_fillcolor='red',
                                       decreasing_line_color='red'
                                       )

        volume_plot = volume(bars, volume_key='money')
        fig = stick_and_volume(candlestick_plot, volume_plot)
        buy_entry_records = records[(records['volume'] > 0)]
        fig.add_trace(buy_marker_graph(buy_entry_records), row=1, col=1)
        sell_entry_records = records[(records['volume'] < 0)]
        fig.add_trace(sell_marker_graph(sell_entry_records), row=1, col=1)
        try:
            announcement = self.announcement.loc[pd.to_datetime(start): pd.to_datetime(end), code, :].reset_index(
                level=1)
            fig.add_trace(announcement_graph(announcement, bars['adj_high'].max() * 1.05), row=1, col=1)
        except:
            pass
        # https://plotly.com/python/marker-style/
        # https://plotly.com/python/text-and-annotations/
        # fig.add_annotation(x=2, y=5,
        #             text="Text annotation with arrow",
        #             showarrow=True,
        #             arrowhead=1)
        fig.update_layout(template='plotly_dark',
                          title="{} {}".format(code, self.securities_info.loc[code].display_name),
                          yaxis_title="Price",
                          xaxis_rangeslider_visible=False)
        return fig

    def get_holding_num_plot(self):
        holding = self.account.get_history_holding()
        # todo

    def add_stock_info(self, stock_series: pd.Series):
        if stock_series.index.nlevels != 2:
            raise ValueError

        stock_series.index.names = ['date', 'code']

        sec_info = self.securities_info
        industry_info = self.sw1_info
        sec_info.index.names = ['code']
        cat = industry_category(self.sw_l1).astype(str)
        industry_info.index.names = ['industry_code']
        industry_info.index = industry_info.index.astype(str)

        stock_df = stock_series.to_frame().join(sec_info['display_name']).join(cat)
        stock_df = stock_df.set_index('industry_code', append=True)
        stock_df = stock_df.join(industry_info['name']).reset_index(level=2)
        stock_df = stock_df.rename(columns={'name': 'sw1_name'})
        stock_df = stock_df[['display_name', 'industry_code', 'sw1_name', stock_series.name]]
        return stock_df

    def get_daily_winner_loser(self, n: int):
        ret = self.market_data['adj_close'].unstack().pct_change()
        holding = self.account.get_history_holding()
        eod_hold = (holding > 0).unstack()
        holding_eod_ret = (eod_hold * ret).stack().astype(float)
        grouper = holding_eod_ret.groupby(level=0)
        winner = grouper.nlargest(n).droplevel(0)
        loser = grouper.nsmallest(n).droplevel(0)

        winner.name = 'ret'
        winner = self.add_stock_info(winner)

        loser.name = 'ret'
        loser = self.add_stock_info(loser)
        return winner, loser

    def save_winner_loser_table(self, n, save_path):
        writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
        winner, loser = self.get_daily_winner_loser(n)
        winner.to_excel(writer, sheet_name='winner', merge_cells=False)
        loser.to_excel(writer, sheet_name='loser', merge_cells=False)
        writer.save()

    def get_excess_return(self, benchmark: IndexTicker, start_date=None, end_date=None):
        benchmark = index_data.get_bars(benchmark, config_path=self.data_config_path, eod_time_adjust=False)
        benchmark = benchmark['close'].pct_change()
        info = self.account.get_account_info()
        excess_ret = (info['cap'].pct_change().dropna() - benchmark).dropna()
        if start_date is not None:
            excess_ret = excess_ret.loc[pd.to_datetime(start_date):]
        if end_date is not None:
            excess_ret = excess_ret.loc[:pd.to_datetime(end_date)]

        return excess_ret

    def get_excess_benchmark_return_plot(self, benchmark: Union[IndexTicker, List[IndexTicker]],
                                         start_date=None,
                                         end_date=None) -> go.Figure:
        fig = go.Figure()
        if isinstance(benchmark, IndexTicker):
            benchmark = [benchmark]

        for b in benchmark:
            excess_ret = self.get_excess_return(b, start_date, end_date)
            fig.add_trace(net_value_line(excess_ret.cumsum(), color=None, name='Excess return {}'.format(b.name)))
        fig.update_layout(title='Excess return')
        return fig

    def get_return_plot(self,
                        benchmark: Optional[Union[IndexTicker, List[IndexTicker]]] = None,
                        start_date=None,
                        end_date=None, timing: Optional[pd.Series]=None) -> go.Figure:
        fig = go.Figure()
        info = self.account.get_account_info()
        cap = info['cap']
        if start_date is not None:
            cap = cap.loc[pd.to_datetime(start_date):]

        if end_date is not None:
            cap = cap.loc[:pd.to_datetime(end_date)]

        fig.add_trace(net_value_line(cap / cap.iloc[0], name='strategy return'))
        if benchmark is not None:
            if isinstance(benchmark, IndexTicker):
                benchmark = [benchmark]
            for b in benchmark:
                benchmark = index_data.get_bars(b, config_path=self.data_config_path, eod_time_adjust=False)
                benchmark = benchmark['close']
                benchmark = benchmark.loc[cap.index[0]: cap.index[-1]]
                fig.add_trace(
                    net_value_line(benchmark / benchmark.iloc[0], color=None, name='{}'.format(b.name)))
        if timing is not None:
            t = timing.copy()
            t = t.reindex(cap.index)
            t = t.fillna(method='ffill')
            t = t.shift(1)
            fig.add_trace(timing_contour(t, (1, (cap / cap.iloc[0]).max())))

        return fig

    def save_absolute_return_report(self, benchmark: IndexTicker, save_path, start_date=None, end_date=None):
        info = self.account.get_account_info()
        benchmark_name = benchmark.name
        benchmark = index_data.get_bars(benchmark, config_path=self.data_config_path, eod_time_adjust=False)
        benchmark = benchmark['close'].pct_change()
        title_name = '{} ({})'.format(self.analysis_name, benchmark_name)
        cap = info['cap'].pct_change().dropna()
        if start_date is not None:
            cap = cap.loc[pd.to_datetime(start_date):]
            # benchmark = benchmark.loc[pd.to_datetime(start_date):]

        if end_date is not None:
            cap = cap.loc[:pd.to_datetime(end_date)]
        qs.reports.html(cap, benchmark=benchmark,
                        title=title_name,
                        output=save_path)

    def save_excess_benchmark_return_report(self, benchmark: IndexTicker, save_path, start_date=None, end_date=None):
        excess_ret = self.get_excess_return(benchmark, start_date, end_date)
        qs.reports.html(excess_ret,
                        title='{} Excess {} return'.format(self.analysis_name, benchmark.name),
                        output=save_path, compounded=False)

    def save_entry_exit_plot(self, save_path: str):
        records = self.account.get_full_trade_records()
        os.makedirs(save_path, exist_ok=True)
        for (code, trade_id), data in records.groupby(['code', 'trade_id']):
            start = (data['date'].iloc[0]).strftime('%Y-%m-%d')
            try:
                end = (data['date'].iloc[-1]).strftime('%Y-%m-%d')
            except:
                end = records['date'].max().strftime('%Y-%m-%d')
                trade_id = 1
            fig = self.get_trade_records_plot(code, data['date'].iloc[0] - pd.Timedelta(days=20),
                                              data['date'].iloc[-1] + pd.Timedelta(days=20))
            file_name = '{}_{}_{}_{}_{}.html'.format(code, self.securities_info.loc[code].display_name, start, end,
                                                     int(trade_id)
                                                     ).replace('*', '')
            fig.write_html(os.path.join(save_path, file_name))

    def save_result(self,
                    save_path: str,
                    benchmark: Optional[Union[IndexTicker, List[IndexTicker]]],
                    start_date=None,
                    end_date=None):
        os.makedirs(save_path, exist_ok=True)
        if isinstance(benchmark, IndexTicker):
            benchmark = [benchmark]

        for b in benchmark:
            self.save_excess_benchmark_return_report(b,
                                                     os.path.join(save_path,
                                                                  '{}_excess_return_{}_report.html'
                                                                  .format(self.analysis_name, b.name)), start_date,
                                                     end_date)
            self.save_absolute_return_report(b, os.path.join(save_path,
                                                             '{}_absolute_return_{}_report.html'
                                                             .format(self.analysis_name, b.name)), start_date, end_date)

        self.save_winner_loser_table(5, os.path.join(save_path, 'top_5_winner_losser.xlsx'))


class CryptoBacktestAnalyst(BacktestAnalyst):
    def __init__(self, bar_data, account, analysis_name="Backtest analysis"):
        self.market_data = bar_data
        self.account = account
        self.analysis_name = analysis_name


    def save_absolute_return_report(self, save_path, start_date=None, end_date=None):
        info = self.account.get_account_info()
        title_name = '{}'.format(self.analysis_name)
        cap = info['cap'].resample('D').last().pct_change().dropna()
        if start_date is not None:
            cap = cap.loc[pd.to_datetime(start_date):]
            # benchmark = benchmark.loc[pd.to_datetime(start_date):]

        if end_date is not None:
            cap = cap.loc[:pd.to_datetime(end_date)]
        qs.reports.html(cap,
                        title=title_name,
                        output=save_path)