import warnings
from collections import OrderedDict, defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pandas.tseries.offsets import BDay
# import utils
from scipy import stats
from scipy.stats import norm
from statsmodels import api as sm

from factor_testing.utils import get_forward_returns_columns, demean_forward_returns


def cal_ic(factor_data: pd.DataFrame, factor_name: str = 'factor',
           group_adjust=False,
           by_group=False):
    def src_ic(group):
        f = group[factor_name]
        _ic = group[get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values(0)]

    if group_adjust:
        factor_data = demean_forward_returns(factor_data, grouper + ['group'])
    if by_group:
        grouper.append('group')

    ic = factor_data.groupby(grouper).apply(src_ic)
    ic.columns = [c + '_IC' for c in ic.columns]
    #             1_period_return  2_period_return  5_period_return  10_period_return
    # date
    # 2006-03-29         0.060898         0.075311         0.020372         -0.132485
    # 2006-03-30         0.029026        -0.023851        -0.144570         -0.146061
    # 2006-03-31        -0.098332        -0.048231        -0.252450         -0.136901
    # 2006-04-03         0.039630        -0.000007        -0.218708         -0.042413
    return ic


def cumulative_ic(cross_sectional_ic: pd.DataFrame):
    return cross_sectional_ic.cumsum()


def ir(cross_sectional_ic: pd.DataFrame):
    return cross_sectional_ic.mean() / cross_sectional_ic.std()


def ic_ability_ratio(ic_df: pd.DataFrame, ratio=0.02):
    ratio = (ic_df.abs() >= ratio).sum() / len(ic_df)
    return ratio


def factor_weights(factor_data, factor_name: str,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False, selection_num=None):
    """
    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Should this computation happen on a long short portfolio? if True,
        weights are computed by demeaning factor values and dividing by the sum
        of their absolute value (achieving gross leverage of 1). The sum of
        positive weights will be the same as the negative weights (absolute
        value), suitable for a dollar neutral long-short portfolio
    group_adjust : bool
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral weights: each group will weight the same and
        if 'demeaned' is enabled the factor values demeaning will occur on the
        group level.
    equal_weight : bool, optional
        if True the assets will be equal-weighted instead of factor-weighted
        If demeaned is True then the factor universe will be split in two
        equal sized groups, top assets with positive weights and bottom assets
        with negative weights

    Returns
    -------
    returns : pd.Series
        Assets weighted by factor value.
    """

    def to_weights(group: pd.Series, _demeaned, _equal_weight, _selection_num):

        if _equal_weight:
            group = group.copy()

            if _demeaned:
                # top assets positive weights, bottom ones negative
                group = group - group.median()

            if _selection_num:
                sorted_group = group.sort_values()
                if len(group) > 2 * _selection_num:
                    sorted_group.iloc[_selection_num: (len(group) - _selection_num)] = 0
                    zero_mask = sorted_group == 0
                    group[zero_mask] = 0

            negative_mask = group < 0
            group[negative_mask] = -1.0
            positive_mask = group > 0
            group[positive_mask] = 1.0

            if _demeaned:
                # positive weights must equal negative weights
                if negative_mask.any():
                    group[negative_mask] /= negative_mask.sum()
                if positive_mask.any():
                    group[positive_mask] /= positive_mask.sum()

        elif _demeaned:
            group = group - group.mean()

        return group / group.abs().sum()

    grouper = [factor_data.index.get_level_values(0)]
    if group_adjust:
        grouper.append('group')

    weights = factor_data.groupby(grouper)[factor_name] \
        .apply(to_weights, demeaned, equal_weight, selection_num)

    if group_adjust:
        weights = weights.groupby(level=0).apply(to_weights, False, False)

    return weights


def quantize_factor(merged_data: pd.DataFrame, factor_name: str, quantiles: list = None, bins: int = None,
                    grouped=False):
    """
    merged_data multi index, level 0 is pd.Timestamp, level 1 is asset code.
    two column, factor and 'group'
    :param grouped:
    :param merged_data:
    :param quantiles:
    :param bins:
    :return:
    """
    merged_data = merged_data.copy().drop_duplicates()
    if not ((quantiles is not None and bins is None) or
            (quantiles is None and bins is not None)):
        raise ValueError('Either quantiles or bins should be provided')

    grouper = [merged_data.index.get_level_values(level=0)]
    if 'group' in merged_data.columns and grouped is True:
        grouper.append('group')

    def quantile_calc(x, _quantiles, _bins):
        if _quantiles is not None and _bins is None:
            return pd.qcut(x, _quantiles, labels=False) + 1
        elif _bins is not None and _quantiles is None:
            return pd.qcut(x, _bins, labels=False, duplicates='drop') + 1

    factor_quantile = merged_data.groupby(grouper)[factor_name] \
        .apply(quantile_calc, quantiles, bins)
    factor_quantile.name = 'factor_quantile'
    return factor_quantile


def factor_returns(factor_data, factor_name: str,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   by_asset=False,
                   selection_num=None):
    """
    Computes period wise returns for portfolio weighted by factor
    values.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    group_adjust : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    equal_weight : bool, optional
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    by_asset: bool, optional
        If True, returns are reported separately for each esset.

    selection_num: int, optional
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    Returns
    -------
    returns : pd.DataFrame
        Period wise factor returns
    """

    weights = \
        factor_weights(factor_data, factor_name, demeaned, group_adjust, equal_weight, selection_num)

    weighted_returns = \
        factor_data[get_forward_returns_columns(factor_data.columns)] \
            .multiply(weights, axis=0)

    if by_asset:
        returns = weighted_returns
    else:
        returns = weighted_returns.groupby(level=0).sum()
    returns = returns.fillna(0)
    return returns


def cumulative_returns(returns):
    """
    Computes cumulative returns from simple daily returns.

    Parameters
    ----------
    returns: pd.Series
        pd.Series containing daily factor returns (i.e. '1D' returns).

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-01-05   1.001310
            2015-01-06   1.000805
            2015-01-07   1.001092
            2015-01-08   0.999200
    """
    return returns.add(1).cumprod()


def positions(weights, period, freq=None):
    raise NotImplementedError
    """
    Builds net position values time series, the portfolio percentage invested
    in each position.

    Parameters
    ----------
    weights: pd.Series
        pd.Series containing factor weights, the index contains timestamps at
        which the trades are computed and the values correspond to assets
        weights
        - see factor_weights for more details
    period: pandas.Timedelta or string
        Assets holding period (1 day, 2 mins, 3 hours etc). It can be a
        Timedelta or a string in the format accepted by Timedelta constructor
        ('1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        weights.index.freq will be used

    Returns
    -------
    pd.DataFrame
        Assets positions series, datetime on index, assets on columns.
        Example:
            index                 'AAPL'         'MSFT'          cash
            2004-01-09 10:30:00   13939.3800     -14012.9930     711.5585
            2004-01-09 15:30:00       0.00       -16012.9930     411.5585
            2004-01-12 10:30:00   14492.6300     -14624.8700       0.0
            2004-01-12 15:30:00   14874.5400     -15841.2500       0.0
            2004-01-13 10:30:00   -13853.2800    13653.6400      -43.6375
    """

    weights = weights.unstack()

    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)

    if freq is None:
        freq = weights.index.freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar",
                      UserWarning)

    #
    # weights index contains factor computation timestamps, then add returns
    # timestamps too (factor timestamps + period) and save them to 'full_idx'
    # 'full_idx' index will contain an entry for each point in time the weights
    # change and hence they have to be re-computed
    #
    trades_idx = weights.index.copy()
    returns_idx = utils.add_custom_calendar_timedelta(trades_idx, period, freq)
    weights_idx = trades_idx.union(returns_idx)

    #
    # Compute portfolio weights for each point in time contained in the index
    #
    portfolio_weights = pd.DataFrame(index=weights_idx,
                                     columns=weights.columns)
    active_weights = []

    for curr_time in weights_idx:

        #
        # fetch new weights that become available at curr_time and store them
        # in active weights
        #
        if curr_time in weights.index:
            assets_weights = weights.loc[curr_time]
            expire_ts = utils.add_custom_calendar_timedelta(curr_time,
                                                            period, freq)
            active_weights.append((expire_ts, assets_weights))

        #
        # remove expired entry in active_weights (older than 'period')
        #
        if active_weights:
            expire_ts, assets_weights = active_weights[0]
            if expire_ts <= curr_time:
                active_weights.pop(0)

        if not active_weights:
            continue
        #
        # Compute total weights for curr_time and store them
        #
        tot_weights = [w for (ts, w) in active_weights]
        tot_weights = pd.concat(tot_weights, axis=1)
        tot_weights = tot_weights.sum(axis=1)
        tot_weights /= tot_weights.abs().sum()

        portfolio_weights.loc[curr_time] = tot_weights

    return portfolio_weights.fillna(0)


def mean_return_by_quantile(factor_data,
                            by_date=False,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False,
                            weight_='equal'):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
    demeaned : bool
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level.

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values(0)] + ['group']
        factor_data = demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ['factor_quantile', factor_data.index.get_level_values(0)]

    if by_group:
        grouper.append('group')

    if weight_ == 'equal':
        group_stats = factor_data.groupby(grouper)[
            get_forward_returns_columns(factor_data.columns)] \
            .agg(['mean', 'std', 'count'])
    elif weight_ == 'cap':
        raise NotImplementedError
    else:
        raise NotImplementedError

    mean_ret = group_stats.T.xs('mean', level=1).T

    if not by_date:
        grouper = [mean_ret.index.get_level_values(0)]
        if by_group:
            grouper.append(mean_ret.index.get_level_values('group'))
        group_stats = mean_ret.groupby(grouper) \
            .agg(['mean', 'std', 'count'])
        mean_ret = group_stats.T.xs('mean', level=1).T

    std_error_ret = group_stats.T.xs('std', level=1).T \
                    / np.sqrt(group_stats.T.xs('count', level=1).T)
    count_stat = group_stats.T.xs('count', level=1).T
    mean_ret.columns = [c.replace('forward_', '').replace('return', weight_ + '_return') for c in mean_ret.columns]
    std_error_ret.columns = [c.replace('forward_', '').replace('return', 'std') for c in std_error_ret.columns]
    count_stat.columns = [c.replace('forward_', '').replace('return', 'count') for c in count_stat.columns]
    return mean_ret, std_error_ret, count_stat


def factor_backtesting(factor_data: pd.DataFrame, factor_name: str,
                       demeaned=True,
                       group_adjust=False,
                       equal_weight=False,
                       selection_num=None
                       ):
    # calculate factor ret

    portfolio_ret = factor_returns(factor_data, factor_name, demeaned, group_adjust, equal_weight,
                                   selection_num=selection_num)
    cum_portfolio_ret = cumulative_returns(portfolio_ret)
    initial = pd.DataFrame(1, index=[cum_portfolio_ret.index[0]], columns=cum_portfolio_ret.columns)

    idx = cum_portfolio_ret.index[1:].to_list()
    idx.append(cum_portfolio_ret.index[-1] + pd.Timedelta(weeks=4))
    cum_portfolio_ret.index = idx
    cum_portfolio_ret = initial.append(cum_portfolio_ret)
    cum_portfolio_ret = cum_portfolio_ret. \
        rename(columns={v: v.replace('return', 'net_value') for v in cum_portfolio_ret.columns})  # type: pd.DataFrame
    # turnover analysis
    # cum_portfolio_ret = cum_portfolio_ret.fillna(method='ffill').fillna(1)

    return cum_portfolio_ret


def quantile_backtesting(factor_data: pd.DataFrame, factor_name: str, quantiles=None, bins=5,
                         demeaned=True,
                         group_adjust=False
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor_data = factor_data.copy()
    factor_data['factor_quantile'] = quantize_factor(factor_data, factor_name, quantiles, bins, group_adjust)
    mean_ret, std_error_ret, count_stat = mean_return_by_quantile(factor_data, by_date=True, by_group=False,
                                                                  demeaned=demeaned,
                                                                  group_adjust=group_adjust, weight_='equal')

    # quantile cum ret
    mean_ret = mean_ret.unstack(level=0)
    count_stat = count_stat.unstack(level=0)
    std_error_ret = std_error_ret.unstack(level=0)

    # appended = mean_ret.iloc[-1:]
    # appended.index = appended.index + pd.Timedelta(weeks=4)
    # mean_ret = mean_ret.append(appended).shift(1).fillna(0)
    mean_ret.columns = pd.MultiIndex.from_tuples([(c[0], str(c[1])) for c in mean_ret.columns])
    first_quantile = mean_ret.columns[0]
    last_quantile = mean_ret.columns[-1]
    mean_ret[(first_quantile[0], '{}-{}'.format(last_quantile[-1], first_quantile[-1]))] = mean_ret[last_quantile] - \
                                                                                           mean_ret[first_quantile]
    mean_ret[(first_quantile[0], '{}-{}'.format(first_quantile[-1], last_quantile[-1]))] = mean_ret[first_quantile] - \
                                                                                           mean_ret[last_quantile]
    cum_ret = mean_ret.add(1).cumprod()

    return cum_ret, mean_ret, count_stat, std_error_ret


def backtesting_metric(returns, cum_ret=None, periods=12) -> pd.DataFrame:
    if cum_ret is None:
        cum_ret = returns.add(1).cumprod()

    backtesting_result = OrderedDict()
    start = returns.index[0].strftime('%Y-%m-%d')
    end = returns.index[-1].strftime('%Y-%m-%d')
    backtesting_result['start'] = start
    backtesting_result['end'] = end
    backtesting_result[' '] = np.nan
    backtesting_result['Compounded Ann Growth Rate'] = cagr(cum_ret)
    backtesting_result['Annualized Return'] = periods * returns.mean()
    backtesting_result['Ret Vol (annualized)'] = returns_volatility(returns) * periods ** 0.5
    backtesting_result['Ret Skew'] = returns_skew(returns)
    backtesting_result['Ret Kurt'] = returns_kurt(returns)
    backtesting_result['sharpe'] = sharpe_ratio(returns, periods=periods)
    backtesting_result['sortino'] = sortino(returns, periods=periods)
    backtesting_result['  '] = np.nan
    backtesting_result['win rate'] = (returns > 0).sum() / len(returns)
    backtesting_result['Best Month'] = best(returns)
    backtesting_result['Worst Month'] = worst(returns)
    backtesting_result['Best Quarter'] = best(returns, 'quarter')
    backtesting_result['Worst Quarter'] = worst(returns, 'quarter')
    backtesting_result['Best Year'] = best(returns, 'year')
    backtesting_result['Worst Year'] = worst(returns, 'year')
    backtesting_result['   '] = np.nan
    dd, ddp = drawdown(cum_ret)

    backtesting_result['drawdown_value'] = dd.abs().max()
    backtesting_result['drawdown_percent'] = ddp.abs().max()

    # for c in ddp.columns:
    #     backtesting_result['drawdown_detail'] = drawdown_details(ddp)

    return pd.DataFrame(backtesting_result).T


def compund_return(returns):
    """
    :param returns:
    :return:
    """
    return returns.add(1).prod() - 1


def deannualized(annualized_return, nperiods=252):
    """
    :param annualized_return:
    :param nperiods:
    :return:
    """
    deannualized_return = np.power(1 + annualized_return, 1. / nperiods) - 1.
    return deannualized_return


def sharpe_ratio(returns, rf=0., periods=252, annualize=True):
    """
    :param returns:
    :param rf:
    :param periods:
    :param annualize:
    :return:
    """
    if rf != 0 and periods is None:
        raise Exception('Must provide periods if rf != 0')

    rf = deannualized(rf, periods)
    res = (returns.mean() - rf) / returns.std()

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)
    return res


def sortino(returns, rf=0, periods=252, annualize=True):
    """
    https://www.investopedia.com/terms/s/sortinoratio.asp
    """

    if rf != 0 and periods is None:
        raise Exception('Must provide periods if rf != 0')

    downside = (returns[returns < 0] ** 2).sum() / len(returns)
    res = returns.mean() / np.sqrt(downside)

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def cagr(net_value):
    """
    :param net_value:
    :return:
    """
    years = (net_value.index[-1] - net_value.index[0]).days / 365.

    res = abs(net_value.iloc[-1] / net_value.iloc[0]) ** (1.0 / years) - 1

    return res


def calmar(cagr_ratio, max_dd):
    """ calculates the calmar ratio (CAGR% / MaxDD%) """
    return cagr_ratio / abs(max_dd)


def returns_volatility(returns: pd.Series):
    return returns.std()


def returns_skew(returns: pd.Series):
    return returns.skew()


def returns_kurt(returns: pd.Series):
    return returns.kurt()


def calmar(cagr_ratio, max_dd):
    return cagr_ratio / abs(max_dd)


def group_returns(returns, groupby, compounded):
    """
    :param returns:
    :param groupby:
    :param compounded:
    :return:
    """
    if compounded:
        return returns.groupby(groupby).apply(compund_return)
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    """
    Aggregates returns based on date periods
    """

    if period is None:
        return returns

    if 'day' in period:
        return group_returns(returns, pd.Grouper(freq='D'), compounded=compounded)
    elif 'week' in period:
        return group_returns(returns, pd.Grouper(freq='W'), compounded=compounded)
    elif 'month' in period:
        return group_returns(returns, pd.Grouper(freq='M'), compounded=compounded)
    elif 'quarter' in period:
        return group_returns(returns, pd.Grouper(freq='Q'), compounded=compounded)
    elif "year" == period:
        return group_returns(returns, pd.Grouper(freq='Y'), compounded=compounded)
    elif not isinstance(period, str):
        return group_returns(returns, period, compounded)
    return returns


def drawdown(net_value: pd.Series):
    rolling_max = net_value.rolling(min_periods=1, window=len(net_value), center=False).max()
    drawdown = net_value - rolling_max
    drawdown_percent = (net_value / rolling_max) - 1
    return drawdown, drawdown_percent


def remove_outliers(returns, quantile=.95):
    """ returns series of returns without the outliers """
    return returns[returns < returns.quantile(quantile)]


def drawdown_details(drawdown):
    """
    calculates drawdown details, including start/end/valley dates,
    duration, max drawdown and max dd for 99% of the dd period
    for every drawdown period
    """

    def _drawdown_details(drawdown):
        # mark no drawdown
        no_dd = drawdown == 0

        # extract dd start dates
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts].index)

        # extract end dates
        ends = no_dd & (~no_dd).shift(1)
        ends = list(ends[ends].index)

        # no drawdown :)
        if not starts:
            return pd.DataFrame(
                index=[], columns=('start', 'valley', 'end', 'days',
                                   'max drawdown', '99% max drawdown'))

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # build dataframe from results
        data = []
        for i, _ in enumerate(starts):
            dd = drawdown[starts[i]:ends[i]]
            clean_dd = -remove_outliers(-dd, .99)
            data.append((starts[i], dd.idxmin(), ends[i],
                         (ends[i] - starts[i]).days,
                         dd.min(), clean_dd.min()))

        df = pd.DataFrame(data=data,
                          columns=('start', 'valley', 'end', 'days',
                                   'max drawdown',
                                   '99% max drawdown'))
        df['days'] = df['days'].astype(int)
        df['max drawdown'] = df['max drawdown'].astype(float)
        df['99% max drawdown'] = df['99% max drawdown'].astype(float)

        df['start'] = df['start'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['end'] = df['end'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['valley'] = df['valley'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df

    return _drawdown_details(drawdown)


def best(returns, aggregate=None, compounded=True):
    """
    returns the best day/month/week/quarter/year's return
    """
    return aggregate_returns(returns, aggregate, compounded).max()


def worst(returns, aggregate=None, compounded=True):
    """
    returns the worst day/month/week/quarter/year's return
    """
    return aggregate_returns(returns, aggregate, compounded).min()


def get_traded_pnl(traded: pd.DataFrame) -> pd.DataFrame:
    traded_ = traded.copy()  # type: pd.DataFrame
    traded_['cum_pos'] = traded_['dealt_qty'].cumsum()
    traded_['pair_id'] = np.where(traded_['cum_pos'] == 0, traded_.index, np.nan)
    traded_['pair_id'] = traded_['pair_id'].bfill()
    # traded_.set_index('order_time', inplace=True)
    traded_pnl = traded_.groupby('pair_id').agg({'cash_inflow': 'sum', 'order_time': 'last'})
    return traded_pnl


def win_rate(traded_pnl: pd.DataFrame, aggregate=None, compounded=True):
    """ calculates the win ratio for a period """
    return len(traded_pnl[traded_pnl['cash_inflow'] > 0]) / len(traded_pnl)


def avg_win(traded_pnl: pd.DataFrame):
    """
    calculates the average winning
    return/trade return for a period
    """
    return traded_pnl['cash_inflow'][traded_pnl['cash_inflow'] > 0].dropna().mean()


def avg_loss(traded_pnl: pd.DataFrame):
    """
    calculates the average low if
    return/trade return for a period
    """
    return traded_pnl['cash_inflow'][traded_pnl['cash_inflow'] < 0].dropna().mean()


def payoff_ratio(traded_pnl):
    """ measures the payoff ratio (average win/average loss) """
    return avg_win(traded_pnl) / abs(avg_loss(traded_pnl))


def kelly(traded_pnl):
    win_loss_ratio = payoff_ratio(traded_pnl)
    win_prob = win_rate(traded_pnl)
    lose_prob = 1 - win_prob
    return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio


def value_at_risk(returns, sigma=1, confidence=0.95):
    """
    calculates the daily value-at-risk
    (variance-covariance calculation with confidence n)
    """
    mu = returns.mean()
    sigma *= returns.std()

    if confidence > 1:
        confidence = confidence / 100

    return norm.ppf(1 - confidence, mu, sigma)


def information_analysis(cross_sectional_ic):
    """
    :param cross_sectional_ic:
    :return:
    """
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = cross_sectional_ic.mean()
    ic_summary_table["IC Std."] = cross_sectional_ic.std()
    ic_summary_table["IR"] = ir(cross_sectional_ic)
    t_stat, p_value = stats.ttest_1samp(cross_sectional_ic, 0, nan_policy='omit')
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(cross_sectional_ic, nan_policy='omit')
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(cross_sectional_ic, nan_policy='omit')
    ic_summary_table['Cumulative IC'] = cross_sectional_ic.sum()
    ic_summary_table['|ic| >= 2% ratio'] = ic_ability_ratio(cross_sectional_ic)
    ic_summary_table = ic_summary_table.applymap('{:,.5f}'.format)
    ic_summary_table['start'] = cross_sectional_ic.index[0].strftime('%Y-%m-%d')
    ic_summary_table['end'] = cross_sectional_ic.index[-1].strftime('%Y-%m-%d')
    ic_summary_table = ic_summary_table.T
    ic_summary_table.columns = [c.replace('_IC', '') for c in ic_summary_table.columns]

    return ic_summary_table


def factor_ols_regression(factor_data: pd.DataFrame, factor_name: str or list) -> pd.DataFrame:
    """

    :param factor_data:
    :param factor_name:
    :return:
    """
    result_dic = {}
    for col in get_forward_returns_columns(factor_data.columns):
        X = sm.add_constant(factor_data[factor_name].values)  # constant is not added by default
        model = sm.OLS(factor_data[col].values, X, missing='drop')
        result = model.fit()
        d = {}
        d['OLS Beta'] = result.params[1]
        d['OLS t-stat'] = result.tvalues[1]
        d['OLS p-value'] = result.pvalues[1]
        result_dic[col] = d
    result_df = pd.DataFrame(result_dic)
    result_df = result_df.applymap('{:,.5f}'.format)
    return result_df


def factor_quantile_regression(factor_data: pd.DataFrame, factor_name: str or list, q=0.5) -> pd.DataFrame:
    """

    :param factor_data:
    :param factor_name:
    :param q:
    :return:
    """

    result_dic = {}
    for col in get_forward_returns_columns(factor_data.columns):
        # X = sm.add_constant(factor_data[factor_name].values)  # constant is not added by default
        model = smf.quantreg('{} ~ {}'.format(col, factor_name), factor_data, missing='drop')
        result = model.fit(q=q)
        d = {}
        d['Quantile Beta (q={})'.format(q)] = result.params[1]
        d['Quantile t-stat (q={})'.format(q)] = result.tvalues[1]
        d['Quantile p-value (q={})'.format(q)] = result.pvalues[1]
        result_dic[col] = d
    result_df = pd.DataFrame(result_dic)
    result_df = result_df.applymap('{:,.5f}'.format)
    return result_df


def factors_correlation(factors: pd.DataFrame):
    pearson_corr = factors.corr()
    spearman_corr = factors.corr(method='spearman')
    return pearson_corr, spearman_corr


def Newey_West_t_statistics(mean_ret):
    """



    :param mean_ret:
    factor_quantile                     1         2         3         4         5
    date
    2006-01-25                   0.000000  0.000000  0.000000  0.000000  0.000000
    2006-02-28                   0.029503 -0.004634 -0.023483 -0.075111 -0.076978




    :return:

               1_period_equal_return
                                1         2         3         4         5
    mean                 0.000918 -0.002344 -0.003672  0.000774 -0.006067
    t-value              1.510150 -0.764564 -0.722198  0.083202 -0.511974


    """
    ret = mean_ret.iloc[1:]  # type: pd.DataFrame
    result = defaultdict(dict)
    for col in ret.columns:
        ones = np.ones(len(ret))
        val = ret[col]
        reg = sm.OLS(endog=val, exog=ones).fit(cov_type='HAC', cov_kwds={'maxlags': None})
        # reg = smf.ols('col ~ 1', data=ret).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        result[col] = {
            'mean': reg.params.values[0],
            'Newey-West t-value': reg.tvalues.values[0]
        }

    return pd.DataFrame(result)
