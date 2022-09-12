import inspect

from factor_zoo.factor_operator.alpha101_operator import *
from factor_zoo.factor_operator.basic_operator import *
from factor_zoo.factor_operator.utils import get_paras_string, PriceVar


def ht_alpha_1(market_data: pd.DataFrame, *, var1: int = 10,
               price_var1: PriceVar = PriceVar.adj_high,
               price_var2: PriceVar = PriceVar.adj_high):
    """
    Alpha#1: correlation(div(vwap, high), high, 10)
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    high = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _high2 = market_data[price_var2.value].unstack().values
    _vwap = vwap(_volume, _amount, _factor)

    f = correlation(div(_vwap, high), _high2, var1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'ht_alpha1_{}'.format(s)
    return f


def ht_alpha_2(market_data: pd.DataFrame, *,
               var1: int = 20,
               var2: int = 20,
               price_var1: PriceVar = PriceVar.adj_high,
               price_var2: PriceVar = PriceVar.adj_low):
    """
    Alpha#2: ts_sum(rank(correlation(high, low, 20)),20)
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    high = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _open = market_data[price_var1.value].unstack().values
    _close = market_data[price_var2.value].unstack().values
    _vwap = vwap(_volume, _amount, _factor)

    f = sum(rank(correlation(high, _low, var1)), var2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'ht_alpha2_{}'.format(s)
    return f


def ht_alpha_3(market_data: pd.DataFrame, *,
               var1: int = 5):
    """
    Alpha#3: -ts_stddev(volume, 5)
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['money'].unstack()
    _amount = unstack.values
    f = - stddev(_amount, var1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()

    f.name = 'ht_alpha3_{}'.format(s)
    return f


def ht_alpha_4(market_data: pd.DataFrame, *,
               var1: int = 10,
               var2: int = 10,
               price_var1: PriceVar = PriceVar.adj_high):
    """
    Alpha#4: -mul(rank(covariance(high, volume, 10)) , rank(ts_stddev(high, 10)))
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    high = unstack.values
    _amount = market_data['money'].unstack().values
    f = -mul(rank(covariance(high, _amount, var1)), rank(stddev(high, var2)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'ht_alpha4_{}'.format(s)
    return f


def ht_alpha_5(market_data: pd.DataFrame, *,
               var1: int = 5,
               var2: int = 10,
               var3: int = 5,
               price_var1: PriceVar = PriceVar.adj_high):
    """
    Alpha#5: -mul(ts_sum( rank(covariance(high, volume, 5)), 5), rank(ts_stddev(high, 5)))    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    high = unstack.values
    _amount = market_data['money'].unstack().values
    f = -mul(sum(rank(covariance(high, _amount, var1)), var2),
             rank(stddev(high, var3)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'ht_alpha5_{}'.format(s)
    return f


def ht_alpha_6(market_data: pd.DataFrame, *,
               var1: int = 5,
               price_var1: PriceVar = PriceVar.adj_high,
               price_var2: PriceVar = PriceVar.adj_high,
               price_var3: PriceVar = PriceVar.adj_high
               ):
    """
    Alpha#6: ts_sum(div(add(high,low)), close), 5)
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    high = unstack.values
    low = market_data[price_var2.value].unstack().values
    close = market_data[price_var3.value].unstack().values
    _amount = market_data['money'].unstack().values
    f = sum(div(add(high, low), close), var1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'ht_alpha6_{}'.format(s)
    return f
