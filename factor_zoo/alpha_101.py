import inspect

from factor_zoo.factor_operator.alpha101_operator import *
from factor_zoo.factor_operator.basic_operator import *
from factor_zoo.factor_operator.utils import get_paras_string, PriceVar, IndClass


def alpha_1(market_data: pd.DataFrame, *, var1: int = 20, var2: float = 2.0, var3: int = 5, var4: float = 5,
            price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    :return:
    """
    assert var1 > 0
    assert var2 > 0
    assert var3 > 0
    assert var4 > 0
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _ret = returns(_close)
    _stddev = stddev(_ret, var1)
    v1 = ternary_conditional_operator(_ret < 0, _stddev, _close)
    f = rank(ts_argmax(signedpower(v1, var2), var3) - var4) - 0.5
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha1_{}'.format(s)
    return f


def alpha_2(market_data: pd.DataFrame, *,
            var1: int = 1, var2: int = 6,
            price_var1: PriceVar = PriceVar.adj_close, price_var2: PriceVar = PriceVar.adj_open):
    """
    Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    :param market_data:
    :param var1:
    :param var2:
    :param price_var1:
    :param price_var2:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _volume = market_data['money'].unstack().values
    _close = market_data[price_var1.value].unstack().values
    _open = market_data[price_var2.value].unstack().values

    f = (-1 * correlation(rank(delta(log(_volume), var1)), rank(((_close - _open) / _open)), var2))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha2_{}'.format(s)
    return f


def alpha_3(market_data: pd.DataFrame, *,
            var1: int = 10,
            price_var1: PriceVar = PriceVar.adj_open):
    """
    Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    :param price_var1:
    :param var1:
    :param market:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _open = unstack.values
    _volume = market_data['money'].unstack().values

    f = (-1 * correlation(rank(_open), rank(_volume), var1))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha3_{}'.format(s)
    return f


def alpha_4(market_data: pd.DataFrame, *,
            var1: int = 9,
            price_var1: PriceVar = PriceVar.adj_low):
    """
    Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    :param market_data:
    :param var1:
    :param price_var1:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _low = unstack.values
    f = (-1 * ts_rank(rank(_low), var1))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha4_{}'.format(s)
    return f


def alpha_5(market_data: pd.DataFrame, *,
            var1: int = 10, var2: int = 10,
            price_var1: PriceVar = PriceVar.adj_open,
            price_var2: PriceVar = PriceVar.adj_close):
    """
    Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    Parameters
    ----------
    market_data
    var1
    var2
    price_var1
    price_var2

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data['money'].unstack()

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _open = market_data[price_var1.value].unstack().values
    _close = market_data[price_var2.value].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = (rank((_open - (sum(_vwap, var1) / var2))) * (- 1 * np.abs(rank((_close - _vwap)))))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha5_{}'.format(s)
    return f


def alpha_6(market_data: pd.DataFrame, *,
            var1: int = 10,
            price_var1: PriceVar = PriceVar.adj_open):
    """
    Alpha#6: (-1 * correlation(open, volume, 10))
    :param market_data:
    :param var1:
    :param price_var1:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data['money'].unstack()
    _open = market_data[price_var1.value].unstack().values
    _volume = unstack.values

    f = (-1 * correlation(_open, _volume, var1))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha6_{}'.format(s)
    return f


def alpha_7(market_data: pd.DataFrame, *,
            var1: int = 20, var2: int = 7, var3: int = 60,
            price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
    :param market_data:
    :param var1:
    :param var2:
    :param var3:
    :param price_var1:
    :return:
    """
    # todo problem all -1 why?
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data['money'].unstack()
    _amount = unstack.values
    _close = market_data[price_var1.value].unstack().values
    f = ternary_conditional_operator(adv(_amount, var1) < _amount,
                                     -1 * ts_rank(abs(delta(_close, var2)), var3), -1.0)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha7_{}'.format(s)
    return f


def alpha_8(market_data: pd.DataFrame, *,
            var1: int = 5, var2: int = 10,
            price_var1: PriceVar = PriceVar.adj_open, price_var2: PriceVar = PriceVar.adj_close):
    """
    Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))

    :param market_data:
    :param var1:
    :param var2:
    :param price_var1:
    :param price_var2:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var2.value].unstack()

    _open = unstack.values
    _ret = returns(market_data[price_var1.value].unstack().values)
    f = (-1 * rank(((sum(_open, var1) * sum(_ret, var1)) - delay((sum(_open, var1) * sum(_ret, var1)), var2))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha8_{}'.format(s)
    return f


def alpha_9(market_data: pd.DataFrame, *,
            var1: int = 1, var2: int = 5, var3: int = 1,
            price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    :param market_data:
    :param var1:
    :param var2:
    :param var3:
    :param price_var1:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values

    condition1 = (0 < ts_min(delta(_close, var1), var2))
    condition2 = (ts_max(delta(_close, var1), var2) < 0)
    ans = delta(_close, var3)
    f = ternary_conditional_operator(condition1, ans, ternary_conditional_operator(condition2, ans, -1 * ans))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha9_{}'.format(s)
    return f


def alpha_10(market_data: pd.DataFrame, *,
             var1: int = 1, var2: int = 4, var3: int = 1,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    :param market_data:
    :param var1: 
    :param var2: 
    :param var3: 
    :param price_var1:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values

    condition1 = 0 < ts_min(delta(_close, var1), var2)
    condition2 = (ts_max(delta(_close, var1), var2) < 0)
    ans = delta(_close, var3)
    f = rank(ternary_conditional_operator(condition1, ans, ternary_conditional_operator(condition2, ans, -ans)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha10_{}'.format(s)
    return f


def alpha_11(market_data: pd.DataFrame, *,
             var1: int = 3,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#11:((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    Parameters
    ----------
    market_data
    var1
    price_var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    f = (rank(ts_max((_vwap - _close), var1)) + rank(ts_min((_vwap - _close), var1))) * rank(delta(_amount, 3))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha11_{}'.format(s)
    return f


def alpha_12(market_data: pd.DataFrame, *,
             var1: int = 1,
             price_var1: PriceVar = PriceVar.adj_close):
    """

    Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    :param market_data:
    :param var1: 
    :param price_var1: 
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values

    f = sign(delta(_amount, var1)) * (-1 * delta(_close, var1))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha12_{}'.format(s)
    return f


def alpha_13(market_data: pd.DataFrame, *,
             var1: int = 5,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))

    :param market_data:
    :param var1: 
    :param price_var1: 
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values

    f = (-1 * rank(covariance(rank(_close), rank(_amount), var1)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha13_{}'.format(s)
    return f


def alpha_14(market_data: pd.DataFrame, *,
             var1: int = 3, var2: int = 10,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open):
    """
    Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    Parameters
    ----------
    market_data
    var1
    var2
    price_var1
    price_var2

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    _open = market_data[price_var2.value].unstack().values
    _returns = returns(_close)

    f = ((-1 * rank(delta(_returns, var1))) * correlation(_open, _amount, var2))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha14_{}'.format(s)
    return f


def alpha_15(market_data: pd.DataFrame, *,
             var1: int = 3, var2: int = 3,
             price_var1: PriceVar = PriceVar.adj_high):
    """
    Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    Parameters
    ----------
    market_data
    var1
    var2
    price_var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _high = unstack.values
    _amount = market_data['money'].unstack().values

    f = (-1 * sum(rank(correlation(rank(_high), rank(_amount), var1)), var2))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha15_{}'.format(s)
    return f


def alpha_16(market_data: pd.DataFrame, *,
             var1: int = 5,
             price_var1: PriceVar = PriceVar.adj_high
             ):
    """
    Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    Parameters
    ----------
    market_data
    var1
    price_var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _high = unstack.values
    _amount = market_data['money'].unstack().values

    f = (-1 * rank(covariance(rank(_high), rank(_amount), var1)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha16_{}'.format(s)
    return f


def alpha_17(market_data: pd.DataFrame, *,
             var1: int = 10, var2: int = 1, var3: int = 1, var4: int = 5, var5: int = 20
             , price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    price_var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    adv20 = adv(_amount, var5)
    f = (((-1 * rank(ts_rank(_close, var1))) * rank(delta(delta(_close, var2), var3)))
         * rank(ts_rank((_amount / adv20), var4)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha17_{}'.format(s)
    return f


def alpha_18(market_data: pd.DataFrame, *,
             var1: int = 5, var2: int = 10,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open):
    """
    Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
    Parameters
    ----------
    market_data
    var1
    var2
    price_var1
    price_var2

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _open = market_data[price_var2.value].unstack().values

    f = (-1 * rank(((stddev(abs((_close - _open)), var1) + (_close - _open)) + correlation(_close, _open, var2))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha18_{}'.format(s)
    return f


def alpha_19(market_data: pd.DataFrame, *,
             var1: int = 7, var2: int = 7, var3: int = 250,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _returns = returns(_close)
    f = ((-1 * sign(((_close - delay(_close, var1)) + delta(_close, var2)))) * (1 + rank((1 + sum(_returns, var3)))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha19_{}'.format(s)
    return f


def alpha_20(market_data: pd.DataFrame, *,
             var1: int = 1, var2: int = 1, var3: int = 1,
             price_var1: PriceVar = PriceVar.adj_close, price_var2: PriceVar = PriceVar.adj_open,
             price_var3: PriceVar = PriceVar.adj_high, price_var4: PriceVar = PriceVar.adj_low
             ):
    """
    Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    price_var2
    price_var3
    price_var4

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _open = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _low = market_data[price_var4.value].unstack().values
    _close = unstack.values

    f = (((-1 * rank((_open - delay(_high, var1))))
          * rank((_open - delay(_close, var2))))
         * rank((_open - delay(_low, var3))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha20_{}'.format(s)
    return f


def alpha_21(market_data: pd.DataFrame, *,
             var1: int = 8, var2: int = 2, var3: int = 20,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    # bolliger band
    ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ?
    (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _amount = market_data['money'].unstack().values
    adv20 = adv(_amount, var3)

    _close = unstack.values
    mean_var1 = (sum(_close, var1) / var1)
    mean_var2 = (sum(_close, var2) / var2)
    std_var2 = stddev(_close, var1)

    f = ternary_conditional_operator((mean_var1 + std_var2) < mean_var2, -1,
                                     ternary_conditional_operator(mean_var2 < (mean_var1 - std_var2), 1,
                                                                  ternary_conditional_operator(
                                                                      (_amount / adv20) >= 1, 1, -1)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha21_{}'.format(s)
    return f


def alpha_22(market_data: pd.DataFrame, *,
             var1: int = 5, var2: int = 5, var3: int = 20,
             price_var1: PriceVar = PriceVar.adj_close, price_var2: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    close_ = unstack.values
    _amount = market_data['money'].unstack().values
    _high = market_data[price_var2.value].unstack().values

    f = (-1 * (delta(correlation(_high, _amount, var1), var2) * rank(stddev(close_, var3))))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha22_{}'.format(s)
    return f


def alpha_23(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 2,
             price_var1: PriceVar = PriceVar.adj_high
             ):
    """
    Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    Parameters
    ----------
    market_data
    var1
    var2
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _high = unstack.values
    f = ternary_conditional_operator(sum(_high, var1) / var1 < _high, (-1 * delta(_high, var2)), 0)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha23_{}'.format(s)
    return f


def alpha_24(market_data: pd.DataFrame, *,
             var1: int = 100, var2: int = 100, var3: int = 3, var4: float = 0.05,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    close_ = unstack.values
    mean_var1 = (sum(close_, var1) / var1)

    f = ternary_conditional_operator((delta(mean_var1, var2) / delay(close_, var2)) <= var4,
                                     (-1 * (close_ - ts_min(close_, var2))), (-1 * delta(close_, var3)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha24_{}'.format(s)
    return f


def alpha_25(market_data: pd.DataFrame, *,
             var1: int = 20,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    
    Parameters
    ----------
    market_data
    var1
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _high = market_data[price_var2.value].unstack().values

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    _returns = returns(_close)
    adv20 = adv(_amount, var1)

    f = rank(((((-1 * _returns) * adv20) * _vwap) * (_high - _close)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha25_{}'.format(s)
    return f


def alpha_26(market_data: pd.DataFrame, *,
             var1: int = 5, var2: int = 5, var3: int = 3,
             price_var1: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _high = unstack.values
    _amount = market_data['money'].unstack().values

    f = (-1 * ts_max(correlation(ts_rank(_amount, var1), ts_rank(_high, var1), var2), var3))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha26_{}'.format(s)
    return f


def alpha_27(market_data: pd.DataFrame, *,
             var1: int = 6, var2: int = 2, var3: float = 0.5
             ):
    """
    Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['money'].unstack()
    _amount = unstack.values
    _factor = market_data['factor'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)

    f = ternary_conditional_operator(var3 < rank((sum(correlation(rank(_amount), rank(_vwap), var1), var2) / var2)), -1,
                                     1)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha27_{}'.format(s)
    return f


def alpha_28(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 5,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_high, price_var3: PriceVar = PriceVar.adj_low
             ):
    """
    Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    

    Parameters
    ----------
    market_data
    var1
    var2
    price_var1
    price_var2
    price_var3
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _high = market_data[price_var2.value].unstack().values
    _low = market_data[price_var3.value].unstack().values
    _amount = market_data['money'].unstack().values
    adv20 = adv(_amount, var1)
    f = scale(((correlation(adv20, _low, var2) + ((_high + _low) / 2)) - _close))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha28_{}'.format(s)
    return f


def alpha_29(market_data: pd.DataFrame, *,
             var1: int = 1, var2: int = 5, var3: int = 2,
             var4: int = 1, var5: int = 1, var6: int = 5,
             var7: int = 6, var8: int = 5,
             price_var1: PriceVar = PriceVar.adj_close
             ):
    """
    Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))

    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    var8
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    returns_ = returns(_close)

    f = ts_min(
        product(
            rank(
                rank(
                    scale(
                        log(
                            sum(
                                ts_min(
                                    rank(
                                        rank(
                                            -1 * rank(delta((_close - var1), var2))
                                        )
                                    ), var3
                                ),
                                var4)
                        ))
                )
            ), var5
        ), var6
    ) + ts_rank(delay((-1 * returns_), var7), var8)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha29_{}'.format(s)
    return f


def alpha_30(market_data: pd.DataFrame, *,
             var1: int = 1, var2: int = 5, var3: int = 20,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values

    f = (((1.0 - rank(sign(_close - delay(_close, var1) +
                           sign(delay(_close, var1) - delay(_close, var1 + 1))) +
                      sign(delay(_close, var1 + 1) - delay(_close, var1 + 2))
                      )
           ) * sum(_amount, var2)) / sum(_amount, var3))
    s = get_paras_string(args)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha30_{}'.format(s)
    return f


def alpha_31(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 10, var3: int = 10,
             var4: int = 3, var5: int = 12,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low
             ):
    """
    Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    price_var1
    price_var2

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values

    _amount = market_data['money'].unstack().values
    adv20 = adv(_amount, var1)

    f = rank(
        rank(
            rank(
                decay_linear(
                    (-1 * rank(
                        rank(
                            delta(
                                _close, var2
                            )
                        ))
                     ), var3
                )
            )
        )
    ) + \
        rank(
            (-1 * delta(_close, var4))
        ) + \
        sign(
            scale(
                correlation(adv20, _low, var5)
            )
        )
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha31_{}'.format(s)
    return f


def alpha_32(market_data: pd.DataFrame, *,
             var1: int = 7, var2: int = 20, var3: int = 5, var4: int = 230,
             price_var1: PriceVar = PriceVar.adj_close,
             ):
    """
    Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    price_var1
    """

    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)

    f = scale(((sum(_close, var1) / var1) - _close)) + (var2 * scale(correlation(_vwap, delay(_close, var3), var4)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha32_{}'.format(s)
    return f


def alpha_33(market_data: pd.DataFrame, *,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open
             ):
    """
    Alpha#33: rank((-1 * ((1 - (open / close))^1)))
    

    Parameters
    ----------
    market_data
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _open = market_data[price_var2.value].unstack()

    f = rank((-1 * ((1 - (_open / _close)))))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha33_{}'.format(s)
    return f


def alpha_34(market_data: pd.DataFrame, *,
             var1: int = 2, var2: int = 5, var3: int = 1,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _returns = returns(_close)

    f = rank(((1 - rank((stddev(_returns, var1) / stddev(_returns, var2)
                         )
                        )
               ) + (1 - rank(delta(_close, var3)))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha34_{}'.format(s)
    return f


def alpha_35(market_data: pd.DataFrame, *,
             var1: int = 32, var2: int = 16, var3: int = 32,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_high,
             price_var3: PriceVar = PriceVar.adj_low

             ):
    """
    Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    price_var2
    price_var3
    """

    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _returns = returns(_close)
    _amount = market_data['money'].unstack().values
    _high = market_data[price_var2.value].unstack().values
    _low = market_data[price_var3.value].unstack().values
    f = (ts_rank(_amount, var1) *
         (1 - ts_rank(((_close + _high) - _low), var2))) * \
        (1 - ts_rank(_returns, var3))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha35_{}'.format(s)
    return f


def alpha_36(market_data: pd.DataFrame, *,
             var1: int = 20,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open
             ):
    """
    Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    

    Parameters
    ----------
    market_data
    var1
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _returns = returns(_close)
    _open = market_data[price_var2.value].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values
    adv20 = adv(_amount, var1)
    _vwap = vwap(_volume, _amount, _factor)

    f = (((((2.21 * rank(correlation((_close - _open), delay(_amount, 1), 15)))
            + (0.7 * rank((_open - _close)))) +
           (0.73 * rank(ts_rank(delay((-1 * _returns), 6), 5)))) +
          rank(abs(correlation(_vwap, adv20, 6)))) +
         (0.6 * rank((((sum(_close, 200) / 200) - _open) * (_close - _open)))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha36_{}'.format(s)
    return f


def alpha_37(market_data: pd.DataFrame, *,
             var1: int = 1, var2: int = 200,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open
             ):
    """

    Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    

    Parameters
    ----------
    market_data
    var1
    var2
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _open = market_data[price_var2.value].unstack().values
    f = (rank(correlation(delay((_open - _close), var1), _close, var2)) + rank((_open - _close)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha37_{}'.format(s)
    return f


def alpha_38(market_data: pd.DataFrame, *,
             var1: int = 10,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open
             ):
    """
    Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    

    Parameters
    ----------
    market_data
    var1
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _open = market_data[price_var2.value].unstack().values
    f = ((-1 * rank(ts_rank(_close, var1))) * rank((_close / _open)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha38_{}'.format(s)
    return f


def alpha_39(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 7, var3: int = 9, var4: int = 250,
             price_var1: PriceVar = PriceVar.adj_close,
             ):
    """
    Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _returns = returns(_close)
    _amount = market_data['money'].unstack().values
    adv20 = adv(_amount, var1)
    f = ((-1 * rank((delta(_close, var2) *
                     (1 - rank(decay_linear((_amount / adv20), var3)))))) *
         (1 + rank(sum(_returns, var4))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha39_{}'.format(s)
    return f


def alpha_40(market_data: pd.DataFrame, *,
             var1: int = 10, var2: int = 10,
             price_var1: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    

    Parameters
    ----------
    market_data
    var1
    var2
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _high = unstack.values
    _amount = market_data['money'].unstack().values
    f = ((-1 * rank(stddev(_high, var1))) * correlation(_high, _amount, var2))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha40_{}'.format(s)
    return f


def alpha_41(market_data: pd.DataFrame, *,
             price_var1: PriceVar = PriceVar.adj_high,
             price_var2: PriceVar = PriceVar.adj_low
             ):
    """
    Alpha#41: (((high * low)^0.5) - vwap)
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    high_ = unstack.values
    low_ = market_data[price_var2.value].unstack()
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = (((high_ * low_) ** 0.5) - _vwap)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha41_{}'.format(s)
    return f


def alpha_42(market_data: pd.DataFrame, *, price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    
    Parameters
    ----------
    market_data
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = (rank((_vwap - _close)) / rank((_vwap + _close)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha42_{}'.format(s)
    return f


def alpha_43(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 20, var3: int = 7, var4: int = 8,
             price_var1: PriceVar = PriceVar.adj_close
             ):
    """
    Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    adv20 = adv(_amount, var1)
    f = (ts_rank((_amount / adv20), var2) * ts_rank((-1 * delta(_close, var3)), var4))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha43_{}'.format(s)
    return f


def alpha_44(market_data: pd.DataFrame, *,
             var1: int = 5,
             price_var1: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#44: (-1 * correlation(high, rank(volume), 5))
    
    

    Parameters
    ----------
    market_data
    var1
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    high_ = unstack.values
    _volume = market_data['volume'].unstack().values
    f = (-1 * correlation(high_, rank(_volume), var1))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha44_{}'.format(s)
    return f


def alpha_45(market_data: pd.DataFrame, *,
             var1: int = 5, var2: int = 20, var3: int = 2,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    :param market_data:
    :param var1:
    :param var2:
    :param var3:
    :param price_var1:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values

    f = -1 * (rank(
        sum(delay(_close, var1), var2) / var2
    ) * correlation(_close, _amount, var3)
              ) * rank(
        correlation(sum(_close, var1), sum(_close, var2), var3)
    )

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha45_{}'.format(s)
    return f


def alpha_46(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 10, var3: float = 0.25, var4: float = 0,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    price_var1
    """
    # assert var1 > var2
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    cond = (((delay(_close, var1) - delay(_close, var2)) / (var1 - var2))
            - ((delay(_close, var2) - _close) / var2))
    cond1 = var3 < cond
    cond2 = cond < var4
    f = ternary_conditional_operator(cond1, -1,
                                     ternary_conditional_operator(cond2, 1, (-1 * (_close - delay(_close, 1)))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha46_{}'.format(s)
    return f


def alpha_47(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 5, var3: int = 5,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values

    _high = market_data[price_var2.value].unstack().values

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv20 = adv(_amount, var1)

    f = ((((rank(
        (1 / _close)) * _amount) / adv20)
          * ((_high * rank((_high - _close))) / (sum(_high, var2) / var2)))
         - rank((_vwap - delay(_vwap, var3))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha47_{}'.format(s)
    return f


def alpha_48(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.subindustries,
             var1: int = 1, var2: int = 250, var3: int = 2,
             price_var1: PriceVar = PriceVar.adj_close,
             ):
    """
    Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), ind.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    ind = market_data[ind_class.value].unstack().values
    f = (indneutralize(((correlation(delta(_close, var1),
                                     delta(delay(_close, var1), var1), var2) * delta(_close, 1)) / _close), ind)
         / sum(((delta(_close, var1) / delay(_close, var1)) ** var3), var2))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha48_{}'.format(s)
    return f


def alpha_49(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 10, var3: float = 0.1,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))

    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    """
    # assert var1 > var2
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    cond = (((delay(_close, var1) - delay(_close, var2)) / (var2 - var1))
            - ((delay(_close, var1) - _close) / var1)) < -var3
    f = ternary_conditional_operator(cond, 1, (-1 * (_close - delay(_close, 1))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha49_{}'.format(s)
    return f


def alpha_50(market_data: pd.DataFrame, *,
             var1: int = 5, var2: int = 5,
             ):
    """

    Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    

    Parameters
    ----------
    market_data
    var1
    var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['money'].unstack()

    _factor = market_data['factor'].unstack().values
    _amount = unstack.values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)

    f = (-1 * ts_max(rank(correlation(rank(_amount), rank(_vwap), var1)), var2))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha50_{}'.format(s)
    return f


def alpha_51(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 10, var3: float = 0.05,
             price_var1: PriceVar = PriceVar.adj_close
             ):
    """
    same with 49
    Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values

    cond = (((delay(_close, var1) - delay(_close, var2)) / (var2 - var1))
            - ((delay(_close, var1) - _close) / var1)) < -var3

    f = ternary_conditional_operator(cond, 1, ((-1 * 1) * (_close - delay(_close, 1))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha51_{}'.format(s)
    return f


def alpha_52(market_data: pd.DataFrame, *,
             var1: int = 5, var2: int = 5, var3: int = 240, var4: int = 20,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low

             ):
    """
    Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    
    
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    _low = market_data[price_var2.value].unstack().values

    _returns = returns(_close)

    f = ((((-1 * ts_min(_low, var1)) + delay(ts_min(_low, var1), var2))
          * rank(((sum(_returns, var3) - sum(_returns, var4)) / (var3 - var4))))
         * ts_rank(_amount, 5))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha52_{}'.format(s)
    return f


def alpha_53(market_data: pd.DataFrame, *,
             var1: int = 9,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    
    """

    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values

    f = (-1 * delta((((_close - _low) - (_high - _close)) / (_close - _low + 0.00001)), var1))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha53_{}'.format(s)
    return f


def alpha_54(market_data: pd.DataFrame, *,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high,
             price_var4: PriceVar = PriceVar.adj_open,
             ):
    """
    Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _open = market_data[price_var4.value].unstack().values

    f = ((-1 * ((_low - _close) * (_open ** 5))) / ((_low - _high) * (_close ** 5)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha54_{}'.format(s)
    return f


def alpha_55(market_data: pd.DataFrame, *,
             var1: int = 12, var2: int = 6,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high,

             ):
    """
    Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    
    """

    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _amount = market_data['money'].unstack().values

    f = -1 * correlation(rank(
        (_close - ts_min(_low, var1)) /
        (ts_max(_high, var1) - ts_min(_low, var1))
    ),
        rank(_amount), var2
    )
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha55_{}'.format(s)
    return f


def alpha_56(market_data: pd.DataFrame, *,
             var1: int = 10, var2: int = 2, var3: int = 3,
             price_var1: PriceVar = PriceVar.adj_close,
             cap_var1: str = 'circ_mv',
             ):
    """
    Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    cap_var1
    """
    # todo cap

    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    cap = market_data[cap_var1].unstack().values
    _returns = returns(_close)
    f = (0 - (1 * (rank((sum(_returns, var1) /
                         sum(sum(_returns, var2), var3))
                        ) * rank((_returns * cap)))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha56_{}'.format(s)
    return f


def alpha_57(market_data: pd.DataFrame, *,
             var1: int = 30, var2: int = 2,
             price_var1: PriceVar = PriceVar.adj_close
             ):
    """
    Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    

    Parameters
    ----------
    market_data
    var1
    var2
    price_var1
    """

    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    f = (0 - (1 * ((_close - _vwap) / decay_linear(rank(ts_argmax(_close, var1)), var2))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha57_{}'.format(s)
    return f


def alpha_58(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.sector,
             var1: int = 3, var2: int = 7, var3: int = 5,
             ):
    """

    Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, ind.sector), volume, 3.92795), 7.89291), 5.50322))
    

    Parameters
    ----------
    ind_class
    market_data
    var1
    var2
    var3
    var1
    var2
    var3
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['money'].unstack()
    _factor = market_data['factor'].unstack().values
    _amount = unstack.values
    _volume = market_data['volume'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    ind = market_data[ind_class.value].unstack().values
    f = (-1 * ts_rank(decay_linear(correlation(indneutralize(_vwap, ind),
                                               _amount, var1), var2), var3))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha58_{}'.format(s)
    return f


def alpha_59(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.industries,
             var1: float = 0.728317, var2: int = 4, var3: int = 16, var4: int = 8
             ):
    """
    Alpha#59: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), ind.industry), volume, 4.25197), 16.2289), 8.19648))
    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    """

    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['money'].unstack()
    _factor = market_data['factor'].unstack().values
    _amount = unstack.values
    _volume = market_data['volume'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    ind = market_data[ind_class.value].unstack().values
    f = -1 * ts_rank(
        decay_linear(
            correlation(
                indneutralize(((_vwap * var1) + (_vwap * (1 - var1))), ind),
                _amount, var2
            ), var3
        ), var4
    )
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha59_{}'.format(s)
    return f


def alpha_60(market_data: pd.DataFrame, *,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high,

             ):
    """
    Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
    
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _amount = market_data['money'].unstack().values

    f = (0 - (1 * ((2 * scale(
        rank(((((_close - _low) - (_high - _close)) / (_high - _low)) * _amount)))
                    ) - scale(rank(ts_argmax(_close, 10))))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha60_{}'.format(s)
    return f


def alpha_61(market_data: pd.DataFrame, *,
             var1: int = 180, var2: int = 16, var3: int = 17
             ):
    """
    Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['money'].unstack()
    _factor = market_data['factor'].unstack().values
    _amount = unstack.values
    _volume = market_data['volume'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    adv180 = adv(_amount, var1)
    f = (rank((_vwap - ts_min(_vwap, var2))) < rank(correlation(_vwap, adv180, var3)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha61_{}'.format(s)
    return f


def alpha_62(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 9,
             var3: int = 22,
             price_var1: PriceVar = PriceVar.adj_open,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high,

             ):
    """
    Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['money'].unstack()
    _factor = market_data['factor'].unstack().values
    _amount = unstack.values
    _volume = market_data['volume'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    adv20 = adv(_amount, var1)
    _open = market_data[price_var1.value].unstack().values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values

    f = (rank(correlation(_vwap, sum(adv20, var3), var2))
         < rank(((rank(_open) + rank(_open)) <
                 (rank(((_high + _low) / 2)) + rank(_high))))) * -1
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha62_{}'.format(s)
    return f


def alpha_63(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.industries,
             var1: int = 180, var2: int = 2, var3: int = 8, var4: float = 0.318108,
             var5: int = 37, var6: int = 13, var7: int = 12,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open
             ):
    """
    Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, ind.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)

    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _open = market_data[price_var2.value].unstack().values

    _amount = market_data['money'].unstack().values
    adv180 = adv(_amount, var1)
    _factor = market_data['factor'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    ind = market_data[ind_class.value].unstack().values
    f = ((rank(decay_linear(delta(
        indneutralize(_close, ind), var2
    ), var3)) - rank(decay_linear(correlation(((_vwap * var4) + (_open * (1 - var4))),
                                              sum(adv180, var5), var6), var7))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha63_{}'.format(s)
    return f


def alpha_64(market_data: pd.DataFrame, *,
             var1: int = 120, var2: float = 0.178404, var3: int = 12, var4: int = 16, var5: int = 3,
             price_var1: PriceVar = PriceVar.adj_open,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _open = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values

    _amount = market_data['money'].unstack().values
    adv120 = adv(_amount, var1)
    _factor = market_data['factor'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)

    f = (rank(correlation(sum(((_open * var2) + (_low * (1 - var2))), var3),
                          sum(adv120, var3), var4
                          )
              ) <
         rank(delta(((((_high + _low) / 2) * var2) + (_vwap * (1 - var2))
                     ), var5
                    )
              )
         ) * -1
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha64_{}'.format(s)
    return f


def alpha_65(market_data: pd.DataFrame, *,
             var1: int = 60, var2: float = 0.00817205,
             var3: int = 8, var4: int = 6, var5: int = 13,
             price_var1: PriceVar = PriceVar.adj_open
             ):
    """
    Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    price_var1
    """
    # todo
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _open = unstack.values
    _amount = market_data['money'].unstack().values
    adv60 = adv(_amount, var1)
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)

    f = ((rank(correlation(((_open * var2) + (_vwap * (1 - var2))),
                           sum(adv60, var3), var4))
          < rank((_open - ts_min(_open, var5)))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha65_{}'.format(s)
    return f


def alpha_66(market_data: pd.DataFrame, *,
             var1: int = 3, var2: int = 7, var3: float = 0.96633, var4: int = 11, var5: int = 6,
             price_var1: PriceVar = PriceVar.adj_open,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high
             ):
    """
    Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    price_var1
    price_var2
    price_var3
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _open = unstack.values
    _amount = market_data['money'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values

    _vwap = vwap(_volume, _amount, _factor)

    f = ((rank(decay_linear(delta(_vwap, var1), var2)) +
          ts_rank(decay_linear(((((_low * var3) + (_low * (1 - var3))) - _vwap)
                                / (_open - ((_high + _low) / 2))
                                ), var4
                               ),
                  var5)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha66_{}'.format(s)
    return f


def alpha_67(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.sector,
             ind_class1: IndClass = IndClass.subindustries,
             var1: int = 20, var2: int = 2, var3: int = 6
             , price_var1: PriceVar = PriceVar.adj_high
             ):
    """
    Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, ind.sector), IndNeutralize(adv20, ind.subindustry), 6.02936))) * -1)
    
    

    Parameters
    ----------
    market_data
    ind_class
    ind_class1
    var1
    var2
    var3
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _amount = market_data['money'].unstack().values
    _high = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values
    ind1 = market_data[ind_class.value].unstack().values
    ind2 = market_data[ind_class1.value].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    adv20 = adv(_amount, var1)

    f = ((rank((_high - ts_min(_high, var2))) **
          rank(correlation(indneutralize(_vwap, ind1),
                           indneutralize(adv20, ind2), var3))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha67_{}'.format(s)
    return f


def alpha_68(market_data: pd.DataFrame, *,
             var1: int = 15, var2: int = 8, var3: int = 13, var4: float = 0.518371, var5: int = 1,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high,

             ):
    """
    Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    price_var1
    price_var2
    price_var3
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _amount = market_data['money'].unstack().values
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv15 = adv(_amount, var1)

    f = ((ts_rank(correlation(rank(_high), rank(adv15), var2), var3)
          < rank(delta(((_close * var4) + (_low * (1 - var4))), var5))) * -1)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha68_{}'.format(s)
    return f


def alpha_69(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.industries,
             var1: int = 20, var2: int = 2, var3: int = 4, var4: float = 0.490655, var5: int = 4, var6: int = 9,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, ind.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    var6
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv20 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = ((rank(ts_max(delta(indneutralize(_vwap, ind), var2), var3))
          ** ts_rank(correlation(((_close * var4) + (_vwap * (1 - var4))), adv20, var5), var6)
          ) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha69_{}'.format(s)
    return f


def alpha_70(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.industries,
             var1: int = 50, var2: int = 1, var3: int = 17,
             price_var1: PriceVar = PriceVar.adj_close,
             ):
    """
    Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, ind.industry), adv50, 17.8256), 17.9171)) * -1)

    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv50 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = ((rank(delta(_vwap, var2)) **
          ts_rank(correlation(indneutralize(_close, ind), adv50, var3), 17.9171)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha70_{}'.format(s)
    return f


def alpha_71(market_data: pd.DataFrame, *,
             var1: int = 180, var2: int = 3, var3: int = 12, var4: int = 18,
             var5: int = 4, var6: int = 15, var7: int = 16, var8: int = 4,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open,
             price_var3: PriceVar = PriceVar.adj_low
             ):
    """
    Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    var8
    price_var1
    price_var2
    price_var3
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _open = market_data[price_var2.value].unstack().values
    _low = market_data[price_var3.value].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv180 = adv(_amount, var1)

    f = max(ts_rank(decay_linear(correlation(
        ts_rank(_close, var2), ts_rank(adv180, var3), var4), var5), var6),
        ts_rank(decay_linear((rank(((_low + _open) - (_vwap + _vwap))) ** 2), var7), var8))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha71_{}'.format(s)
    return f


def alpha_72(market_data: pd.DataFrame, *,
             var1: int = 40, var2: int = 8, var3: int = 10, var4: int = 3, var5: int = 18, var6: int = 6, var7: int = 2,
             price_var1: PriceVar = PriceVar.adj_high, price_var2: PriceVar = PriceVar.adj_low

             ):
    """
    Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))


    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _high = unstack.values
    _low = market_data[price_var2.value].unstack().values

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv40 = adv(_amount, var1)

    f = (rank(decay_linear(correlation(((_high + _low) / 2), adv40, var2), var3))
         / rank(decay_linear(correlation(ts_rank(_vwap, var4), ts_rank(_amount, var5), var6), var7))
         )

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha72_{}'.format(s)
    return f


def alpha_73(market_data: pd.DataFrame, *,
             var1: int = 4, var2: int = 2, var3: float = 0.147155, var4: int = 2, var5: int = 3, var6: int = 16,
             price_var1: PriceVar = PriceVar.adj_open, price_var2: PriceVar = PriceVar.adj_low
             ):
    """
    Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _open = unstack.values

    _low = market_data[price_var2.value].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    f = (max(rank(decay_linear(delta(_vwap, var1), var2)),
             ts_rank(decay_linear(((delta(((_open * var3) + (_low * (1 - var3))), var4) /
                                    ((_open * var3) + (_low * (1 - var3)))) * -1), var5), var6)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha73_{}'.format(s)
    return f


def alpha_74(market_data: pd.DataFrame, *,
             var1: int = 30, var2: int = 37, var3: int = 15, var4: float = 0.0261661,
             var5: int = 11,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _high = market_data[price_var2.value].unstack().values

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv30 = adv(_amount, var1)

    f = ((rank(correlation(_close, sum(adv30, var2), var3))
          < rank(correlation(rank(((_high * var4) + (_vwap * (1 - var4)))), rank(_amount), var5))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha74_{}'.format(s)
    return f


def alpha_75(market_data: pd.DataFrame, *,
             var1: int = 50, var2: int = 4, var3: int = 12,
             price_var1: PriceVar = PriceVar.adj_low

             ):
    """

    Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))

    :param market_data:
    :return:
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _low = unstack.values

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv50 = adv(_amount, var1)

    f = (rank(correlation(_vwap, _amount, var2)) < rank(correlation(rank(_low), rank(adv50), var3)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha75_{}'.format(s)
    return f


def alpha_76(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.sector,
             var1: int = 81, var2: int = 1, var3: int = 11, var4: int = 8, var5: int = 19, var6: int = 17,
             var7: int = 19,
             price_var1: PriceVar = PriceVar.adj_low,

             ):
    """
    Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, ind.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    price_var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _low = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv81 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = (max(rank(decay_linear(delta(_vwap, var2), var3)),
             ts_rank(decay_linear(ts_rank(correlation(
                 indneutralize(_low, ind), adv81, var4), var5), var6), var7)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha76_{}'.format(s)
    return f


def alpha_77(market_data: pd.DataFrame, *,
             var1: int = 40, var2: int = 20, var3: int = 3, var4: int = 5,
             price_var1: PriceVar = PriceVar.adj_low,
             price_var2: PriceVar = PriceVar.adj_high,

             ):
    """
    Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    price_var1
    price_var2

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _low = unstack.values
    _high = market_data[price_var2.value].unstack().values

    _amount = market_data['money'].unstack().values
    adv40 = adv(_amount, var1)
    _factor = market_data['factor'].unstack().values
    _volume = market_data['volume'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)

    f = min(rank(decay_linear(((((_high + _low) / 2) + _high) - (_vwap + _high)), var2)),
            rank(decay_linear(correlation(((_high + _low) / 2), adv40, var3), var4)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha77_{}'.format(s)
    return f


def alpha_78(market_data: pd.DataFrame, *,
             var1: int = 40, var2: float = 0.352233, var3: int = 19, var4: int = 6, var5: int = 5,
             price_var1: PriceVar = PriceVar.adj_low
             ):
    """
    Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _low = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv40 = adv(_amount, var1)

    f = (rank(correlation(sum(((_low * var2) + (_vwap * (1 - var2))), var3),
                          sum(adv40, var3), var4)) ** rank(correlation(rank(_vwap), rank(_amount), var5)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha78_{}'.format(s)
    return f


def alpha_79(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.sector,
             var1: int = 150, var2: float = 0.60733, var3: int = 1,
             var4: int = 3, var5: int = 9, var6: int = 14,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open
             ):
    """
    Alpha#79: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), ind.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))

    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    var6
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _open = market_data[price_var2.value].unstack().values

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv150 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = (rank(delta(indneutralize(((_close * var2) + (_open * (1 - var2))), ind), var3))
         < rank(correlation(ts_rank(_vwap, var4), ts_rank(adv150, var5), var6)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha79_{}'.format(s)
    return f


def alpha_80(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.industries,
             var1: int = 10, var2: float = 0.868128, var3: int = 4, var4: int = 5, var5: int = 5,
             price_var1: PriceVar = PriceVar.adj_open,
             price_var2: PriceVar = PriceVar.adj_high

             ):
    """
    Alpha#80: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), ind.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
    
    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _open = unstack.values
    _high = market_data[price_var2.value].unstack().values
    _amount = market_data['money'].unstack().values
    adv10 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = ((rank(sign(delta(indneutralize(((_open * var2) + (_high * (1 - var2))), ind), var3))) **
          ts_rank(correlation(_high, adv10, var4), var5)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha80_{}'.format(s)
    return f


def alpha_81(market_data: pd.DataFrame, *,
             var1: int = 10, var2: int = 49, var3: int = 8, var4: int = 4, var5: int = 14, var6: int = 5,
             ):
    """
    Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)

    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    var6
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['money'].unstack()
    _factor = market_data['factor'].unstack().values
    _amount = unstack.values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv10 = adv(_amount, var1)
    f = ((rank(log(product(rank((rank(correlation(_vwap, sum(adv10, var2), var3)) ** var4)), var5)))
          < rank(correlation(rank(_vwap), rank(_amount), var6))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha81_{}'.format(s)
    return f


def alpha_82(market_data: pd.DataFrame, *, ind_class: IndClass = IndClass.sector,
             var1: int = 1, var2: int = 14, var3: float = 0.634196, var4: int = 17, var5: int = 6, var6: int = 13,
             price_var1: PriceVar = PriceVar.adj_open
             ):
    """
    Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, ind.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    var6
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _open = unstack.values
    _amount = market_data['money'].unstack().values
    ind = market_data[ind_class.value].unstack().values
    f = (min(rank(decay_linear(delta(_open, var1), var2)),
             ts_rank(decay_linear(correlation(indneutralize(_amount, ind),
                                              ((_open * var3) + (_open * (1 - var3))), var4
                                              ), var5), var6)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha82_{}'.format(s)
    return f


def alpha_83(market_data: pd.DataFrame, *,
             var1: int = 5, var2: int = 2, var3: int = 5,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high,

             ):
    """
    Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)

    f = ((rank(delay(((_high - _low) / (sum(_close, var1) / var1)), var2)) *
          rank(rank(_amount))) / (((_high - _low) / (sum(_close, var3) / var3)) / (_vwap - _close)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha83_{}'.format(s)
    return f


def alpha_84(market_data: pd.DataFrame, *,
             var1: int = 15, var2: int = 20, var3: int = 4,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))

    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    f = signedpower(ts_rank((_vwap - ts_max(_vwap, var1)), var2), delta(_close, var3))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha84_{}'.format(s)
    return f


def alpha_85(market_data: pd.DataFrame, *,
             var1: int = 30, var2: float = 0.876703, var3: int = 9, var4: int = 3, var5: int = 10, var6: int = 7,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_high,
             price_var3: PriceVar = PriceVar.adj_low,

             ):
    """
    Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    var6
    price_var1
    price_var2
    price_var3
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var3.value].unstack().values
    _high = market_data[price_var2.value].unstack().values
    _amount = market_data['money'].unstack().values
    adv30 = adv(_amount, var1)

    f = (rank(correlation(((_high * var2) + (_close * (1 - var2))), adv30, var3)) **
         rank(correlation(ts_rank(((_high + _low) / 2), var4), ts_rank(_amount, var5), var6)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha85_{}'.format(s)
    return f


def alpha_86(market_data: pd.DataFrame, *,
             var1: int = 20, var2: int = 14, var3: int = 6, var4: int = 20,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_open,
             ):
    """
    Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    price_var1
    price_var2
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _open = market_data[price_var2.value].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv20 = adv(_amount, var1)

    f = ((ts_rank(correlation(_close, sum(adv20, var2), var3), var4)
          < rank(((_open + _close) - (_vwap + _open)))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha86_{}'.format(s)
    return f


def alpha_87(market_data: pd.DataFrame, *, ind_class: IndClass = IndClass.industries,
             var1: int = 81, var2: float = 0.369701, var3: int = 1, var4: int = 2, var5: int = 13, var6: int = 4,
             var7: int = 14,
             price_var1: PriceVar = PriceVar.adj_close,
             ):
    """
    Alpha#87: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, ind.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    adv81 = adv(_amount, var1)
    _factor = market_data['factor'].unstack().values
    _volume = market_data['volume'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    ind = market_data[ind_class.value].unstack().values
    f = (max(rank(decay_linear(delta(((_close * var2) + (_vwap * (1 - var2))), var3), var4)),
             ts_rank(decay_linear(abs(correlation(indneutralize(adv81, ind), _close, var5)),
                                  var6), var7)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha87_{}'.format(s)
    return f


def alpha_88(market_data: pd.DataFrame, *,
             var1: int = 60, var2: int = 8, var3: int = 8, var4: int = 20, var5: int = 8, var6: int = 6, var7: int = 2,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high,
             price_var4: PriceVar = PriceVar.adj_open

             ):
    """
    Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _open = market_data[price_var4.value].unstack().values

    _amount = market_data['money'].unstack().values
    adv60 = adv(_amount, var1)

    f = min(rank(decay_linear(((rank(_open) + rank(_low)) - (rank(_high) + rank(_close))), var2)),
            ts_rank(decay_linear(correlation(ts_rank(_close, var3),
                                             ts_rank(adv60, var4), var5), var6), var7))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha88_{}'.format(s)
    return f


def alpha_89(market_data: pd.DataFrame, *, ind_class: IndClass = IndClass.industries,
             var1: int = 10, var2: float = 0.967285, var3: int = 6, var4: int = 5, var5: int = 3, var6: int = 3,
             var7: int = 10, var8: int = 15,
             price_var1: PriceVar = PriceVar.adj_low

             ):
    """

    Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, ind.industry), 3.48158), 10.1466), 15.3012))
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _low = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv10 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = (ts_rank(decay_linear(correlation(((_low * var2) + (_low * (1 - var2))), adv10, var3), var4),
                 var5) - ts_rank(decay_linear(delta(indneutralize(_vwap, ind), var6), var7), var8))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha89_{}'.format(s)
    return f


def alpha_90(market_data: pd.DataFrame, *, ind_class: IndClass = IndClass.subindustries,
             var1: int = 40, var2: int = 4, var3: int = 5, var4: int = 3,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low

             ):
    """
    Alpha#90: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, ind.subindustry), low, 5.38375), 3.21856)) * -1)
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _amount = market_data['money'].unstack().values
    adv40 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = ((rank((_close - ts_max(_close, var2))) **
          ts_rank(correlation(indneutralize(adv40, ind), _low, var3), var4)) * -1)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha90_{}'.format(s)
    return f


def alpha_91(market_data: pd.DataFrame, *, ind_class: IndClass = IndClass.industries,
             var1: int = 30, var2: int = 9, var3: int = 16, var4: int = 3, var5: int = 4, var6: int = 4, var7: int = 2,
             price_var1: PriceVar = PriceVar.adj_close
             ):
    """
    Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, ind.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values
    adv30 = adv(_amount, var1)
    _vwap = vwap(_volume, _amount, _factor)
    ind = market_data[ind_class.value].unstack().values
    f = ((ts_rank(decay_linear(decay_linear(
        correlation(indneutralize(_close, ind), _amount, var2), var3), var4), var5)
          - rank(decay_linear(correlation(_vwap, adv30, var6), var7))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha91_{}'.format(s)
    return f


def alpha_92(market_data: pd.DataFrame, *,
             var1: int = 30, var2: int = 14, var3: int = 18,
             var4: int = 7, var5: int = 6, var6: int = 6,
             price_var1: PriceVar = PriceVar.adj_close,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_high,
             price_var4: PriceVar = PriceVar.adj_open

             ):
    """
    Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _open = market_data[price_var4.value].unstack().values
    _amount = market_data['money'].unstack().values
    adv30 = adv(_amount, var1)

    f = min(ts_rank(decay_linear(((((_high + _low) / 2) + _close) < (_low + _open)), var2), var3),
            ts_rank(decay_linear(correlation(rank(_low), rank(adv30), var4), var5), var6))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha92_{}'.format(s)
    return f


def alpha_93(market_data: pd.DataFrame, *, ind_class: IndClass = IndClass.industries,
             var1: int = 81, var2: int = 17, var3: int = 19,
             var4: int = 7, var5: float = 0.524434, var6: int = 2,
             var7: int = 16,
             price_var1: PriceVar = PriceVar.adj_close,
             ):
    """

    Alpha#93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, ind.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
    

    Parameters
    ----------
    market_data
    ind_class
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values

    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv81 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = (ts_rank(decay_linear(correlation(indneutralize(_vwap, ind), adv81, var2), var3), var4)
         / rank(decay_linear(delta(((_close * var5) + (_vwap * (1 - var5))), var6), var7)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha93_{}'.format(s)
    return f


def alpha_94(market_data: pd.DataFrame, *,
             var1: int = 60, var2: int = 11, var3: int = 19, var4: int = 4, var5: int = 18, var6: int = 2
             ):
    """
    Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    var6
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data['money'].unstack()
    _factor = market_data['factor'].unstack().values
    _amount = unstack.values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv60 = adv(_amount, var1)

    f = ((rank((_vwap - ts_min(_vwap, var2))) **
          ts_rank(correlation(ts_rank(_vwap, var3), ts_rank(adv60, var4), var5), var6)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha94_{}'.format(s)
    return f


def alpha_95(market_data: pd.DataFrame, *,
             var1: int = 40, var2: int = 12, var3: int = 19, var4: int = 12, var5: int = 5, var6: int = 11,
             price_var1: PriceVar = PriceVar.adj_high,
             price_var2: PriceVar = PriceVar.adj_low,
             price_var3: PriceVar = PriceVar.adj_open,

             ):
    """
    Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _high = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _open = market_data[price_var3.value].unstack().values

    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values
    adv40 = adv(_amount, var1)

    f = (rank((_open - ts_min(_open, var2))) <
         ts_rank((rank(correlation(sum(((_high + _low) / 2), var3), sum(adv40, var3), var4)) ** var5), var6))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha95_{}'.format(s)
    return f


def alpha_96(market_data: pd.DataFrame, *,
             var1: int = 60, var2: int = 3, var3: int = 4, var5: int = 8, var6: int = 7, var7: int = 4, var8: int = 3,
             var9: int = 12, var10: int = 14, var11: int = 13,
             price_var1: PriceVar = PriceVar.adj_close):
    """
    Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var5
    var6
    var7
    var8
    var9
    var10
    var11
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    adv60 = adv(_amount, var1)
    _factor = market_data['factor'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)

    f = (max(ts_rank(decay_linear(correlation(rank(_vwap), rank(_amount), var2),
                                  var3), var5),
             ts_rank(decay_linear(ts_argmax(correlation(ts_rank(_close, var6),
                                                        ts_rank(adv60, var7), var8), var9), var10), var11)) * -1)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha96_{}'.format(s)
    return f


def alpha_97(market_data: pd.DataFrame, *,
             ind_class: IndClass = IndClass.industries,
             var1: int = 60, var2: float = 0.721001, var3: int = 3, var4: int = 20, var5: int = 7, var6: int = 17,
             var7: int = 4, var8: int = 18, var9: int = 15, var10: int = 6,
             price_var1: PriceVar = PriceVar.adj_low

             ):
    """
    Alpha#97: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), ind.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    """

    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _low = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv60 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values
    f = ((rank(decay_linear(delta(indneutralize(((_low * var2)
                                                 + (_vwap * (1 - var2))), ind), var3), var4))
          - ts_rank(decay_linear(ts_rank(correlation(ts_rank(_low, var5),
                                                     ts_rank(adv60, var6), var7), var8), var9), var10)) * -1)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha97_{}'.format(s)
    return f


def alpha_98(market_data: pd.DataFrame, *,
             var1: int = 5, var2: int = 15, var3: int = 26, var4: int = 4, var5: int = 7, var6: int = 20, var7: int = 8,
             var8: int = 6, var9: int = 8,
             price_var1: PriceVar = PriceVar.adj_open,
             ):
    """
    Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))

    

    Parameters
    ----------
    market_data
    var1
    var2
    var3
    var4
    var5
    var6
    var7
    var8
    var9
    price_var1
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _open = unstack.values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _volume = market_data['volume'].unstack().values

    _vwap = vwap(_volume, _amount, _factor)
    adv5 = adv(_amount, var1)
    adv15 = adv(_amount, var2)

    f = (rank(decay_linear(correlation(_vwap, sum(adv5, var3), var4), var5)) -
         rank(decay_linear(ts_rank(ts_argmin(correlation(rank(_open),
                                                         rank(adv15), var6), var7), var8), var9)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha98_{}'.format(s)
    return f


def alpha_99(market_data: pd.DataFrame, *,
             var1: int = 60, var2: int = 19, var3: int = 8, var4: int = 6,
             price_var1: PriceVar = PriceVar.adj_high, price_var2: PriceVar = PriceVar.adj_low

             ):
    """

    Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
    """
    # todo
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)

    unstack = market_data[price_var1.value].unstack()
    _high = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _amount = market_data['money'].unstack().values
    adv60 = adv(_amount, var1)

    f = ((rank(correlation(sum(((_high + _low) / 2), var2),
                           sum(adv60, var2), var3)) < rank(correlation(_low, _amount, var4))) * -1)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha99_{}'.format(s)
    return f


def alpha_100(market_data: pd.DataFrame, *,
              ind_class: IndClass = IndClass.subindustries,
              var1: int = 20, var2: int = 5, var3: int = 30,
              price_var1: PriceVar = PriceVar.adj_close,
              price_var2: PriceVar = PriceVar.adj_low,
              price_var3: PriceVar = PriceVar.adj_high,
              price_var4: PriceVar = PriceVar.adj_open,

              ):
    """
    Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), ind.subindustry), ind.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), ind.subindustry))) * (volume / adv20))))
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _open = market_data[price_var4.value].unstack().values
    _amount = market_data['money'].unstack().values
    adv20 = adv(_amount, var1)
    ind = market_data[ind_class.value].unstack().values

    f = (0 - (1 * ((1.5 *
                    scale(
                        indneutralize(
                            indneutralize(
                                rank(
                                    ((((_close - _low) - (_high - _close)) / (_high - _low)) * _amount)
                                ), ind
                            ), ind
                        )
                    ) - scale(indneutralize(
                (correlation(_close, rank(adv20), var2)
                 - rank(ts_argmin(_close, var3))
                 ), ind
            )
            )
                    ) * (_amount / adv20)
                   )
              )
         )
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha100_{}'.format(s)
    return f


def alpha_101(market_data: pd.DataFrame, *,
              price_var1: PriceVar = PriceVar.adj_close,
              price_var2: PriceVar = PriceVar.adj_low,
              price_var3: PriceVar = PriceVar.adj_high,
              price_var4: PriceVar = PriceVar.adj_open,

              ):
    """
    Alpha#101: ((close - open) / ((high - low) + .001))
    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data[price_var1.value].unstack()
    _close = unstack.values
    _low = market_data[price_var2.value].unstack().values
    _high = market_data[price_var3.value].unstack().values
    _open = market_data[price_var4.value].unstack().values

    f = ((_close - _open) / ((_high - _low) + .001))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'Alpha101_{}'.format(s)
    return f
