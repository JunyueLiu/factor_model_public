import inspect

from numpy import sign

from factor_zoo.factor_operator.alpha101_operator import *
from factor_zoo.factor_operator.basic_operator import *
from factor_zoo.factor_operator.gtja_operator import *
from factor_zoo.factor_operator.utils import get_paras_string


def gtja_alpha1(market_data: pd.DataFrame, var1=1, var2=6):
    """
    SAME with alpha 101 alpha2
    Alpha1 (-1 * correlation(rank(delta(LOG(VOLUME), 1)), rank(((_close - _open) / _open)), 6))
    Parameters
    ----------
    market_data
    var1
    var2

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _ret = returns(_close)
    _open = market_data['adj_open'].unstack().values
    _amount = market_data['money'].unstack().values
    f = -1 * correlation(rank(delta(log(_amount), var1)),
                         rank(((_close - _open) / _open)), var2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha1_{}'.format(s)
    return f


def gtja_alpha2(market_data: pd.DataFrame, var1: int = 1):
    """
    Alpha2 (-1 * delta((((_close - _low) - (_high - _close)) / (_high - _low)), 1))
    Parameters
    ----------
    market_data
    var1

    Returns
    -------

    """
    # similar with alpha101 53 but with different parameter
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    f = (-1 * delta((((_close - _low) - (_high - _close)) / (_high - _low)), var1))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha2_{}'.format(s)
    return f


def gtja_alpha3(market_data: pd.DataFrame, var1=6):
    """
    sum((_close=delay(_close,1)?0:_close-(_close>delay(_close,1)?min(_low,delay(_close,1)):max(_high,delay(_close,1)))),6)

    Parameters
    ----------
    market_data

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _low = market_data['adj_low'].unstack().values
    _high = market_data['adj_high'].unstack().values

    f = sum(ternary_conditional_operator(
        _close == delay(_close, 1), 0,
        ternary_conditional_operator(_close - (_close > delay(_close, 1)),
                                     min(_low, delay(_close, 1)),
                                     max(_high, delay(_close, 1))
                                     )), var1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha3_{}'.format(s)
    return f


def gtja_alpha4(market_data: pd.DataFrame, ):
    # todo
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    raise NotImplementedError


def gtja_alpha5(market_data: pd.DataFrame, ):
    """
    (-1 * TsmaX(correlation(ts_rank(VOLUME, 5), ts_rank(_high, 5), 5), 3))
    Parameters
    ----------
    market_data

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _amount = market_data['money'].unstack().values
    f = (-1 * ts_max(correlation(
        ts_rank(_amount, 5), ts_rank(_high, 5), 5), 3))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha5_{}'.format(s)
    return f


def gtja_alpha6(market_data: pd.DataFrame, var1=0.85, var2=4):
    """
    Alpha6 (rank(SIGN(delta((((_open * 0.85) + (_high * 0.15))), 4)))* -1)
    Parameters
    ----------
    market_data
    var1
    var2

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    f = (rank(sign(delta((_open * var1 + _high * (1 - var1)), var2))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha6_{}'.format(s)
    return f


def gtja_alpha7(market_data: pd.DataFrame, ):
    """
    Alpha7 ((rank(max((VWAP - _close), 3)) + rank(min((VWAP - _close), 3))) * rank(delta(VOLUME, 3)))
    Parameters
    ----------
    market_data

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = ((rank(max((_vwap - _close), 3)) +
          rank(min((_vwap - _close), 3))) * rank(delta(_amount, 3)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha7_{}'.format(s)
    return f


def gtja_alpha8(market_data: pd.DataFrame, var1=0.2, var2=4):
    """
    Alpha8 rank(delta(((((_high + _low) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    Parameters
    ----------
    market_data
    var1
    var2

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['money'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = rank(delta(((((_high + _low) / 2) * var1) + (_vwap * (1 - var1))), var2) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha8_{}'.format(s)
    return f


def gtja_alpha9(market_data: pd.DataFrame):
    """
    Alpha9 sma(((_high+_low)/2-(delay(_high,1)+delay(_low,1))/2)*(_high-_low)/VOLUME,7,2)
    Parameters
    ----------
    market_data

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _amount = market_data['money'].unstack().values
    f = sma(((_high + _low) / 2 - (delay(_high, 1) + delay(_low, 1)) / 2) * (_high - _low) / _amount, 7, 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha9_{}'.format(s)
    return f


def gtja_alpha10(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _ret = returns(_close)
    f = rank(max((ternary_conditional_operator(_ret < 0, stddev(_ret, 20), _close) ** 2), 5))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha_10_{}'.format(s)
    return f


def gtja_alpha11(market_data: pd.DataFrame, var1=6):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _amount = market_data['money'].unstack().values
    f = sum(((_close - _low) - (_high - _close)) / (_high - _low) * _amount, var1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha11_{}'.format(s)
    return f


def gtja_alpha12(market_data: pd.DataFrame, var1=10):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)

    f = (rank((_open - (sum(_vwap, var1) / var1)))) * (-1 * (rank(abs((_close - _vwap)))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha12_{}'.format(s)
    return f


def gtja_alpha13(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = (((_high * _low) ** 0.5) - _vwap)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha13_{}'.format(s)
    return f


def gtja_alpha14(market_data: pd.DataFrame, var1=5):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = _close - delay(_close, var1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha14_{}'.format(s)
    return f


def gtja_alpha15(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    f = _open / delay(_close, 1) - 1
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha15_{}'.format(s)
    return f


def gtja_alpha16(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['money'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = (-1 * ts_max(rank(correlation(rank(_amount), rank(_vwap), 5)), 5))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha16_{}'.format(s)
    return f


def gtja_alpha17(market_data: pd.DataFrame, var1=15, var2=5):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = rank((_vwap - max(_vwap, var1))) ** delta(_close, var2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha17_{}'.format(s)
    return f


def gtja_alpha18(market_data: pd.DataFrame, var1=5):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = _close / delay(_close, var1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha18_{}'.format(s)
    return f


def gtja_alpha19(market_data: pd.DataFrame, var1=5):
    """
    Alpha19
(_close<delay(_close,5)?(_close-delay(_close,5))/delay(_close,5):(_close=delay(_close,5)?0:(_close-D
ELAY(_close,5))/_close))
    Parameters
    ----------
    market_data
    var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = ternary_conditional_operator((_close < delay(_close, var1)),
                                     (_close - delay(_close, var1)) / delay(_close, var1),
                                     ternary_conditional_operator(_close == delay(_close, var1), 0,
                                                                  (_close - delay(_close, var1)) / _close))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha19_{}'.format(s)
    return f


def gtja_alpha20(market_data: pd.DataFrame, var1=6):
    """
    Alpha20 (_close-delay(_close,6))/delay(_close,6)*100
    Parameters
    ----------
    market_data
    var1

    Returns
    -------

    """
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = (_close - delay(_close, var1)) / delay(_close, var1) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha20_{}'.format(s)
    return f


def gtja_alpha21(market_data: pd.DataFrame):
    """

    Alpha21 REGBETA(mean(_close,6),SEQUENCE(6))

    Parameters
    ----------
    market_data

    Returns
    -------

    """
    raise NotImplementedError
    # todo
    # args = inspect.getargvalues(inspect.currentframe())
    # s = get_paras_string(args)
    # unstack = market_data['adj_close'].unstack()
    # _open = market_data['adj_open'].unstack().values
    # _high = market_data['adj_high'].unstack().values
    # _low = market_data['adj_low'].unstack().values
    # _volume = market_data['money'].unstack().values
    # _factor = market_data['factor'].unstack().values
    # _amount = market_data['money'].unstack().values
    # _vwap = vwap(_volume, _amount, _factor)
    #
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha12_{}'.format(s)
    # return f


def gtja_alpha22(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    # todo
    raise NotImplementedError
    f = Smean(((_close - mean(_close, 6)) / mean(_close, 6) - delay((_close - mean(_close, 6)) / mean(_close, 6), 3)),
              12, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha12_{}'.format(s)
    return f


def gtja_alpha23(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = sma(ternary_conditional_operator(_close > delay(_close, 1), stddev(_close, 20), 0), 20, 1) / \
        (sma(ternary_conditional_operator(_close > delay(_close, 1), stddev(_close, 20), 0), 20, 1)
         + sma(ternary_conditional_operator(_close <= delay(_close, 1), stddev(_close, 20), 0), 20, 1)) * 100

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha23_{}'.format(s)
    return f


def gtja_alpha24(market_data: pd.DataFrame, var1=5, var2=5, var3=1):
    assert var2 > var3
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = sma(_close - delay(_close, var1), var2, var3)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha24_{}'.format(s)
    return f


def gtja_alpha25(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    _ret = returns(_close)
    f = ((-1 * rank((delta(_close, 7) *
                     (1 - rank(decay_linear((_amount / mean(_amount, 20)), 9)))))) * (1 + rank(sum(_ret, 250))))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha25_{}'.format(s)
    return f


def gtja_alpha26(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = (((sum(_close, 7) / 7) - _close) + (correlation(_vwap, delay(_close, 5), 230)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha26_{}'.format(s)
    return f


def gtja_alpha27(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['money'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    # todo
    raise NotImplementedError
    f = WMA((_close - delay(_close, 3)) / delay(_close, 3) * 100 + (_close - delay(_close, 6)) / delay(_close, 6) * 100,
            12)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha27_{}'.format(s)
    return f


def gtja_alpha28(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    f = 3 * sma((_close - ts_min(_low, 9)) / (ts_max(_high, 9) - ts_min(_low, 9)) * 100, 3, 1) - 2 * sma(
        sma((_close - ts_min(_low, 9)) / (max(_high, 9) - ts_max(_low, 9)) * 100, 3, 1), 3, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha28_{}'.format(s)
    return f


def gtja_alpha29(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _amount = market_data['money'].unstack().values
    f = (_close - delay(_close, 6)) / delay(_close, 6) * _amount
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha29_{}'.format(s)
    return f


def gtja_alpha30(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['money'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    # todo
    raise NotImplementedError
    # f = WMA((REGRESI(_close / delay(_close) - 1, MKT, SMB, HML，60)) ^ 2, 20)
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha12_{}'.format(s)
    # return f


def gtja_alpha31(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = (_close - mean(_close, 12)) / mean(_close, 12) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha12_{}'.format(s)
    return f


def gtja_alpha32(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _amount = market_data['money'].unstack().values
    f = -1 * sum(rank(correlation(rank(_high), rank(_amount), 3)), 3)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha32_{}'.format(s)
    return f


def gtja_alpha33(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _amount = market_data['money'].unstack().values
    _ret = returns(_close)
    f = ((((-1 * ts_min(_low, 5)) +
           delay(ts_min(_low, 5), 5)) * rank(((sum(_ret, 240) - sum(_ret, 20)) / 220))) * ts_rank(_amount, 5))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha33_{}'.format(s)
    return f


def gtja_alpha34(market_data: pd.DataFrame, var1=12):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = mean(_close, var1) / _close
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha34_{}'.format(s)
    return f


def gtja_alpha35(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _amount = market_data['money'].unstack().values
    f = (min(rank(decay_linear(delta(_open, 1), 15)),
             rank(decay_linear(correlation((_amount), ((_open * 0.65) + (_open * 0.35)), 17), 7))) * -1)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha35_{}'.format(s)
    return f


def gtja_alpha36(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = rank(sum(correlation(rank(_amount), rank(_vwap)), 6), 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha36_{}'.format(s)
    return f


def gtja_alpha37(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _ret = returns(_close)
    f = (-1 * rank(((sum(_open, 5) * sum(_ret, 5)) -
                    delay((sum(_open, 5) * sum(_ret, 5)), 10))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha37_{}'.format(s)
    return f


def gtja_alpha38(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ternary_conditional_operator(((sum(_high, 20) / 20) < _high),
                                     (-1 * delta(_high, 2)), 0)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha38_{}'.format(s)
    return f


def gtja_alpha39(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank(decay_linear(
        delta((_close), 2), 8))
          - rank(decay_linear(correlation(((_vwap * 0.3) + (_open * 0.7)),
                                          sum(mean(_amount, 180), 37), 14), 12))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha39_{}'.format(s)
    return f


def gtja_alpha40(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator(_close > delay(_close, 1), _amount, 0), 26) / sum(
        ternary_conditional_operator(_close <= delay(_close, 1), _amount, 0), 26) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha40_{}'.format(s)
    return f


def gtja_alpha41(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(max(delta(_vwap, 3), 5)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha41_{}'.format(s)
    return f


def gtja_alpha42(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    f = ((-1 * rank(stddev(_high, 10))) * correlation(_high, _amount, 10))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha42_{}'.format(s)
    return f


def gtja_alpha43(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator(_close > delay(_close, 1), _amount,
                                         ternary_conditional_operator(_close < delay(_close, 1), -_amount, 0)), 6)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha43_{}'.format(s)
    return f


def gtja_alpha44(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (ts_rank(decay_linear(correlation(_low,
                                          mean(_amount, 10), 7), 6), 4) +
         ts_rank(decay_linear(delta(_vwap, 3), 10), 15))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha44_{}'.format(s)
    return f


def gtja_alpha45(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(delta((_close * 0.6 + _open * 0.4), 1)) *
         rank(correlation(_vwap, mean(_amount, 150), 15)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha45_{}'.format(s)
    return f


def gtja_alpha46(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (mean(_close, 3) + mean(_close, 6) + mean(_close, 12) + mean(_close, 24)) / (4 * _close)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha46_{}'.format(s)
    return f


def gtja_alpha47(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma((ts_max(_high, 6) - _close) / (ts_max(_high, 6) - ts_min(_low, 6)) * 100, 9, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha47_{}'.format(s)
    return f


def gtja_alpha48(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (-1 * ((rank(((sign((_close - delay(_close, 1))) +
                       sign((delay(_close, 1) - delay(_close, 2)))) + sign(
        (delay(_close, 2) - delay(_close, 3)))))) * sum(_amount, 5)) / sum(_amount, 20))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha48_{}'.format(s)
    return f


def gtja_alpha49(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator((_high + _low) >= (delay(_high, 1) + delay(_low, 1)), 0,
                                         max(abs(_high - delay(_high, 1)), abs(_low - delay(_low, 1)))), 12) / \
        (sum(ternary_conditional_operator((_high + _low) >= (delay(_high, 1) + delay(_low, 1)), 0,
                                          max(abs(_high - delay(_high, 1)), abs(_low - delay(_low, 1)))), 12)
         + sum(ternary_conditional_operator((_high + _low) <= (delay(_high, 1) + delay(_low, 1)), 0,
                                            max(abs(_high - delay(_high, 1)),
                                                abs(_low - delay(_low, 1)))), 12))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha49_{}'.format(s)
    return f


def gtja_alpha50(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    # todo
    # f = sum(ternary_conditional_operator((_high + _low) <= (delay(_high, 1) + delay(_low, 1)), 0, max(abs(_high-delay(_high, 1)),
    #     abs(_low - delay(_low, 1)))), 12) / (sum(((_high + _low) <= (delay(_high, 1) + delay(_low, 1))?0:max(abs(_high-delay(_high, 1)), abs(_low-delay(_low, 1)))), 12)+sum(
    #     ((_high + _low) >= (delay(_high, 1) + delay(_low, 1))?0:max(abs(_high-delay(_high, 1)), abs(_low - delay(_low, 1)))), 12))-sum(((_high + _low) >= (delay(_high, 1) + delay(_low, 1))?0:max(abs(HI
    # GH-delay(_high, 1)), abs(_low - delay(_low, 1)))), 12) / (sum(((_high + _low) >= (delay(_high, 1) + delay(_low, 1))?0:
    # max(abs(_high-delay(_high, 1)), abs(_low - delay(_low, 1)))), 12)+sum(((_high + _low) <= (delay(_high, 1) + delay(_low, 1))?0: max(
    #     abs(_high - delay(_high, 1)), abs(_low - delay(_low, 1)))), 12))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha5_{}'.format(s)
    return f


def gtja_alpha51(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    # todo
    #     f = sum(((_high+_low)<=(delay(_high,1)+delay(_low,1))?0:max(abs(_high-delay(_high,1)),abs(_low-delay(L
    # OW,1)))),12)/(sum(((_high+_low)<=(delay(_high,1)+delay(_low,1))?0:max(abs(_high-delay(_high,1)),abs(L
    # OW-delay(_low,1)))),12)+sum(((_high+_low)>=(delay(_high,1)+delay(_low,1))?0:max(abs(_high-delay(HI
    # GH,1)),abs(_low-delay(_low,1)))),12))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha51_{}'.format(s)
    return f


def gtja_alpha52(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(max(0, _high - delay((_high + _low + _close) / 3, 1)), 26) / \
        sum(max(0, delay((_high + _low + _close) / 3, 1) - _low), 26) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha5_{}'.format(s)
    return f


def gtja_alpha53(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    f = COUNT(_close > delay(_close, 1), 12) / 12 * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha5_{}'.format(s)
    return f


def gtja_alpha54(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (-1 * rank((stddev(abs(_close - _open))
                    + (_close - _open)) + correlation(_close, _open, 10)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha54_{}'.format(s)
    return f


def gtja_alpha55(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    # todo
    #     f = sum(16*(_close-delay(_close,1)+(_close-_open)/2+delay(_close,1)-delay(_open,1))/((abs(_high-delay(CL
    # OSE,1))>abs(_low-delay(_close,1)) &
    # abs(_high-delay(_close,1))>abs(_high-delay(_low,1))?abs(_high-delay(_close,1))+abs(_low-delay(CLOS
    # E,1))/2+abs(delay(_close,1)-delay(_open,1))/4:(abs(_low-delay(_close,1))>abs(_high-delay(_low,1)) &
    # abs(_low-delay(_close,1))>abs(_high-delay(_close,1))?abs(_low-delay(_close,1))+abs(_high-delay(CLO
    # SE,1))/2+abs(delay(_close,1)-delay(_open,1))/4:abs(_high-delay(_low,1))+abs(delay(_close,1)-delay(OP
    # EN,1))/4)))*max(abs(_high-delay(_close,1)),abs(_low-delay(_close,1))),20)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha5_{}'.format(s)
    return f


def gtja_alpha56(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank((_open - ts_min(_open, 12))) < rank((rank(correlation(sum(((_high + _low) / 2), 19),
                                                                    sum(mean(_amount, 40), 19), 13)) ** 5)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha56_{}'.format(s)
    return f


def gtja_alpha57(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma((_close - ts_min(_low, 9)) / (ts_max(_high, 9) - ts_min(_low, 9)) * 100, 3, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha57_{}'.format(s)
    return f


def gtja_alpha58(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    f = COUNT(_close > delay(_close, 1), 20) / 20 * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha58_{}'.format(s)
    return f


def gtja_alpha59(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator(_close == delay(_close, 1), 0,
                                         ternary_conditional_operator(_close - (_close > delay(_close, 1),
                                                                                min(_low, delay(_close, 1)),
                                                                                max(_high, delay(_close, 1)))),
                                         20))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha59_{}'.format(s)
    return f


def gtja_alpha60(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(((_close - _low) - (_high - _close)) / (_high - _low) * _amount, 20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha60_{}'.format(s)
    return f


def gtja_alpha61(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (max(rank(decay_linear(delta(_vwap, 1), 12)),
             rank(decay_linear(rank(correlation(_low, mean(_amount, 80), 8)), 17))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha61_{}'.format(s)
    return f


def gtja_alpha62(market_data: pd.DataFrame):
    # similart to 101 alpha6
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (-1 * correlation(_high, rank(_volume), 5))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha62_{}'.format(s)
    return f


def gtja_alpha63(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(max(_close - delay(_close, 1), 0), 6, 1) / sma(abs(_close - delay(_close, 1)), 6, 1) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha63_{}'.format(s)
    return f


def gtja_alpha64(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (max(rank(decay_linear(correlation(rank(_vwap), rank(_amount), 4), 4)),
             rank(decay_linear(max(correlation(rank(_close), rank(mean(_amount, 60)), 4), 13), 14))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha64_{}'.format(s)
    return f


def gtja_alpha65(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = mean(_close, 6) / _close
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha65_{}'.format(s)
    return f


def gtja_alpha66(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_close - mean(_close, 6)) / mean(_close, 6) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha66_{}'.format(s)
    return f


def gtja_alpha67(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(max(_close - delay(_close, 1), 0), 24, 1) / sma(abs(_close - delay(_close, 1)), 24, 1) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha67_{}'.format(s)
    return f


def gtja_alpha68(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(((_high + _low) / 2 - (delay(_high, 1) + delay(_low, 1)) / 2) * (_high - _low) / _amount, 15, 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha68_{}'.format(s)
    return f


def gtja_alpha69(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    DTM = ternary_conditional_operator(_open <= delay(_open, 1), 0, max((_high - _open), (_open - delay(_open, 1))))
    DBM = ternary_conditional_operator(_open >= delay(_open, 1), 0, max((_open - _low), (_open - delay(_open, 1))))
    f = ternary_conditional_operator(sum(DTM, 20) > sum(DBM, 20),
                                     (sum(DTM, 20) - sum(DBM, 20)) / sum(DTM, 20),
                                     ternary_conditional_operator(sum(DTM, 20) == sum(DBM, 20), 0, (
                                             sum(DTM, 20) - sum(DBM, 20)) / sum(DBM, 20)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha69_{}'.format(s)
    return f


def gtja_alpha70(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    f = stddev(_amount, 6)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha70_{}'.format(s)
    return f


def gtja_alpha71(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_close - mean(_close, 24)) / mean(_close, 24) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha71_{}'.format(s)
    return f


def gtja_alpha72(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma((ts_max(_high, 6) - _close) / (ts_max(_high, 6) - ts_min(_low, 6)) * 100, 15, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha72_{}'.format(s)
    return f


def gtja_alpha73(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = ((ts_rank(decay_linear(decay_linear(correlation((_close), _amount, 10), 16), 4), 5) -
          rank(decay_linear(correlation(_vwap, mean(_amount, 30), 4), 3))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha73_{}'.format(s)
    return f


def gtja_alpha74(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(correlation(sum(((_low * 0.35) + (_vwap * 0.65)), 20),
                          sum(mean(_amount, 40), 20), 7)) +
         rank(correlation(rank(_vwap), rank(_amount), 6)))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha74_{}'.format(s)
    return f


def gtja_alpha75(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    f = COUNT(_close > _open & BANCHMARKINDEX_close < BANCHMARKINDEX_open, 50) \
        / COUNT(BANCHMARKINDEX_close < BANCHMARKINDEX_open, 50)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha75_{}'.format(s)
    return f


def gtja_alpha76(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = stddev(abs((_close / delay(_close, 1) - 1)) / _amount, 20) / mean(
        abs((_close / delay(_close, 1) - 1)) / _amount, 20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha76_{}'.format(s)
    return f


def gtja_alpha77(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = min(rank(decay_linear(((((_high + _low) / 2) + _high) -
                               (_vwap + _high)), 20)),
            rank(decay_linear(correlation(((_high + _low) / 2), mean(_amount, 40), 3), 6)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha77_{}'.format(s)
    return f


def gtja_alpha78(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    f = ((_high + _low + _close) / 3 - MA((_high + _low + _close) / 3, 12)) / (
            0.015 * mean(abs(_close - mean((_high + _low + _close) / 3, 12)), 12))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha5_{}'.format(s)
    return f


def gtja_alpha79(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(max(_close - delay(_close, 1), 0), 12, 1) / sma(abs(_close - delay(_close, 1)), 12, 1) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha79_{}'.format(s)
    return f


def gtja_alpha80(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    f = (_amount - delay(_amount, 5)) / delay(_amount, 5) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha80_{}'.format(s)
    return f


def gtja_alpha81(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    f = sma(_amount, 21, 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha81_{}'.format(s)
    return f


def gtja_alpha82(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    f = sma((ts_max(_high, 6) - _close) / (ts_max(_high, 6) - ts_min(_low, 6)) * 100, 20, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha82_{}'.format(s)
    return f


def gtja_alpha83(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _high = market_data['adj_high'].unstack().values
    _amount = market_data['money'].unstack().values
    f = (-1 * rank(covariance(rank(_high), rank(_amount), 5)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha83_{}'.format(s)
    return f


def gtja_alpha84(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    f = sum(ternary_conditional_operator(_close > delay(_close, 1), _amount,
                                         ternary_conditional_operator(
                                             _close < delay(_close, 1), -_amount, 0)), 20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha84_{}'.format(s)
    return f


def gtja_alpha85(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (ts_rank((_amount / mean(_amount, 20)), 20) * ts_rank((-1 * delta(_close, 7)), 8))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha85_{}'.format(s)
    return f


def gtja_alpha86(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (ternary_conditional_operator(0.25 <
                                      (((delay(_close, 20) - delay(_close, 10)) / 10)
                                       - ((delay(_close, 10) - _close) / 10))), (-1 * 1),
         (ternary_conditional_operator((((delay(_close, 20) - delay(_close, 10)) / 10) -
                                        ((delay(_close, 10) - _close) / 10)) < 0), 1,
          ((-1 * 1) * (_close - delay(_close, 1)))))

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha86_{}'.format(s)
    return f


def gtja_alpha87(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank(decay_linear(delta(_vwap, 4), 7)) + ts_rank(decay_linear(((((_low * 0.9) + (_low * 0.1)) - _vwap) /
                                                                         (_open - ((_high + _low) / 2))), 11), 7)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha87_{}'.format(s)
    return f


def gtja_alpha88(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_close - delay(_close, 20)) / delay(_close, 20) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha88_{}'.format(s)
    return f


def gtja_alpha89(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = 2 * (sma(_close, 13, 2) - sma(_close, 27, 2) - sma(sma(_close, 13, 2) - sma(_close, 27, 2), 10, 2))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha89_{}'.format(s)
    return f


def gtja_alpha90(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(correlation(rank(_vwap), rank(_amount), 5)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha90_{}'.format(s)
    return f


def gtja_alpha91(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank((_close - max(_close, 5))) * rank(correlation((mean(_amount, 40)), _low, 5))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha91_{}'.format(s)
    return f


def gtja_alpha92(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (max(rank(decay_linear(delta(((_close * 0.35) + (_vwap * 0.65)), 2), 3)),
             ts_rank(decay_linear(abs(correlation((mean(_amount, 180)), _close, 13)), 5), 15)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha92_{}'.format(s)
    return f


def gtja_alpha93(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator(_open >= delay(_open, 1), 0, max((_open - _low), (_open - delay(_open, 1)))),
            20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha93_{}'.format(s)
    return f


def gtja_alpha94(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator(_close > delay(_close, 1), _amount,
                                         ternary_conditional_operator(_close < delay(_close, 1), -_amount, 0)), 30)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha94_{}'.format(s)
    return f


def gtja_alpha95(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = stddev(_amount, 20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha95_{}'.format(s)
    return f


def gtja_alpha96(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(sma((_close - ts_min(_low, 9)) / (ts_max(_high, 9) - ts_min(_low, 9)) * 100, 3, 1), 3, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha96_{}'.format(s)
    return f


def gtja_alpha97(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    f = stddev(_amount, 10)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha97_{}'.format(s)
    return f


def gtja_alpha98(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    # f = ((((delta((sum(_close, 100) / 100), 100) / delay(_close, 100)) < 0.05) ||
    #       ((delta((sum(_close, 100) / 100), 100) / delay(_close, 100)) == 0.05))
    #     ? (-1 * (_close - TSmin(_close, 100))): (
    #         -1 * delta(_close, 3)))
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha80_{}'.format(s)
    # return f


def gtja_alpha99(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (-1 * rank(covariance(rank(_close), rank(_amount), 5)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha99_{}'.format(s)
    return f


def gtja_alpha100(market_data: pd.DataFrame):
    # same with 95
    # Alpha95 STD(AMOUNT,20)
    # Alpha97 STD(VOLUME,10)
    # Alpha100 STD(VOLUME,20)
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = stddev(_amount, 20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha100_{}'.format(s)
    return f


def gtja_alpha101(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank(correlation(_close, sum(mean(_volume, 30), 37), 15)) <
          rank(correlation(rank(((_high * 0.1) + (_vwap * 0.9))), rank(_amount), 11))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha101_{}'.format(s)
    return f


def gtja_alpha102(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _amount = market_data['money'].unstack().values
    f = sma(max(_amount - delay(_amount, 1), 0), 6, 1) / sma(abs(_amount - delay(_amount, 1)), 6, 1) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha102_{}'.format(s)
    return f


def gtja_alpha103(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    f = ((20 - _lowDAY(_low, 20)) / 20) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha103_{}'.format(s)
    return f


def gtja_alpha104(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (-1 * (delta(correlation(_high, _amount, 5), 5) * rank(stddev(_close, 20))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha104_{}'.format(s)
    return f


def gtja_alpha105(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (-1 * correlation(rank(_open), rank(_amount), 10))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha105_{}'.format(s)
    return f


def gtja_alpha106(market_data: pd.DataFrame):
    # 传统反转因子
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = _close - delay(_close, 20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha106_{}'.format(s)
    return f


def gtja_alpha107(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (((-1 * rank((_open - delay(_high, 1)))) * rank((_open - delay(_close, 1)))) *
         rank((_open - delay(_low, 1))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha107_{}'.format(s)
    return f


def gtja_alpha108(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank((_high - min(_high, 2))) ** rank(correlation(_vwap, (mean(_amount, 120)), 6))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha108_{}'.format(s)
    return f


def gtja_alpha109(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    f = sma(_high - _low, 10, 2) / sma(sma(_high - _low, 10, 2), 10, 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha109_{}'.format(s)
    return f


def gtja_alpha110(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(max(0, _high - delay(_close, 1)), 20) / sum(max(0, delay(_close, 1) - _low), 20) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha101_{}'.format(s)
    return f


def gtja_alpha111(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(_amount * ((_close - _low) - (_high - _close)) /
            (_high - _low), 11, 2) - sma(_amount * ((_close - _low) - (_high - _close)) / (_high - _low), 4, 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha101_{}'.format(s)
    return f


def gtja_alpha112(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (sum(ternary_conditional_operator(_close - delay(_close, 1) > 0, _close - delay(_close, 1), 0), 12)
         - sum(ternary_conditional_operator(_close - delay(_close, 1) < 0, abs(_close - delay(_close, 1)), 0), 12)) / \
        (sum(ternary_conditional_operator(_close - delay(_close, 1) > 0, _close - delay(_close, 1), 0), 12) +
         sum(ternary_conditional_operator(_close - delay(_close, 1) < 0, abs(_close - delay(_close, 1)), 0), 12)) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha112_{}'.format(s)
    return f


def gtja_alpha113(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (-1 * ((rank((sum(delay(_close, 5), 20) / 20)) *
                correlation(_close, _amount, 2)) * rank(correlation(sum(_close, 5), sum(_close, 20), 2))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha113_{}'.format(s)
    return f


def gtja_alpha114(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank(delay(((_high - _low) / (sum(_close, 5) / 5)), 2)) * rank(rank(_amount))) /
         (((_high - _low) / (sum(_close, 5) / 5)) / (_vwap - _close)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha114_{}'.format(s)
    return f


def gtja_alpha115(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(correlation(((_high * 0.9) + (_close * 0.1)), mean(_amount, 30), 10)) **
         rank(correlation(ts_rank(((_high + _low) / 2), 4), ts_rank(_amount, 10), 7)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha115_{}'.format(s)
    return f


def gtja_alpha116(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    # f = REGBETA(_close, SEQUENCE, 20)
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha101_{}'.format(s)
    # return f


def gtja_alpha117(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((ts_rank(_amount, 32) *
          (1 - ts_rank(((_close + _high) - _low), 16))) * (1 - ts_rank(_ret, 32)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha117_{}'.format(s)
    return f


def gtja_alpha118(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(_high - _open, 20) / sum(_open - _low, 20) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha118_{}'.format(s)
    return f


def gtja_alpha119(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(decay_linear(correlation(_vwap, sum(mean(_amount, 5), 26), 5), 7))
         - rank(decay_linear(ts_rank(min(correlation(rank(_open), rank(mean(_amount, 15)), 21), 9), 7), 8)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha119_{}'.format(s)
    return f


def gtja_alpha120(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank((_vwap - _close)) / rank((_vwap + _close)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha120_{}'.format(s)
    return f


def gtja_alpha121(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank((_vwap - min(_vwap, 12))) **
          ts_rank(correlation(ts_rank(_vwap, 20), ts_rank(mean(_amount, 60), 2), 18), 3)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha121_{}'.format(s)
    return f


def gtja_alpha122(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (sma(sma(sma(log(_close), 13, 2), 13, 2), 13, 2) -
         delay(sma(sma(sma(log(_close), 13, 2), 13, 2), 13, 2), 1)) / \
        delay(sma(sma(sma(log(_close), 13, 2), 13, 2), 13, 2), 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha122_{}'.format(s)
    return f


def gtja_alpha123(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank(correlation(sum(((_high + _low) / 2), 20),
                           sum(mean(_amount, 60), 20), 9))
          < rank(correlation(_low, _amount, 6))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha123_{}'.format(s)
    return f


def gtja_alpha124(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_close - _vwap) / decay_linear(rank(ts_max(_close, 30)), 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha124_{}'.format(s)
    return f


def gtja_alpha125(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(decay_linear(correlation(_vwap, mean(_amount, 80), 17), 20)) /
         rank(decay_linear(delta(((_close * 0.5) + (_vwap * 0.5)), 3), 16)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha125_{}'.format(s)
    return f


def gtja_alpha126(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_close + _high + _low) / 3
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha126_{}'.format(s)
    return f


def gtja_alpha127(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (mean((100 * (_close - max(_close, 12)) / (max(_close, 12))) ** 2)) ** (1 / 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha127_{}'.format(s)
    return f


def gtja_alpha128(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    hlc3 = (_high + _low + _close) / 3
    f = 100 - (100 / (1 + sum(ternary_conditional_operator(hlc3 > delay(hlc3, 1), hlc3 * _amount, 0), 14) / sum(
        ternary_conditional_operator(hlc3 < delay(hlc3, 1), hlc3 * _amount, 0), 14)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha128_{}'.format(s)
    return f


def gtja_alpha129(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator(_close - delay(_close, 1) < 0, abs(_close - delay(_close, 1)), 0), 12)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha101_{}'.format(s)
    return f


def gtja_alpha130(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(decay_linear(correlation(((_high + _low) / 2),
                                       mean(_amount, 40), 9), 10)) /
         rank(decay_linear(correlation(rank(_vwap), rank(_amount), 7), 3)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha130_{}'.format(s)
    return f


def gtja_alpha131(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    f = (rank(delta(_vwap, 1)) ** ts_rank(correlation(_close, mean(_amount, 50), 18), 18))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha131_{}'.format(s)
    return f


def gtja_alpha132(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = mean(_amount, 20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha132_{}'.format(s)
    return f


def gtja_alpha133(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    # f = ((20 - _highDAY(_high, 20)) / 20) * 100 - ((20 - _lowDAY(_low, 20)) / 20) * 100
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha101_{}'.format(s)
    # return f


def gtja_alpha134(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_close - delay(_close, 12)) / delay(_close, 12) * _amount
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha134_{}'.format(s)
    return f


def gtja_alpha135(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    f = sma(delay(_close / delay(_close, 20), 1), 20, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha135_{}'.format(s)
    return f


def gtja_alpha136(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((-1 * rank(delta(_ret, 3))) * correlation(_open, _amount, 10))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha136_{}'.format(s)
    return f


def gtja_alpha137(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # todo
    raise NotImplementedError
    # f = 16 * (_close - delay(_close, 1) + (_close - _open) / 2 + delay(_close, 1) - delay(_open, 1)) / (
    #     (abs(_high - delay(_close,
    #                        1)) > abs(_low - delay(_close, 1)) &
    #      abs(_high - delay(_close, 1)) > abs(_high - delay(_low, 1))?abs(_high-delay(_close, 1)) + abs(_low - delay(CLOS
    # E, 1)) / 2 + abs(delay(_close, 1) - delay(_open, 1)) / 4: (
    #         abs(_low - delay(_close, 1)) > abs(_high - delay(_low, 1)) &
    #         abs(_low - delay(_close, 1)) > abs(_high - delay(_close, 1))
    #     ?abs(_low-delay(_close, 1))+abs(_high - delay(CLO
    # SE, 1)) / 2 + abs(delay(_close, 1) - delay(_open, 1)) / 4: abs(_high - delay(_low, 1)) + abs(
    #     delay(_close, 1) - delay(OP
    # EN, 1)) / 4)))*max(abs(_high - delay(_close, 1)), abs(_low - delay(_close, 1)))
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha101_{}'.format(s)
    # return f


def gtja_alpha138(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank(decay_linear(delta((((_low * 0.7) +
                                    (_vwap * 0.3))), 3), 20)) -
          ts_rank(decay_linear(ts_rank(correlation(ts_rank(_low, 8),
                                                   ts_rank(mean(_amount, 60), 17), 5), 19), 16), 7)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha138_{}'.format(s)
    return f


def gtja_alpha139(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = -1 * correlation(_open, _amount, 10)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha139_{}'.format(s)
    return f


def gtja_alpha140(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = min(rank(decay_linear(((rank(_open) + rank(_low)) - (rank(_high) + rank(_close))), 8)),
            ts_rank(decay_linear(correlation(ts_rank(_close, 8), ts_rank(mean(_amount, 60), 20), 8), 7), 3))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha140_{}'.format(s)
    return f


def gtja_alpha141(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(correlation(rank(_high), rank(mean(_amount, 15)), 9)) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha141_{}'.format(s)
    return f


def gtja_alpha142(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (((-1 * rank(ts_rank(_close, 10))) * rank(delta(delta(_close, 1), 1))) * rank(
        ts_rank((_amount / mean(_amount, 20)), 5)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha142_{}'.format(s)
    return f


def gtja_alpha143(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    # f = _close > delay(_close, 1)?(_close - delay(_close, 1)) / \
    #                               delay(_close, 1) * SELF: SELF
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha143_{}'.format(s)
    # return f


def gtja_alpha144(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    # f = sumIF(abs(_close / delay(_close, 1) - 1) / _amount, 20, _close < delay(_close, 1)) / \
    #     COUNT(_close < delay(_close, 1), 20)
    #
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha144_{}'.format(s)
    # return f


def gtja_alpha145(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (mean(_amount, 9) - mean(_amount, 26)) / mean(_amount, 12) * 100
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha145_{}'.format(s)
    return f


def gtja_alpha146(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = mean(
        (_close - delay(_close, 1)) / delay(_close, 1) - sma((_close - delay(_close, 1)) / delay(_close, 1), 61, 2),
        20) * ((_close - delay(_close, 1)) / delay(_close, 1) - sma((_close - delay(_close, 1)) / delay(_close, 1), 61,
                                                                    2)) / \
        sma(((_close - delay(_close, 1)) / delay(_close, 1) - (
                (_close - delay(_close, 1)) / delay(_close, 1) - sma((_close - delay(_close, 1)) / delay(_close, 1),
                                                                     61, 2))) ** 2, 60)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha146_{}'.format(s)
    return f


def gtja_alpha147(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    f = REGBETA(mean(_close, 12), SEQUENCE(12))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha147_{}'.format(s)
    return f


def gtja_alpha148(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((rank(correlation(_open, sum(mean(_amount, 60), 9), 6)) < rank((_open - ts_min(_open, 14)))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha148_{}'.format(s)
    return f


def gtja_alpha149(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    f = REGBETA(FILTER(_close / delay(_close, 1) - 1, BANCHMARKINDEX_close < delay(BANCHMARKINDEX_close, 1)),
                FILTER(BANCHMARKINDEX_close / delay(BANCHMARKINDEX_close, 1) - 1,
                       BANCHMARKINDEX_close < delay(BANCHMARKINDEX_close, 1)), 252)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha149_{}'.format(s)
    return f


def gtja_alpha150(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_close + _high + _low) / 3 * _amount
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha150_{}'.format(s)
    return f


def gtja_alpha151(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(_close - delay(_close, 20), 20, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha151_{}'.format(s)
    return f


def gtja_alpha152(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(mean(delay(sma(delay(_close / delay(_close, 9), 1), 9, 1), 1), 12) - mean(
        delay(sma(delay(_close / delay(_close, 9), 1), 9, 1), 1), 26), 9, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha152_{}'.format(s)
    return f


def gtja_alpha153(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (mean(_close, 3) + mean(_close, 6) + mean(_close, 12) + mean(_close, 24)) / 4
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha153_{}'.format(s)
    return f


def gtja_alpha154(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((_vwap - min(_vwap, 16)) < (correlation(_vwap, mean(_amount, 180), 18)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha154_{}'.format(s)
    return f


def gtja_alpha155(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(_amount, 13, 2) - sma(_amount, 27, 2) - sma(sma(_amount, 13, 2) - sma(_amount, 27, 2), 10, 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha155_{}'.format(s)
    return f


def gtja_alpha156(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (max(rank(decay_linear(delta(_vwap, 5), 3)), rank(
        decay_linear(((delta(((_open * 0.15) + (_low * 0.85)), 2) / ((_open * 0.15) + (_low * 0.85))) * -1), 3))) * -1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha156_{}'.format(s)
    return f


def gtja_alpha157(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    # f = (min(PROD(rank(rank(log(sum(ts_min(rank(rank((-1 * rank(delta((_close - 1), 5))))), 2), 1)))), 1), 5)
    #      + ts_rank(delay((-1 * _ret), 6), 5))
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha157_{}'.format(s)
    # return f


def gtja_alpha158(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((_high - sma(_close, 15, 2)) - (_low - sma(_close, 15, 2))) / _close
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha158_{}'.format(s)
    return f


def gtja_alpha159(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # f = ((_close - sum(min(_low, delay(_close, 1)), 6)) / sum(max(HGIH, delay(_close, 1)) - min(_low, delay(_close, 1)),
    #                                                           6)
    #      * 12 * 24 + (_close - sum(min(_low, delay(_close, 1)), 12)) / sum(
    #             max(HGIH, delay(_close, 1)) - min(_low, delay(_close, 1)), 12) * 6 * 24 + (
    #              _close - sum(min(_low, delay(_close, 1)), 24)) / sum(
    #             max(HGIH, delay(_close, 1)) - min(_low, delay(_close, 1)), 24) * 6 * 24) * 100 / (
    #             6 * 12 + 6 * 24 + 12 * 24)
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha159_{}'.format(s)
    # return f


def gtja_alpha160(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(ternary_conditional_operator(_close <= delay(_close, 1), stddev(_close, 20), 0), 20, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha160_{}'.format(s)
    return f


def gtja_alpha161(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = mean(max(max((_high - _low), abs(delay(_close, 1) - _high)), abs(delay(_close, 1) - _low)), 12)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha161_{}'.format(s)
    return f


def gtja_alpha162(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (sma(max(_close - delay(_close, 1), 0), 12, 1) / sma(abs(_close - delay(_close, 1)), 12, 1) * 100 - min(
        sma(max(_close - delay(_close, 1), 0), 12, 1) / sma(abs(_close - delay(_close, 1)), 12, 1) * 100, 12)) / (
                max(sma(max(_close - delay(_close, 1), 0), 12, 1) / sma(abs(_close - delay(_close, 1)), 12,
                                                                        1) * 100, 12) - min(
            sma(max(_close - delay(_close, 1), 0), 12, 1) / sma(abs(_close - delay(_close, 1)), 12, 1) * 100, 12))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha162_{}'.format(s)
    return f


def gtja_alpha163(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = rank(((((-1 * _ret) * mean(_amount, 20)) * _vwap) * (_high - _close)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha163_{}'.format(s)
    return f


def gtja_alpha164(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma((ternary_conditional_operator((_close > delay(_close, 1)), 1 / (_close - delay(_close, 1)), 1)
             - min(ternary_conditional_operator((_close > delay(_close, 1)), 1 / (_close - delay(_close, 1)), 1),
                   12)) / (_high - _low) * 100,
            13, 2)

    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha164_{}'.format(s)
    return f


def gtja_alpha165(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    raise NotImplementedError
    f = max(sumAC(_close - mean(_close, 48))) - min(sumAC(_close - mean(_close, 48))) / stddev(_close, 48)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha165_{}'.format(s)
    return f


def gtja_alpha166(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = -20 * 19 ** 1.5 * sum(_close / delay(_close, 1) - 1 - mean(_close / delay(_close, 1) - 1, 20), 20) / \
        (19 * 18 * (sum((_close / delay(_close, 1), 20) ** 2, 20)) ** 1.5)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha166_{}'.format(s)
    return f


def gtja_alpha167(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator(_close - delay(_close, 1) > 0, _close - delay(_close, 1), 0), 12)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha167_{}'.format(s)
    return f


def gtja_alpha168(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (-1 * _amount / mean(_amount, 20))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha168_{}'.format(s)
    return f


def gtja_alpha169(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(
        mean(delay(sma(_close - delay(_close, 1), 9, 1), 1), 12) - mean(delay(sma(_close - delay(_close, 1), 9, 1), 1),
                                                                        26), 10, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha169_{}'.format(s)
    return f


def gtja_alpha170(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((((rank((1 / _close)) * _amount) / mean(_amount, 20)) * (
            (_high * rank((_high - _close))) / (sum(_high, 5) / 5))) - rank((_vwap - delay(_vwap, 5))))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha170_{}'.format(s)
    return f


def gtja_alpha171(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((-1 * ((_low - _close) * (_open ** 5))) / ((_close - _high) * (_close ** 5)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha171_{}'.format(s)
    return f


def gtja_alpha172(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # f = mean(abs(sum((LD > 0 & LD > HD)?LD: 0, 14)*100 / sum(TR, 14) - sum((HD > 0 & HD > LD)?HD: 0, 14)*100 / sum(TR,(HD > 0 & HD > LD)?HD: 0, 14)*100 / sum(TR, 14))*100, 6)
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha172_{}'.format(s)
    # return f


def gtja_alpha173(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = 3 * sma(_close, 13, 2) - 2 * sma(sma(_close, 13, 2), 13, 2) + sma(sma(sma(log(_close), 13, 2), 13, 2), 13, 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha173_{}'.format(s)
    return f


def gtja_alpha174(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sma(ternary_conditional_operator(_close > delay(_close, 1), stddev(_close, 20), 0), 20, 1)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha174_{}'.format(s)
    return f


def gtja_alpha175(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = mean(max(max((_high - _low), abs(delay(_close, 1) - _high)), abs(delay(_close, 1) - _low)), 6)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha175_{}'.format(s)
    return f


def gtja_alpha176(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = correlation(rank(((_close - ts_min(_low, 12)) / (ts_max(_high, 12) - ts_min(_low, 12)))), rank(_amount), 6)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha176_{}'.format(s)
    return f


def gtja_alpha177(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # f = ((20 - _highDAY(_high, 20)) / 20) * 100
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha177_{}'.format(s)
    # return f


def gtja_alpha178(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_close - delay(_close, 1)) / delay(_close, 1) * _amount
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha178_{}'.format(s)
    return f


def gtja_alpha179(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = rank(correlation(_vwap, _amount, 4)) * rank(correlation(rank(_low), rank(mean(_amount, 50)), 12))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha179_{}'.format(s)
    return f


def gtja_alpha180(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (ternary_conditional_operator(mean(_amount, 20) < _amount),
         ((-1 * ts_rank(abs(delta(_close, 7)), 60)) * sign(delta(_close, 7)), (-1 * _amount)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha180_{}'.format(s)
    return f


def gtja_alpha181(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # f = sum(((_close / delay(_close, 1) - 1) - mean((_close / delay(_close, 1) - 1), 20)) - (
    #         BANCHMARKINDEX_close - mean(BANCHMARKINDEX_close, 20)) ^ 2, 20) / sum(
    #     (BANCHMARKINDEX_close - mean(BANCHMARKINDEX_close, 20)) ^ 3)
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha181_{}'.format(s)
    # return f


def gtja_alpha182(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # f = COUNT((_close > _open & BANCHMARKINDEX_close > BANCHMARKINDEX_open)
    # OR(_close < _open & BANCHMARKINDEX_close < BANCHMARKINDEX_open), 20) / 20
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha182_{}'.format(s)
    # return f


def gtja_alpha183(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # f = max(sumAC(_close - mean(_close, 24))) - min(sumAC(_close - mean(_close, 24))) / STD(_close, 24)
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha183_{}'.format(s)
    # return f


def gtja_alpha184(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (rank(correlation(delay((_open - _close), 1), _close, 200)) + rank((_open - _close)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha184_{}'.format(s)
    return f


def gtja_alpha185(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = rank((-1 * ((1 - (_open / _close)) ** 2)))
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha185_{}'.format(s)
    return f


def gtja_alpha186(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    # f = (mean(abs(sum((LD > 0 & LD > HD)?LD:0, 14) * 100 / sum(TR, 14) - sum((HD > 0 &
    #                                                                           HD > LD)?HD: 0, 14)*100 / sum(TR, 14)) / (
    #                                                                                                                        sum((
    #                                                                                                                                LD > 0 & LD > HD)
    #                                                                                                                        ?LD:0, 14) * 100 / sum(
    #     TR, 14) + sum((HD > 0 &
    #                    HD > LD)?HD: 0, 14)*100 / sum(TR, 14))*100, 6)+delay(mean(abs(sum((LD > 0 &
    #                                                                                       LD > HD)?LD: 0, 14)*100 / sum(
    #     TR, 14) - sum((HD > 0 & HD > LD)?HD: 0, 14)*100 / sum(TR, 14)) / (sum((LD > 0 &
    #                                                                            LD > HD)?LD:0, 14) * 100 / sum(TR,
    #                                                                                                           14) + sum(
    #     (HD > 0 & HD > LD)?HD: 0, 14)*100 / sum(TR, 14))*100, 6), 6)) / 2
    # f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    # f.name = 'gtja_alpha186_{}'.format(s)
    # return f


def gtja_alpha187(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = sum(ternary_conditional_operator(_open <= delay(_open, 1), 0, max((_high - _open),
                                                                          (_open - delay(_open, 1)))), 20)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha187_{}'.format(s)
    return f


def gtja_alpha188(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = (_high - _low - sma(_high - _low, 11, 2)) / sma(_high - _low, 11, 2)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha187_{}'.format(s)
    return f


def gtja_alpha189(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = mean(abs(_close - mean(_close, 6)), 6)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha189_{}'.format(s)
    return f


def gtja_alpha190(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = mean(abs(_close - mean(_close, 6)), 6)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha190_{}'.format(s)
    return f


def gtja_alpha191(market_data: pd.DataFrame):
    args = inspect.getargvalues(inspect.currentframe())
    s = get_paras_string(args)
    unstack = market_data['adj_close'].unstack()
    _close = unstack.values
    _open = market_data['adj_open'].unstack().values
    _high = market_data['adj_high'].unstack().values
    _low = market_data['adj_low'].unstack().values
    _volume = market_data['volume'].unstack().values
    _factor = market_data['factor'].unstack().values
    _amount = market_data['money'].unstack().values
    _vwap = vwap(_volume, _amount, _factor)
    _ret = returns(_close)
    f = ((correlation(mean(_amount, 20), _low, 5) + ((_high + _low) / 2)) - _close)
    f = pd.DataFrame(f, index=unstack.index, columns=unstack.columns).stack()
    f.name = 'gtja_alpha191_{}'.format(s)
    return f
