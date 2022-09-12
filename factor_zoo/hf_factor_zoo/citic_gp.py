import pandas as pd

from data_management.dataIO.exotic_data import get_exotic, Exotic
from factor_zoo.factor_operator.citic_operator import cs_add, cs_mul, ts_mean, ts_max, cs_cube, cs_sub, ts_min, cs_div, \
    cs_curt, cs_sqrt, ts_minmaxnorm


def alpha1(config_path: str,
           start_date: str,
           end_date: str,
           *,
           var1: int = 3,
           var2: int = 3) -> pd.Series:
    """
    cs_add(cs_cube(ts_max(closeswapstd,3)),cs_mul(corrpriceswap,ts_mean(closeretavg,3)))
    Parameters
    ----------
    config_path
    var1
    var2

    Returns
    -------

    """
    data = get_exotic(Exotic.citic_hf_basic_operator, cols=['close_swapstd', 'corrpriceswap', 'close_retavg'],
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    closeswapstd = data['close_swapstd'].unstack()
    corrpriceswap = data['corrpriceswap'].unstack()
    closeretavg = data['close_retavg'].unstack()
    idx = closeretavg.index
    cols = closeretavg.columns

    f = cs_add(cs_cube(ts_max(closeswapstd.values, var1)),
               cs_mul(corrpriceswap.values, ts_mean(closeretavg.values, var2)))
    f = pd.DataFrame(f, index=idx, columns=cols).stack()
    f.name = 'alpha1_{}_{}'.format(var1, var2)
    return f


def alpha1_mv(config_path: str,
              start_date: str,
              end_date: str, *,
              var1: int = 3,
              var2: int = 3,
              n: int = 5) -> pd.Series:
    f = alpha1(config_path, start_date, end_date, var1=var1, var2=var2)
    name = f.name
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = '{}_mv_{}'.format(name, n)
    return f


def alpha2(config_path: str, start_date: str,
           end_date: str, *,
           var1: int = 3, var2: int = 3, var3: int = 3):
    """
    cs_add(cs_mul(ts_mean(closeswapstd,3),ts_max(corrpriceswap,3)),cs_sub(closeretavg,ts_min(openretavg,3)))
    Parameters
    ----------
    config_path
    var1
    var2
    var3

    Returns
    -------

    """
    data = get_exotic(Exotic.citic_hf_basic_operator,
                      cols=['close_swapstd', 'corrpriceswap', 'close_retavg', 'open_retavg'],
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    closeswapstd = data['close_swapstd'].unstack()
    corrpriceswap = data['corrpriceswap'].unstack()
    closeretavg = data['close_retavg'].unstack()
    openretavg = data['open_retavg'].unstack()
    idx = closeretavg.index
    cols = closeretavg.columns

    f = cs_add(cs_mul(ts_mean(closeswapstd.values, var1),
                      ts_max(corrpriceswap.values, var2)),
               cs_sub(closeretavg.values, ts_min(openretavg.values, var3)))
    f = pd.DataFrame(f, index=idx, columns=cols).stack()
    f.name = 'alpha2_{}_{}_{}'.format(var1, var2, var3)
    return f


def alpha2_mv(config_path: str,
              start_date: str,
              end_date: str, *, var1: int = 3, var2: int = 3, var3: int = 3, n: int = 5) -> pd.Series:
    f = alpha2(config_path, start_date, end_date, var1=var1, var2=var2, var3=var3)
    name = f.name
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = '{}_mv_{}'.format(name, n)
    return f


def alpha3(config_path: str,
           start_date: str,
           end_date: str, *,
           var1: int = 3,
           var2: int = 3,
           var3: int = 5):
    """
    cs_mul(cs_mul(ts_mean(closeretavg,3),ts_max(closeretavg,3)),cs_cube(ts_mean(swapstd,5)))
    Parameters
    ----------
    config_path
    var1
    var2
    var3

    Returns
    -------

    """
    data = get_exotic(Exotic.citic_hf_basic_operator, cols=['close_retavg', 'swapstd'],
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    closeretavg = data['close_retavg'].unstack()
    swapstd = data['swapstd'].unstack()
    idx = closeretavg.index
    cols = closeretavg.columns

    f = cs_mul(cs_mul(ts_mean(closeretavg.values, var1), ts_max(closeretavg.values, var2)),
               cs_cube(ts_mean(swapstd.values, var3)))
    f = pd.DataFrame(f, index=idx, columns=cols).stack()
    f.name = 'alpha3_{}_{}_{}'.format(var1, var2, var3)
    return f


def alpha3_mv(config_path: str,
              start_date: str,
              end_date: str, *,
              var1: int = 3,
              var2: int = 3,
              var3: int = 5,
              n: int = 5) -> pd.Series:
    f = alpha3(config_path, start_date, end_date, var1=var1, var2=var2, var3=var3)
    name = f.name
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = '{}_mv_{}'.format(name, n)
    return f


def alpha4(config_path: str,
           start_date: str,
           end_date: str, *,
           var1: int = 3,
           var2: int = 3,
           var3: int = 3):
    """
    cs_div(cs_mul(ts_max(closeswapstd,3),ts_mean(closeretavg,3)),cs_curt(ts_mean(openpriceavg,3)))
    Parameters
    ----------
    config_path
    var1
    var2
    var3

    Returns
    -------

    """
    data = get_exotic(Exotic.citic_hf_basic_operator,
                      cols=['close_swapstd', 'close_retavg', 'open_retavg', 'open_priceavg'],
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    closeswapstd = data['close_swapstd'].unstack()
    closeretavg = data['close_retavg'].unstack()
    openpriceavg = data['open_priceavg'].unstack()
    idx = closeretavg.index
    cols = closeretavg.columns

    f = cs_div(cs_mul(ts_max(closeswapstd.values, var1),
                      ts_mean(closeretavg.values, var2)
                      ),
               cs_curt(ts_mean(
                   openpriceavg.values, var3)
               )
               )
    f = pd.DataFrame(f, index=idx, columns=cols).stack()
    f.name = 'alpha4_{}_{}_{}'.format(var1, var2, var3)
    return f


def alpha4_mv(config_path: str,
              start_date: str,
              end_date: str, *,
              var1: int = 3,
              var2: int = 3,
              var3: int = 3,
              n: int = 5) -> pd.Series:
    f = alpha4(config_path, start_date, end_date, var1=var1, var2=var2, var3=var3)
    name = f.name
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = '{}_mv_{}'.format(name, n)
    return f


def alpha5(config_path: str,
           start_date: str,
           end_date: str, *,
           var1: int = 3,
           var2: int = 3):
    """
    cs_cube(cs_add(cs_cube(ts_mean(highretstd, 3)), cs_add(closeretavg, ts_max(closeswapstd, 3))))
    Parameters
    ----------
    config_path
    var1
    var2

    Returns
    -------

    """
    data = get_exotic(Exotic.citic_hf_basic_operator,
                      cols=['close_swapstd', 'close_retavg', 'high_retstd'],
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    closeswapstd = data['close_swapstd'].unstack()
    closeretavg = data['close_retavg'].unstack()
    highretstd = data['high_retstd'].unstack()
    idx = closeretavg.index
    cols = closeretavg.columns

    f = cs_cube(
        cs_add(cs_cube(ts_mean(highretstd.values, var1)), cs_add(closeretavg.values, ts_max(closeswapstd.values, var2))))
    f = pd.DataFrame(f, index=idx, columns=cols).stack()
    f.name = 'alpha5_{}_{}'.format(var1, var2)
    return f


def alpha5_mv(config_path: str,
              start_date: str,
              end_date: str, *,
              var1: int = 3,
              var2: int = 3,
              n: int = 5) -> pd.Series:
    f = alpha5(config_path, start_date, end_date, var1=var1, var2=var2)
    name = f.name
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = '{}_mv_{}'.format(name, n)
    return f


def alpha6(config_path: str,
           start_date: str,
           end_date: str, *,
           var1: int = 3,
           var2: int = 3,
           var3: int = 3):
    """
    cs_div(cs_mul(ts_max(closeswapstd,3),ts_mean(retavg,3)),cs_sqrt(ts_min(openretavg,3)))
    Parameters
    ----------
    config_path
    var1
    var2
    var3

    Returns
    -------

    """
    data = get_exotic(Exotic.citic_hf_basic_operator,
                      cols=['close_swapstd', 'close_retavg', 'open_retavg', 'retavg'],
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    closeswapstd = data['close_swapstd'].unstack()
    closeretavg = data['close_retavg'].unstack()
    openretavg = data['open_retavg'].unstack()
    retavg = data['retavg'].unstack()
    idx = closeretavg.index
    cols = closeretavg.columns

    f = cs_div(cs_mul(ts_max(closeswapstd.values, var1),
                      ts_mean(retavg.values, var2)), cs_sqrt(ts_min(openretavg.values, var3)))
    f = pd.DataFrame(f, index=idx, columns=cols).stack()
    f.name = 'alpha6_{}_{}_{}'.format(var1, var2, var3)
    return f


def alpha6_mv(config_path: str,
              start_date: str,
              end_date: str, *,
              var1: int = 3,
              var2: int = 3,
              var3: int = 3,
              n: int = 5) -> pd.Series:
    f = alpha6(config_path, start_date, end_date, var1=var1, var2=var2, var3=var3)
    name = f.name
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = '{}_mv_{}'.format(name, n)
    return f


def alpha7(config_path: str,
           start_date: str,
           end_date: str, *,
           var1: int = 3,
           var2: int = 3):
    """
    cs_add(cs_cube(ts_max(highswapstd,3)), cs_curt(ts_minmaxnorm(closeretavg,10)))
    Parameters
    ----------
    config_path
    var1
    var2

    Returns
    -------

    """
    data = get_exotic(Exotic.citic_hf_basic_operator,
                      cols=['high_swapstd', 'close_retavg'],
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    highswapstd = data['high_swapstd'].unstack()
    closeretavg = data['close_retavg'].unstack()
    idx = closeretavg.index
    cols = closeretavg.columns

    f = cs_add(cs_cube(ts_max(highswapstd.values, var1)), cs_curt(ts_minmaxnorm(closeretavg.values, var2)))
    f = pd.DataFrame(f, index=idx, columns=cols).stack()
    f.name = 'alpha7_{}_{}'.format(var1, var2)
    return f


def alpha7_mv(config_path: str, start_date: str,
              end_date: str, *, var1: int=3, var2: int =3, n: int = 5) -> pd.Series:
    f = alpha7(config_path, start_date, end_date, var1=var1, var2=var2)
    name = f.name
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = '{}_mv_{}'.format(name, n)
    return f


def alpha8(config_path: str, start_date: str,
           end_date: str, *,
           var1: int = 3,
           var2: int = 3,
           var3: int = 10):
    """
    cs_mul(cs_mul(ts_max(highswapavg,3), ts_max(closeswapstd,3)), cs_mul(highswapstd,ts_minmaxnorm(closeretavg,10)))
    Parameters
    ----------
    config_path
    var1
    var2

    Returns
    -------

    """
    data = get_exotic(Exotic.citic_hf_basic_operator,
                      cols=['high_swapstd', 'close_retavg', 'high_swapavg', 'close_swapstd'],
                      pub_start_date=start_date,
                      pub_end_date=end_date,
                      config_path=config_path, verbose=0)
    highswapstd = data['high_swapstd'].unstack()
    closeretavg = data['close_retavg'].unstack()
    highswapavg = data['high_swapavg'].unstack()
    closeswapstd = data['close_swapstd'].unstack()
    idx = closeretavg.index
    cols = closeretavg.columns

    f = cs_mul(cs_mul(ts_max(highswapavg, var1), ts_max(closeswapstd, var2)),
               cs_mul(highswapstd, ts_minmaxnorm(closeretavg, var3)))
    f = pd.DataFrame(f, index=idx, columns=cols).stack()
    f.name = 'alpha8_{}_{}_{}'.format(var1, var2, var3)
    return f


def alpha8_mv(config_path: str, start_date: str,
              end_date: str, *, var1: int = 3, var2: int = 3, var3: int = 10, n: int = 5) -> pd.Series:
    f = alpha8(config_path, start_date, end_date, var1=var1, var2=var2, var3=var3)
    name = f.name
    f = f.groupby(level=1).rolling(n).mean().droplevel(0).sort_index()
    f.name = '{}_mv_{}'.format(name, n)
    return f


if __name__ == '__main__':
    config_path = '../../cfg/data_input.ini'
    # alpha1(config_path)
    # alpha2(config_path)
    f = alpha8(config_path, '2010-01-01', '2016-01-01')
