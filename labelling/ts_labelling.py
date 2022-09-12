import pandas as pd
import numpy as np


def triple_barrier_labler(daily_volatility, prices, upper_lower_multipliers=(3,1), t_final=10):
    barriers = pd.DataFrame(columns=['days_passed',
                                     'price', 'vert_barrier',
                                     'top_barrier', 'bottom_barrier'],
                            index=daily_volatility.index)

    for day, vol in daily_volatility.iteritems():
        days_passed = len(daily_volatility.loc[daily_volatility.index[0]: day])
        # set the vertical barrier
        if (days_passed + t_final < len(daily_volatility.index) and t_final != 0):
            vert_barrier = daily_volatility.index[days_passed + t_final]
        else:
            vert_barrier = np.nan
        # set the top barrier
        if upper_lower_multipliers[0] > 0:
            top_barrier = prices.loc[day] + prices.loc[day] * \
                          upper_lower_multipliers[0] * vol
        else:
            # set it to NaNs
            top_barrier = pd.Series(index=prices.index)
        # set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = prices.loc[day] - prices.loc[day] * \
                             upper_lower_multipliers[1] * vol
        else:
            # set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)
        barriers.loc[day, ['days_passed', 'price',
                           'vert_barrier', 'top_barrier', 'bottom_barrier']] = \
            days_passed, prices.loc[day], vert_barrier, top_barrier, bottom_barrier
        return barriers
