import pandas as pd



def trend_quarterly_gross_profits(quarterly_income, num_quarter=8):
    """
    Coefficient from a quarter variable in predicting gross profits
    :return:
    """
    pass






def average_gross_profits(quarterly_income, num_quarter=8):
    """
    Akbas, Ferhat, Chao Jiang, and Paul D. Koch, 2017,
    "The Trend in Firm Profitability and the Cross-Section of Stock Returns", Journal of Accounting and Economics, 92 1-32.

    Average gross profits over the past 8 quarters

    :return:
    """
    total_profit = quarterly_income['total_profit_quarterly']
    average_gross_profits = total_profit.groupby(level=1).rolling(num_quarter).mean().droplevel(0).sort_index()
    average_gross_profits.name = 'average_gross_profits_{}'.format(num_quarter)
    return average_gross_profits


def volatility_gross_profits(quarterly_income, num_quarter=8):
    """
    Akbas, Ferhat, Chao Jiang, and Paul D. Koch, 2017,
    "The Trend in Firm Profitability and the Cross-Section of Stock Returns", Journal of Accounting and Economics, 92 1-32.
    Standard deviation of gross profits over past 8 quarters
    :return:
    """
    total_profit = quarterly_income['total_profit_quarterly']
    volatility_gross_profits = total_profit.groupby(level=1).rolling(num_quarter).std().droplevel(0).sort_index()
    volatility_gross_profits.name = 'volatility_gross_profits_{}'.format(num_quarter)
    return volatility_gross_profits

def volatility_eps(quarterly_income, num_quarter=8):
    """

    :param quarterly_income:
    :param num_quarter:
    :return:
    """
    eps = quarterly_income['eps_quarterly']
    volatility_eps = eps.groupby(level=1).rolling(num_quarter).std().droplevel(0).sort_index()
    volatility_eps.name = 'volatility_eps_{}'.format(num_quarter)
    return volatility_eps

def volatility_total_operating_revenue(quarterly_income, num_quarter=8):
    """

    :param quarterly_income:
    :param num_quarter:
    :return:
    """
    total_operating_revenue_quarterly = quarterly_income['total_operating_revenue_quarterly']
    volatility_total_operating_revenue = total_operating_revenue_quarterly.groupby(level=1).rolling(num_quarter).std().droplevel(0).sort_index()
    volatility_total_operating_revenue.name = 'volatility_total_operating_revenue_{}'.format(num_quarter)
    return volatility_total_operating_revenue

def volatility_operating_profit(quarterly_income, num_quarter=8):
    """

    :param quarterly_income:
    :param num_quarter:
    :return:
    """
    operating_profit = quarterly_income['operating_profit_quarterly']
    volatility_operating_profit = operating_profit.groupby(level=1).rolling(num_quarter).std().droplevel(0).sort_index()
    volatility_operating_profit.name = 'volatility_operating_profit_{}'.format(num_quarter)
    return volatility_operating_profit



