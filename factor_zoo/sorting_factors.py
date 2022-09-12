from itertools import product

import pandas as pd


def uni_sort_portfolio(data: pd.DataFrame, sort_name='capital',
                       sort_quantile=(0.5, 0.5),
                       breakpoint: None or tuple = None,
                       weight: str = 'cap'):
    sort_data = data.loc[data.index[0][0], :]
    i = 1
    cum = 0
    groups = []
    if breakpoint is None:
        for q in sort_quantile:
            first_str = sort_name + '_' + str(i)
            sort_data[first_str] = (sort_data[sort_name] >= sort_data[sort_name].quantile(cum)) & \
                                   (sort_data[sort_name] < sort_data[sort_name].quantile(q + cum))
            cum += q
            i += 1
            groups.append(first_str)
    else:
        for bp in breakpoint:
            first_str = sort_name + '_' + str(i)
            sort_data[first_str] = (sort_data[sort_name] >= cum & (sort_data[sort_name] < (bp + cum)))
            cum += bp
            i += 1
            groups.append(first_str)

    sort_data['group'] = ''
    sort_data['weight'] = 0

    for group in groups:
        if weight == 'cap':
            sort_data[group] = sort_data['capital'][sort_data[group]]
            sort_data[group].fillna(0, inplace=True)
            sort_data['weight'][sort_data[group] > 0] = sort_data[group] / sort_data[group].sum()
        elif weight == 'equal':
            # sort_data[group] = sort_data[s1] & sort_data[s2]
            sort_data['weight'][sort_data[group]] = sort_data[group] / sort_data[group].sum()
        sort_data['group'][sort_data[group] > 0] = group

    sort_data = sort_data[['group', 'weight']]
    data = data['ret'].to_frame().join(sort_data)

    def cal_portfolio_ret(cross_section_data):
        return (cross_section_data['ret'] * cross_section_data['weight']).sum()

    grouper = [data.index.get_level_values(level=0), 'group']
    portfolio_ret = data.groupby(grouper).apply(lambda x: cal_portfolio_ret(x))
    portfolio_ret.name = 'ret'
    portfolio_ret = portfolio_ret.reset_index(level=1)

    def cal_hedge_portfolio_ret(port_ret):
        first_long = [g for g in port_ret['group'] if g.startswith(groups[0])]
        first_short = [g for g in port_ret['group'] if g.startswith(groups[-1])]

        portfolio1 = (1 / len(first_long)) * (port_ret[port_ret['group'].isin(first_long)]['ret'].sum()) \
                     - (1 / len(first_short)) * (port_ret[port_ret['group'].isin(first_short)]['ret'].sum())

        return pd.Series([portfolio1], index=['portfolio_1'])

    hedged_portfolio_ret = portfolio_ret.groupby(level=0).apply(lambda x: cal_hedge_portfolio_ret(x))
    return hedged_portfolio_ret


def double_sort_portfolio(data: pd.DataFrame, first_sort='capital',
                          first_quantile=(0.5, 0.5),
                          first_breakpoint: None or tuple = None,
                          second_sort='book_to_price_ratio',
                          second_quantile=(0.3, 0.4, 0.3),
                          second_breakpoint: None or tuple = None,
                          independent_sort: bool = True,
                          weight: str = 'cap'):
    sort_data = data.loc[data.index[0][0], :]

    i = 1
    cum = 0
    first_col = []
    if first_breakpoint is None:
        for q in first_quantile:
            first_str = first_sort + '_' + str(i)

            sort_data[first_str] = (sort_data[first_sort] >= sort_data[first_sort].quantile(cum)) & \
                                   (sort_data[first_sort] < sort_data[first_sort].quantile(q + cum))
            cum += q
            i += 1
            first_col.append(first_str)
    else:
        for bp in first_breakpoint:
            first_str = first_sort + '_' + str(i)
            sort_data[first_str] = (sort_data[first_sort] >= bp) & \
                                   (sort_data[first_sort] < (bp + cum))
            cum += bp
            i += 1
            first_col.append(first_str)

    if independent_sort:
        j = 1
        cum = 0
        second_col = []
        if second_breakpoint is None:
            for q in second_quantile:
                second_str = second_sort + '_' + str(j)
                sort_data[second_str] = (sort_data[second_sort] >= sort_data[second_sort].quantile(cum)) & \
                                        (sort_data[second_sort] < sort_data[second_sort].quantile(q + cum))
                cum += q
                j += 1
                second_col.append(second_str)
        else:
            for bp in second_breakpoint:
                second_str = second_sort + '_' + str(i)
                sort_data[second_str] = (sort_data[second_sort] >= bp) & \
                                        (sort_data[second_sort] < (bp + cum))
                cum += bp
                i += 1
                second_col.append(second_str)
    else:
        raise NotImplementedError
    sort_data['group'] = ''
    sort_data['weight'] = 0

    for s1, s2 in list(product(first_col, second_col)):
        col = s1 + '/' + s2
        if weight == 'cap':
            sort_data[col] = sort_data['capital'][sort_data[s1] & sort_data[s2]]
            sort_data[col].fillna(0, inplace=True)
            sort_data['weight'][sort_data[s1] & sort_data[s2]] = sort_data[col] / sort_data[col].sum()
        elif weight == 'equal':
            sort_data[col] = sort_data[s1] & sort_data[s2]
            sort_data['weight'][sort_data[s1] & sort_data[s2]] = sort_data[col] / sort_data[col].sum()
        sort_data['group'][sort_data[s1] & sort_data[s2]] = col

    sort_data = sort_data[['group', 'weight']]
    data = data['ret'].to_frame().join(sort_data)

    def cal_portfolio_ret(cross_section_data):
        return (cross_section_data['ret'] * cross_section_data['weight']).sum()

    grouper = [data.index.get_level_values(level=0), 'group']
    portfolio_ret = data.groupby(grouper).apply(lambda x: cal_portfolio_ret(x))
    portfolio_ret.name = 'ret'
    portfolio_ret = portfolio_ret.reset_index(level=1)

    def cal_hedge_portfolio_ret(port_ret):
        first_long = [g for g in port_ret['group'] if g.startswith(first_col[0])]
        first_short = [g for g in port_ret['group'] if g.startswith(first_col[-1])]
        second_long = [g for g in port_ret['group'] if g.endswith(second_col[-1])]
        second_short = [g for g in port_ret['group'] if g.endswith(second_col[0])]
        portfolio1 = (1 / len(first_long)) * (port_ret[port_ret['group'].isin(first_long)]['ret'].sum()) \
                     - (1 / len(first_short)) * (port_ret[port_ret['group'].isin(first_short)]['ret'].sum())

        portfolio2 = (1 / len(second_long)) * (port_ret[port_ret['group'].isin(second_long)]['ret'].sum()) \
                     - (1 / len(second_short)) * (port_ret[port_ret['group'].isin(second_short)]['ret'].sum())
        return pd.Series([portfolio1, portfolio2], index=['portfolio_1', 'portfolio_2'])

    hedged_portfolio_ret = portfolio_ret.groupby(level=0).apply(lambda x: cal_hedge_portfolio_ret(x))
    return hedged_portfolio_ret


def triple_sort_portfolio(data, first_sort='capital',
                          first_quantile=(0.5, 0.5),
                          second_sort='book_to_price_ratio',
                          second_quantile=(0.3, 0.4, 0.3),
                          third_sort='book_to_price_ratio',
                          third_quantile=(0.3, 0.4, 0.3),
                          independent_sort=True,
                          weight: str = 'cap'):
    pass


def cal_smb_hml_portfolio(merged_data: pd.DataFrame, rebalance: str = 'A-MAY'):
    portfolio_ret = merged_data.groupby(pd.Grouper(level=0, freq=rebalance)) \
        .apply(lambda x: double_sort_portfolio(x, first_sort='capital', first_quantile=(0.5, 0.5),
                                               second_sort='book_to_price_ratio', second_quantile=(0.3, 0.4, 0.3),
                                               independent_sort=True, weight='cap'))
    portfolio_ret.columns = ['SMB', 'HML']
    portfolio_ret = portfolio_ret.droplevel(0)
    return portfolio_ret


def cal_size_portfolio(merged_data: pd.DataFrame, rebalance: str = 'A-MAY'):
    portfolio_ret = merged_data.groupby(pd.Grouper(level=0, freq=rebalance)) \
        .apply(lambda x: uni_sort_portfolio(x, sort_name='capital', sort_quantile=(0.5, 0.5), weight='cap'))
    portfolio_ret.columns = ['Uni_SMB']
    portfolio_ret = portfolio_ret.droplevel(0)
    return portfolio_ret
