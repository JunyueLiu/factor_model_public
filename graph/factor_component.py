import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def ic_bar_chart(ic_stats: pd.Series):
    # all              -0.058837
    # csi300           -0.023845
    # zz500            -0.035462
    # zz800            -0.030699
    # zz1000           -0.053025
    # market_connect   -0.040345
    fig = go.Figure(data=[
        go.Bar(name=ic_stats.name, x=ic_stats.index, y=ic_stats.values)
    ])
    return fig


def cum_ic_chart(cumsum_ic: pd.DataFrame):
    fig = go.Figure()
    for c in cumsum_ic.columns:
        fig.add_scatter(x=cumsum_ic.index, y=cumsum_ic[c].values, mode='lines', name=c[1])
    return fig


def cum_ret_chart(cum_ret: pd.DataFrame):
    fig = go.Figure()
    for c in cum_ret.columns:
        fig.add_scatter(x=cum_ret.index, y=cum_ret[c].values, mode='lines', name='{} portfolio'.format(c))

    return fig


def ic_bar_comparison_chart(ic_stats: pd.Series):
    # factor    label                                                 universe
    # f1 label  all               0.016647
    #            csi300            0.032386
    #            market_connect    0.020310
    #            zz1000            0.023684
    #            zz500             0.019091
    #            zz800             0.018518
    # f2  label  all               0.033916
    #           csi300            0.020415
    #           market_connect    0.026526
    #           zz1000            0.023522
    #           zz500             0.032948
    #          zz800             0.021927
    # dtype: float64
    title = 'Label: {}'.format(ic_stats.index.get_level_values(1)[0])
    fig = go.Figure()
    for factor_name, data in ic_stats.groupby(level=0):
        fig.add_trace(go.Bar(name=factor_name, x=data.index.get_level_values(2), y=data, text=data,
                             texttemplate='%{text:.4f}', textposition='outside'
                             ))

    fig.update_layout(title=title)
    return fig


def cum_ic_comparison_chart(cumsum_ic: pd.DataFrame):
    title = 'Label: {}'.format(cumsum_ic.columns.get_level_values(1)[0])
    fig = go.Figure()
    line_dash = [None, 'dash', 'dot', 'dashdot']
    colors = px.colors.qualitative.Plotly
    i = 0
    for factor_name, data in cumsum_ic.groupby(level=0, axis=1):
        j = 0
        for c in data.columns:
            fig.add_scatter(x=cumsum_ic.index, y=cumsum_ic[c].values, mode='lines',
                            line=dict(color=colors[j % len(colors)], dash=line_dash[i % len(line_dash)]),
                            name='{}+{}'.format(factor_name, c[2]))
            j += 1
        i += 1
    fig.update_layout(title=title)
    return fig


def ic_comparison_chart(ic_to_compare: pd.DataFrame):
    title = 'Label: {}'.format(ic_to_compare.columns.get_level_values(1)[0])
    ic = ic_to_compare.mean()
    icir = ic_to_compare.mean() / ic_to_compare.std()
    cum_ic = ic_to_compare.cumsum()
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("IC", "ICIR", "Cumulative IC"), vertical_spacing=0.1)
    fig.add_traces(ic_bar_comparison_chart(ic).data, 1, 1)
    fig.add_traces(ic_bar_comparison_chart(icir).data, 2, 1)
    fig.add_traces(cum_ic_comparison_chart(cum_ic).data, 3, 1)
    fig.update_layout(height=1500,
                      title_text=title)
    return fig
