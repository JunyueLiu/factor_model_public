import pandas as pd
import numpy as np
from scipy import stats
from plotly import graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
def _infer_strftime_format(data: pd.DatetimeIndex):
    unique_index = np.unique(data.values)
    unique_index.sort()
    time_delta = unique_index[1:] - unique_index[:-1]
    # get mode using scipy
    td = stats.mode(time_delta)[0][0]
    td = td.astype('timedelta64[m]')

    day = td / np.timedelta64('1', 'D')

    if day < 1:
        return '%Y/%m/%d %H:%M:%S'
    else:
        return '%Y/%m/%d'

def net_value_line(net_value: pd.Series, color='#00477D', name='net value', fill=None):
    strftime_format = _infer_strftime_format(net_value.index)
    timestamp = net_value.index  # type:pd.DatetimeIndex
    timestamp = timestamp.strftime(strftime_format)
    return go.Scatter(x=timestamp, y=net_value.values,
                      mode='lines', line_color=color,
                      name=name, fill=fill)


def net_value_plot(strategy_net_value: pd.Series,
                   benchmark: pd.Series or None = None,
                   strategy_name='strategy', fill=None):
    fig = go.Figure()
    fig.add_trace(net_value_line(strategy_net_value / strategy_net_value[0], name=strategy_name, fill=fill))
    if benchmark is not None:
        benchmark_copy = benchmark[
            (benchmark.index >= strategy_net_value.index[0]) & (benchmark.index <= strategy_net_value.index[-1])]
        fig.add_trace(net_value_line(benchmark_copy / benchmark_copy[0], color='#FFCC00', name='benchmark'))

    x_axis = fig.data[0].x
    tick_value = [x_axis[i] for i in range(0, len(x_axis), len(x_axis) // 5)]
    tick_text = [x_axis[i][0:10] for i in range(0, len(x_axis), len(x_axis) // 5)]
    fig.update_xaxes(ticktext=tick_text, tickvals=tick_value)

    return fig

def timing_contour(timing: pd.Series, y = (0, 1)):
    strftime_format = _infer_strftime_format(timing.index)
    timestamp = timing.index  # type:pd.DatetimeIndex
    timestamp = timestamp.strftime(strftime_format)
    colorscale = [[0, 'white'],
                  [0.5, 'mediumturquoise'], [1, 'navy']]
    return go.Contour(
        z=[timing.to_list(), timing.to_list()],
        x=timestamp,  # horizontal axis
        y=y,
        colorscale='Emrld',
        opacity=0.5
    )

if __name__ == '__main__':
    series = pd.read_csv(r'C:\Users\liuju\PycharmProjects\factor_model\research\quail\double_timing2\self_timing.csv',
                     index_col=0, infer_datetime_format=True).squeeze()
    series.index = pd.to_datetime(series.index)
    series = series.resample('D').last().fillna(method='ffill')
    c = timing_contour(series)
    fig = go.Figure(c)
    fig.show()