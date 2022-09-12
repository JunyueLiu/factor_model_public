from enum import Enum
from typing import List, Union, Tuple


class ChartType(Enum):
    Line = 'line'
    Column = 'column'


def two_color_heatmap(workbook, worksheet, first_row, first_col, last_row, last_col):
    format1 = workbook.add_format({'bg_color': '#FFC7CE',
                                   'font_color': '#9C0006'})

    # Add a format. Green fill with dark green text.
    format2 = workbook.add_format({'bg_color': '#C6EFCE',
                                   'font_color': '#006100'})

    worksheet.conditional_format(first_row, first_col, last_row, last_col, {'type': 'cell',
                                                                            'criteria': '>=',
                                                                            'value': 0,
                                                                            'format': format2})
    worksheet.conditional_format(first_row, first_col, last_row, last_col, {'type': 'cell',
                                                                            'criteria': '<',
                                                                            'value': 0,
                                                                            'format': format1})


def t_statistics_heatmap(workbook, worksheet, first_row, first_col, last_row, last_col):
    format1 = workbook.add_format({'bg_color': '#FFC7CE',
                                   'font_color': '#9C0006'})

    # Add a format. Green fill with dark green text.
    format2 = workbook.add_format({'bg_color': '#C6EFCE',
                                   'font_color': '#006100'})
    # green for larger than 2
    worksheet.conditional_format(first_row, first_col, last_row, last_col, {'type': 'cell',
                                                                            'criteria': '>=',
                                                                            'value': 2,
                                                                            'format': format2})
    # red for larger than 2
    worksheet.conditional_format(first_row, first_col, last_row, last_col, {'type': 'cell',
                                                                            'criteria': '<',
                                                                            'value': -2,
                                                                            'format': format1})


def three_color_scale(worksheet, first_row, first_col, last_row, last_col):
    worksheet.conditional_format(first_row, first_col, last_row, last_col, {'type': '3_color_scale'})


def autofit_column(worksheet, df):
    for column in df:
        column_width = max(df[column].astype(str).map(len).max(), len(column))
        col_idx = df.columns.get_loc(column)
        worksheet.set_column(col_idx, col_idx, column_width)


def border_by_first_level_columns(workbook, worksheet, df):
    first_level_columns = df.columns.get_level_values(0).drop_duplicates()
    cell_format1 = workbook.add_format()
    cell_format1.set_left()
    cell_format2 = workbook.add_format()
    cell_format2.set_right()
    for c in first_level_columns:
        slice_ = df.columns.get_loc(c)
        start_col = slice_.start
        stop_col = slice_.stop
        column_width1 = df.iloc[:, start_col].astype(str).map(len).max()
        column_width2 = df.iloc[:, stop_col - 1].astype(str).map(len).max()
        worksheet.set_column(start_col + 1, start_col + 1, column_width1, cell_format=cell_format1)
        worksheet.set_column(stop_col + 1, stop_col + 1, column_width2, cell_format=cell_format2)


def corr_matrix_heatmap(worksheet, first_row, first_col, last_row, last_col):
    worksheet.conditional_format(first_row, first_col, last_row, last_col,
                                 {'type': '3_color_scale',
                                  'min_value': -1,
                                  'mid_value': 0,
                                  'max_value': 1,
                                  # 'min_color': '#FF0000',
                                  # 'mid_color': '#FFFFFF',
                                  # 'max_color': '#0000FF'
                                  })


def insert_chart(workbook, worksheet,
                 chart_type: ChartType,
                 sheet_name: str,
                 first_rows: Union[int, List[int]], first_cols: Union[int, List[int]],
                 last_rows: Union[int, List[int]], last_cols: Union[int, List[int]],
                 series_names: Union[str, List[str]],
                 chart_location,
                 x_title='', y_title='',
                 chart_title='',
                 x_scale: float = 1, y_scale: float = 1):
    chart = workbook.add_chart({'type': chart_type.value})
    #     [sheetname, first_row, first_col, last_row, last_col]

    first_rows = first_rows if isinstance(first_rows, list) else [first_rows]
    first_cols = first_cols if isinstance(first_cols, list) else [first_cols]
    last_rows = last_rows if isinstance(last_rows, list) else [last_rows]
    last_cols = last_cols if isinstance(last_cols, list) else [last_cols]
    series_names = series_names if isinstance(series_names, list) else [series_names]

    for name, fr, fc, lr, lc in zip(series_names, first_rows, first_cols, last_rows, last_cols):
        chart.add_series({
            'name': name,
            'categories': [sheet_name, fr, first_cols[0] - 1, lr, first_cols[0] - 1],
            'values': [sheet_name, fr, fc, lr, lc],
        })
    # Configure the chart axes.
    chart.set_x_axis({'name': x_title})
    chart.set_y_axis({'name': y_title, 'major_gridlines': {'visible': False}})
    chart.set_title({'name': chart_title})
    chart.set_size({'x_scale': x_scale, 'y_scale': y_scale})
    # Insert the chart into the worksheet.
    worksheet.insert_chart(chart_location, chart)


def insert_chart2(workbook, worksheet,
                  chart_type: ChartType,
                  sheet_name: str,
                  category_rows_cols: Tuple[str, int, int, int, int],
                  first_rows: Union[int, List[int]], first_cols: Union[int, List[int]],
                  last_rows: Union[int, List[int]], last_cols: Union[int, List[int]],
                  series_names: Union[str, List[str]],
                  chart_location,
                  x_title='', y_title='',
                  chart_title='',
                  x_scale: float = 1, y_scale: float = 1):
    chart = workbook.add_chart({'type': chart_type.value})
    #     [sheetname, first_row, first_col, last_row, last_col]

    first_rows = first_rows if isinstance(first_rows, list) else [first_rows]
    first_cols = first_cols if isinstance(first_cols, list) else [first_cols]
    last_rows = last_rows if isinstance(last_rows, list) else [last_rows]
    last_cols = last_cols if isinstance(last_cols, list) else [last_cols]
    series_names = series_names if isinstance(series_names, list) else [series_names]

    for name, fr, fc, lr, lc in zip(series_names, first_rows, first_cols, last_rows, last_cols):
        chart.add_series({
            'name': name,
            'categories': list(category_rows_cols),
            'values': [sheet_name, fr, fc, lr, lc],
        })
    # Configure the chart axes.
    chart.set_x_axis({'name': x_title})
    chart.set_y_axis({'name': y_title, 'major_gridlines': {'visible': False}})
    chart.set_title({'name': chart_title})
    chart.set_size({'x_scale': x_scale, 'y_scale': y_scale})
    # Insert the chart into the worksheet.
    worksheet.insert_chart(chart_location, chart)


def insert_chart3(workbook, worksheet,
                  chart_type: ChartType,
                  sheet_name: List[str],
                  category_rows_cols: Tuple[str, int, int, int, int],
                  first_rows: Union[int, List[int]], first_cols: Union[int, List[int]],
                  last_rows: Union[int, List[int]], last_cols: Union[int, List[int]],
                  series_names: Union[str, List[str]],
                  chart_location,
                  x_title='', y_title='',
                  chart_title='',
                  x_scale: float = 1, y_scale: float = 1):
    chart = workbook.add_chart({'type': chart_type.value})
    #     [sheetname, first_row, first_col, last_row, last_col]

    first_rows = first_rows if isinstance(first_rows, list) else [first_rows]
    first_cols = first_cols if isinstance(first_cols, list) else [first_cols]
    last_rows = last_rows if isinstance(last_rows, list) else [last_rows]
    last_cols = last_cols if isinstance(last_cols, list) else [last_cols]
    series_names = series_names if isinstance(series_names, list) else [series_names]

    for name, sn, fr, fc, lr, lc in zip(series_names, sheet_name, first_rows, first_cols, last_rows, last_cols):
        chart.add_series({
            'name': name,
            'categories': list(category_rows_cols),
            'values': [sn, fr, fc, lr, lc],
        })
    # Configure the chart axes.
    chart.set_x_axis({'name': x_title})
    chart.set_y_axis({'name': y_title, 'major_gridlines': {'visible': False}})
    chart.set_title({'name': chart_title})
    chart.set_size({'x_scale': x_scale, 'y_scale': y_scale})
    # Insert the chart into the worksheet.
    worksheet.insert_chart(chart_location, chart)


def insert_NewlyWest_chart(workbook, worksheet,
                           sheet_name: str,
                           row: int, columns: Tuple,
                           chart_location,
                           x_title='', y_title='',
                           chart_title='',
                           x_scale: float = 1, y_scale: float = 1, column_trendline=None):
    chart = workbook.add_chart({'type': ChartType.Column.value})
    #     [sheetname, first_row, first_col, last_row, last_col]
    d = {
        'name': 'mean_ret',
        'values': [sheet_name, row, columns[0], row, columns[1]],
        'data_labels': {'value': True},
    }
    if column_trendline:
        d['trendline'] = {'type': 'polynomial', 'order': 3, }
    chart.add_series(d)
    # Configure the chart axes.
    chart.set_x_axis({'name': x_title})
    chart.set_y_axis({'name': y_title, 'major_gridlines': {'visible': False}})
    chart.set_title({'name': chart_title})
    chart.set_size({'x_scale': x_scale, 'y_scale': y_scale})
    # Insert the chart into the worksheet.
    worksheet.insert_chart(chart_location, chart)
