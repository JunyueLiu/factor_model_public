import os
from typing import Callable, Union, Optional, Tuple, Dict

import pandas as pd

from data_management.dataIO.component_data import get_index_component, get_north_connect_component, \
    get_industry_component, get_industry_info, IndustryCategory
from data_management.dataIO.fundamental_data import get_stock_info
from data_management.dataIO.index_data import IndexTicker
from data_management.dataIO.trade_data import get_trade, TradeTable
from data_management.dataIO.utils import read_pickle
from data_management.keeper.Keeper import Keeper
from data_management.keeper.utils import add_stock_info, add_industry_info, add_daily_basic, add_component
from factor_zoo.industry import industry_category


class ZooKeeper(Keeper):
    def __init__(self, config_path: str):
        super(ZooKeeper, self).__init__(config_path)

    def append_factor(self, category: str, factor_name: str,
                      append_values: pd.Series):
        factor_values, d = self.get_factor_values(category, factor_name)


        self.save_factor_value(category, factor_values)




    def list_factor_categories(self, local=True):
        if local:
            return [category for category in os.listdir(self.metadata_path) if
                    os.path.isdir(os.path.join(self.metadata_path, category))]
        elif self.arctic_available:
            self.store.reload_cache()
            return [c for c in self.store.list_libraries() if c.startswith(self.context_name)]

    def cal_factor(self, factor_function: Callable, **kwargs):
        raise NotImplementedError

    def get_factor_values(self, category: str, factor_name: str, start_date=None, end_date=None, local_factor=True) \
            -> Tuple[pd.Series, Dict]:
        factor_values, d = self.get_values_and_meta(category, factor_name, start_date, end_date, local_factor)
        factor_type = d['values_type']
        if factor_type == 'cross-section':
            factor_values = factor_values.stack()
            factor_values.name = factor_name
        return factor_values, d

    def get_factor_values_from_csv(self, category: str, factor_name: str, start_date=None, end_date=None, local_factor=True) \
            -> pd.DataFrame:

        load_folder = os.path.join(self.local_path, category)
        load_path = os.path.join(load_folder, factor_name + '.csv')
        factor_values = pd.read_csv(load_path, index_col='date', engine='python')
        factor_values.index = pd.to_datetime(factor_values.index)
        print('load local {}/{} csv success'.format(category, factor_name))
                
        return factor_values

    def get_factor_value_and_info(self, category: str, factor_name: str, start_date=None, end_date=None,
                                  local_factor=True):
        input_config = self.config['Info']['data_input']
        factor_data = self.get_factor_values(category, factor_name, start_date, end_date, local_factor)
        if start_date is None:
            start_date = factor_data.index.get_level_values(0)[0]

        if end_date is None:
            end_date = factor_data.index.get_level_values(0)[-1]
        dates = factor_data.index.get_level_values(0).drop_duplicates().to_list()
        stock_info = get_stock_info(input_config)
        csi_300 = get_index_component(IndexTicker.csi300, dates, config_path=input_config)
        zz_500 = get_index_component(IndexTicker.zz500, dates, config_path=input_config)
        market_connect = get_north_connect_component(dates, config_path=input_config)
        sw_l1 = get_industry_component(IndustryCategory.sw_l1, date=dates, config_path=input_config)
        sw1_info = get_industry_info(IndustryCategory.sw_l1, config_path=input_config)
        daily_basic = get_trade(TradeTable.daily_basic, start_date=start_date, end_date=end_date,
                                config_path=input_config)
        factor_data = add_stock_info(factor_data, stock_info)
        factor_data = add_industry_info(factor_data, sw_l1, sw1_info)
        factor_data = add_daily_basic(factor_data, daily_basic)
        factor_data = add_component(factor_data, {'csi300': csi_300,
                                                  'zz500': zz_500, 'market_connect': market_connect})
        return factor_data

    def export_factor_value_info_excel(self, category: str,
                                       factor_name: str,
                                       start_date=None, end_date=None,
                                       local_factor=True):
        factor_data = self.get_factor_value_and_info(category, factor_name, start_date, end_date, local_factor)
        excel_path = input('Input save target path')

        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        factor_data.to_excel(writer, merge_cells=False)
        worksheet = writer.sheets['Sheet1']
        worksheet.autofilter(0, 0, len(factor_data), len(factor_data.columns))
        writer.save()


    def append_factor_value(self,category: str,
                          factor_values: Union[pd.Series, pd.DataFrame],
                          factor_name: Optional[str] = None,
                          to_arctic: bool = True,
                          source_code: Optional[str] = None, comment=None, **kwargs):
        if factor_name is None:
            if factor_values.name is None:
                raise ValueError('Must provide factor name. Use factor_name parameter or set Series name')
            else:
                factor_name = factor_values.name

        kwargs['source_code'] = source_code
        kwargs['comment'] = comment
        self.append_values_and_meta(category, factor_name, factor_values, to_arctic, **kwargs)


    def save_factor_value(self, category: str,
                          factor_values: Union[pd.Series, pd.DataFrame],
                          factor_name: Optional[str] = None,
                          to_arctic: bool = True,
                          source_code: Optional[str] = None, comment=None, **kwargs):
        """

        :param category:
        :param factor_values:
        :param factor_name:
        :param source_code:
        :param comment:
        :param kwargs:
        :return:
        """
        if factor_name is None:
            if factor_values.name is None:
                raise ValueError('Must provide factor name. Use factor_name parameter or set Series name')
            else:
                factor_name = factor_values.name

        kwargs['source_code'] = source_code
        kwargs['comment'] = comment
        self.save_values_and_meta(category, factor_name, factor_values, to_arctic, **kwargs)


    def save_factor_value_csv(self, category: str,
                          factor_values: Union[pd.Series, pd.DataFrame],
                          factor_name: Optional[str] = None):
        """

        :param category:
        :param factor_values:
        :param factor_name:
        :return:
        """
        if factor_name is None:
            if factor_values.name is None:
                raise ValueError('Must provide factor name. Use factor_name parameter or set Series name')
            else:
                factor_name = factor_values.name

        save_folder = os.path.join(self.local_path, category)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, factor_name + '.csv')
        if isinstance(factor_values, pd.Series):
            factor_values.to_frame().to_csv(save_path)
        elif isinstance(factor_values, pd.DataFrame):
            factor_values.to_csv(save_path)
        else:
            raise NotImplementedError
        print('save local {}/{} to csv success'.format(category, factor_name))



    def print_all_factor_zoo_info(self, local_factor=True):
        info = []
        if local_factor:
            for c in self.list_factor_categories(local_factor):
                meta_folder = os.path.join(self.metadata_path, c)
                for factor_name in os.listdir(meta_folder):
                    d = read_pickle(os.path.join(meta_folder, factor_name))
                    d['factor_name'] = factor_name.replace('.pickle', '')
                    d['category'] = c.replace('factor_zoo.', '')
                    if 'source_code' in d:
                        del d['source_code']
                    if 'values_hash' in d:
                        del d['values_hash']
                    info.append(d)
        elif self.arctic_available:
            for c in self.list_factor_categories(local_factor):
                for factor_name in self.store[c].list_symbols():
                    d = self.store[c].read_metadata(factor_name).metadata
                    d['factor_name'] = factor_name
                    d['category'] = c.replace('factor_zoo.', '')
                    if 'source_code' in d:
                        del d['source_code']
                    if 'values_hash' in d:
                        del d['values_hash']
                    info.append(d)
        else:
            raise Exception('Arctic not available')
        df = pd.DataFrame(info).set_index(['category', 'factor_name']).sort_index()
        print(df.to_string())

    def __save_calculated_factor_value(self):
        raise NotImplementedError

    def generate_holding(self, stock_series: pd.Series, data_input_config, date: str=None):
        industry_info = get_industry_info(IndustryCategory.sw_l1, config_path=data_input_config)
        sw_l1 = get_industry_component(IndustryCategory.sw_l1, config_path=data_input_config)
        sec_info = get_stock_info(config_path=data_input_config)

        # sec_info.index.names = ['code']
        # cat = industry_category(sw_l1).astype(str)
        # industry_info.index.names = ['industry_code']
        # industry_info.index = industry_info.index.astype(str)

        # f = f.loc[date].to_frame().join(sec_info['display_name']).join(cat)
        # f = f.set_index('industry_code', append=True)
        # f = f.join(industry_info['name']).reset_index(level=2)
        # f = f.rename(columns={'name': 'sw1_name'})
        # f = f[['display_name', 'industry_code', 'sw1_name', 'ret']]

        if stock_series.index.nlevels != 2:
            raise ValueError

        stock_series.index.names = ['date', 'code']

        sec_info = sec_info
        industry_info = industry_info
        sec_info.index.names = ['code']
        cat = industry_category(sw_l1).astype(str)
        industry_info.index.names = ['industry_code']
        industry_info.index = industry_info.index.astype(str)

        stock_df = stock_series.to_frame().join(sec_info['display_name']).join(cat)
        stock_df = stock_df.set_index('industry_code', append=True)
        stock_df = stock_df.join(industry_info['name']).reset_index(level=2)
        stock_df = stock_df.rename(columns={'name': 'sw1_name'})
        stock_df = stock_df[['display_name', 'industry_code', 'sw1_name', stock_series.name]]
        if date:
            stock_df = stock_df.loc[date]
        return stock_df



if __name__ == '__main__':
    # keeper = Keeper('../../cfg/factor_keeper_setting.ini')
    # factor = pd.read_parquet('../../data/factors/analyst/alsue0.parquet')
    # keeper.save_factor_value('analyst', factor)
    # f = keeper.get_factor_value('analyst', 'alsue0')
    # keeper.print_all_factor_zoo_info()
    # keeper.get_factor_value('turnover','ideal_turnover_daily_basic_20_0.25')

    aliyun_cfg_path = '../../cfg/factor_keeper_setting.ini'
    keeper2 = ZooKeeper(aliyun_cfg_path)
    # f = keeper2.get_factor_values('moneyflow', 'avg_net_amount_l_10')
    # keeper2.export_factor_value_info_excel('analyst', 'alsue0', start_date='2021-08-01', end_date='2021-09-01')
    # f = keeper2.store.get_library('factor_zoo.turnover').read('ideal_turnover_daily_basic_20_0.25').data
    # s = keeper2.get_factor_values('turnover','ideal_turnover_daily_basic_20_0.25', '2021-01-01')
