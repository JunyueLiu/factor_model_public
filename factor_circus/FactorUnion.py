import hashlib
import os.path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple, List, Optional

import pandas as pd
import tqdm

from data_management.dataIO.trading_calendar import trading_dates_offsets, Market, get_trading_date
from data_management.keeper.ZooKeeper import ZooKeeper
from data_management.pandas_utils.cache import cross_sectional_resample
from factor_circus.preprocessing import FactorProcesser, UniverseSelector


def transform_column(paras):
    factor_values, processes = paras
    name = factor_values.name
    if not any(isinstance(p, list) for p in processes):
        for strategy in processes:
            if isinstance(strategy, FactorProcesser):
                factor_values = strategy.fit_transform(factor_values)
            else:
                raise ValueError('Expected FactorPreprocesser, but {} is given'.format(type(strategy)))
        df = factor_values.unstack()
        df.columns = pd.MultiIndex.from_product([df.columns, [name]])
    else:
        fs = []
        for process in processes:
            factor_values1 = factor_values.copy()
            for strategy in process:
                if isinstance(strategy, FactorProcesser):
                    factor_values1 = strategy.fit_transform(factor_values1)
                else:
                    raise ValueError('Expected FactorPreprocesser, but {} is given'.format(type(strategy)))
            name = factor_values1.name
            factor_values1 = factor_values1.unstack()
            factor_values1.columns = pd.MultiIndex.from_product([factor_values1.columns, [name]])
            fs.append(factor_values1)
        df = pd.concat(fs, axis=1)

    return df


class FactorUnion:
    def __init__(self, factor_identities: Dict[Tuple[str, str], List[FactorProcesser]],
                 zookeeper: ZooKeeper,
                 start_date: str,
                 end_date: str,
                 offset: Optional[str],
                 data_input_config_path: str,
                 market: Market = Market.AShares,
                 local_factor: bool = True,
                 post_universe_selector: Optional[UniverseSelector] = None
                 ):
        """
        Use to learn factor from Zookeeper, preprocess the raw factor data, and concat it into a single dataframe
        :param factor_identities:
        :param zookeeper:
        :param start_date:
        :param end_date:
        :param offset:
        :param data_input_config_path:
        :param market:
        :param local_factor:
        """
        self.factor_identities = factor_identities
        self.zookeeper = zookeeper
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.data_input_config_path = data_input_config_path
        self.market = market
        self.local_factor = local_factor
        self.post_universe_selector = post_universe_selector

        trading_dates = get_trading_date(self.market, start_date=start_date, end_date=end_date,
                                         config_path=self.data_input_config_path)
        if offset:
            self.offset = trading_dates_offsets(trading_dates, offset)
        else:
            self.offset = None
        self.freq = offset
        self.factor_names = None
        self.factor_data: Optional[pd.DataFrame] = None
        self.save_folder = os.path.join(os.path.abspath(__file__), '../../data/factor_union')
        os.makedirs(self.save_folder, exist_ok=True)

    def transform(self):
        paras = [(category, factor_name) for (category, factor_name), processes in
                 self.factor_identities.items()]
        if len(paras) < 50:
            results = []
            for p in tqdm.tqdm(paras):
                r = self._get_factor(p)
                results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(tqdm.tqdm(executor.map(self._get_factor, paras, chunksize=10), total=len(paras)))
        # start_time = datetime.datetime.now()
        # data = pd.concat(results, axis=1)
        data = self.concat_results(results)
        data.index.names = ['date', 'code']

        paras = [(data[factor_name], processes) for (category, factor_name), processes in
                 self.factor_identities.items()]
        with ProcessPoolExecutor(max_workers=3) as executor:
            results = list(tqdm.tqdm(executor.map(transform_column, paras, chunksize=10), total=len(paras)))
        data = self.concat_results(results)
        data = data.dropna()
        if self.post_universe_selector is not None:
            data = self.post_universe_selector.fit_transform(data)

        data.index.names = ['date', 'code']
        self.factor_names = data.columns
        self.factor_data = data
        return data

    def _get_factor(self, paras):
        category, factor_name = paras
        values, d = self.zookeeper.get_factor_values(category, factor_name,
                                                     self.start_date, self.end_date,
                                                     self.local_factor)
        print(
            '{}: {} {}'.format(factor_name, values.index.get_level_values(0)[0], values.index.get_level_values(0)[-1]))

        if self.freq is None:
            pass
        elif d['freq'] == 'daily':
            if self.freq != 'D':
                values = cross_sectional_resample(values, self.offset)
        elif d['freq'] == 'monthly':
            values = cross_sectional_resample(values, self.offset)
        else:
            raise NotImplementedError

        df = values.unstack()
        df.columns = pd.MultiIndex.from_product([df.columns, [values.name]])
        df = df.loc[self.start_date: self.end_date]
        return df

    def concat_results(self, results: List[pd.Series]):
        data = pd.concat(results, axis=1)
        data = data.stack(level=0)
        data = data.sort_index()
        return data

    def get_sankey_diagram(self):
        # todo
        pass

    def load_factor_data(self, factor_circus_name, start_date=None, end_date=None):
        self.factor_data = pd.read_parquet(os.path.join(self.save_folder, factor_circus_name + '.parquet'))
        if start_date is not None:
            self.factor_data = self.factor_data.loc[pd.to_datetime(start_date):]
        if end_date is not None:
            self.factor_data = self.factor_data.loc[:pd.to_datetime(end_date)]

        return self.factor_data.copy()

    def save_factor_data(self, factor_circus_name=None):
        if not factor_circus_name:
            factor_circus_name = '_'.join(self.factor_names)
            factor_circus_name = 'factor_union_' + hashlib.sha256(factor_circus_name.encode('utf-8')).hexdigest()[:20]
        if os.path.exists(os.path.join(self.save_folder, factor_circus_name)):
            raise FileExistsError('Cannot save.')
        if self.factor_data is None:
            raise ValueError('Must transform or load to save')

        self.factor_data.to_parquet(os.path.join(self.save_folder, factor_circus_name + '.parquet'))
