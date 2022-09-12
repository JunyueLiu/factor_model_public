import configparser
import datetime
import os
from typing import Union

import pandas as pd

from arctic.date import DateRange
from arctic.exceptions import LibraryNotFoundException, NoDataFoundException
from arctic.store.version_store import VersionStore
from data_management.cache_janitor.cache import hash_series, hash_dataframe
from data_management.dataIO.utils import get_arctic_store, read_pickle, save_pickle


class Keeper:
    def __init__(self, config_path):
        self.config_path = config_path
        config = configparser.ConfigParser()
        config.read(os.path.abspath(self.config_path))
        try:
            store = get_arctic_store(config)
            self.arctic_available = True
        except:
            store = None
            self.arctic_available = False
        self.config = config
        self.store = store
        self.factor_values = None
        base_path = os.path.join(os.path.abspath(__file__), '../../../')
        self.local_path = os.path.abspath(os.path.join(base_path, config['Local'].get('relative_path')))
        if not os.path.exists(self.local_path):
            self.local_path = config['Local'].get('absolute_path')
        self.context_name = config['Context']['name']
        self.local_path = os.path.join(os.path.abspath(self.local_path), self.context_name)
        os.makedirs(self.local_path, exist_ok=True)
        self.metadata_path = os.path.join(self.local_path, 'metadata')
        os.makedirs(self.metadata_path, exist_ok=True)

    def append_values_and_meta(self, category: str, factor_name: str,
                               values: Union[pd.Series, pd.DataFrame],
                               to_arctic: bool = True, **meta_kwargs):
        d = {}
        d['data_type'] = str(type(values))
        upload_time = datetime.datetime.now()
        lib_name = '{}.{}'.format(self.context_name, category)

        if isinstance(values, pd.DataFrame):
            if len(values.columns) == 1:
                values = values.squeeze()

        save_folder = os.path.join(self.local_path, category)
        meta_folder = os.path.join(self.metadata_path, category)
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(meta_folder, exist_ok=True)
        save_path = os.path.join(save_folder, factor_name + '.parquet')
        meta_save_path = os.path.join(meta_folder, factor_name + '.pickle')

        if not values.index.get_level_values(0).freq:
            time_idx = values.index.get_level_values(0)
            freq = self.infer_freq(time_idx)
        else:
            freq = values.index.get_level_values(0).freq

        if values.index.nlevels == 2:
            values = values.unstack()
        else:
            raise NotImplementedError

        try:
            old_values, _ = self.get_values_and_meta(category, factor_name,
                                                     local_factor=False if to_arctic else True)
            new_values = values[values.index > old_values.index[-1]]
            if len(new_values) == 0:
                print("already up to date for {}:{}".format(category, factor_name))
                return
            if isinstance(values, pd.Series):
                values = old_values.append(new_values)
                values.name = factor_name
            elif isinstance(values, pd.DataFrame):
                values = old_values.stack().append(new_values.stack())
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            return

        if values.index.nlevels == 1:
            factor_type = 'time-series'
            factor_values_hash = hash_series(values)
        elif values.index.nlevels == 2:
            factor_type = 'cross-section'
            values = values.unstack()
            factor_values_hash = hash_dataframe(values)
        else:
            raise NotImplementedError

        d['values_type'] = factor_type
        d['freq'] = freq
        d['upload_time'] = upload_time
        d['values_hash'] = factor_values_hash
        for k, w in meta_kwargs.items():
            d[k] = w

        if isinstance(values, pd.Series):
            values.to_frame().to_parquet(save_path)
        elif isinstance(values, pd.DataFrame):
            values.to_parquet(save_path)
        else:
            raise NotImplementedError
        save_pickle(d, meta_save_path)
        print('save local {}/{} success'.format(category, factor_name))

        if to_arctic and self.arctic_available:
            if not self.store.library_exists(lib_name):
                self.store.initialize_library(lib_name)
            lib = self.store.get_library(lib_name)
            lib.detele(factor_name)
            lib.write(factor_name, values, metadata=d)
            print('upload {}/{} success'.format(category, factor_name))
        elif to_arctic and not self.arctic_available:
            raise Exception('Arctic not available')

    def save_values_and_meta(self, category: str, factor_name: str,
                             values: Union[pd.Series, pd.DataFrame],
                             to_arctic: bool = True, **meta_kwargs):
        d = {}
        d['data_type'] = str(type(values))
        upload_time = datetime.datetime.now()
        lib_name = '{}.{}'.format(self.context_name, category)

        if isinstance(values, pd.DataFrame):
            if len(values.columns) == 1:
                values = values.squeeze()

        if not values.index.get_level_values(0).freq:
            time_idx = values.index.get_level_values(0)
            freq = self.infer_freq(time_idx)
        else:
            freq = values.index.get_level_values(0).freq

        if values.index.nlevels == 1:
            factor_type = 'time-series'
            factor_values_hash = hash_series(values)
        elif values.index.nlevels == 2:
            factor_type = 'cross-section'
            values = values.unstack()
            factor_values_hash = hash_dataframe(values)
        else:
            raise NotImplementedError

        d['values_type'] = factor_type
        d['freq'] = freq
        d['upload_time'] = upload_time
        d['values_hash'] = factor_values_hash
        for k, w in meta_kwargs.items():
            d[k] = w
        save_folder = os.path.join(self.local_path, category)
        meta_folder = os.path.join(self.metadata_path, category)
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(meta_folder, exist_ok=True)
        save_path = os.path.join(save_folder, factor_name + '.parquet')
        meta_save_path = os.path.join(meta_folder, factor_name + '.pickle')
        if isinstance(values, pd.Series):
            values.to_frame().to_parquet(save_path)
        elif isinstance(values, pd.DataFrame):
            values.to_parquet(save_path)
        else:
            raise NotImplementedError
        save_pickle(d, meta_save_path)
        print('save local {}/{} success'.format(category, factor_name))

        if to_arctic and self.arctic_available:
            if not self.store.library_exists(lib_name):
                self.store.initialize_library(lib_name)
            lib = self.store.get_library(lib_name)
            lib.write(factor_name, values, metadata=d)
            print('upload {}/{} success'.format(category, factor_name))
        elif to_arctic and not self.arctic_available:
            raise Exception('Arctic not available')

    def get_metadata(self, category: str, value_name: str):
        meta_folder = os.path.join(self.metadata_path, category)
        meta_save_path = os.path.join(meta_folder, value_name + '.pickle')
        d = read_pickle(meta_save_path)
        return d

    def get_values_and_meta(self, category: str, value_name: str, start_date=None, end_date=None, local_factor=True):
        lib_name = '{}.{}'.format(self.context_name, category)
        if local_factor:
            save_folder = os.path.join(self.local_path, category)
            meta_folder = os.path.join(self.metadata_path, category)
            save_path = os.path.join(save_folder, value_name + '.parquet')
            meta_save_path = os.path.join(meta_folder, value_name + '.pickle')
            d = read_pickle(meta_save_path)
            values = pd.read_parquet(save_path)
            if start_date is not None:
                values = values.loc[pd.to_datetime(start_date):]
            if end_date is not None:
                values = values.loc[:pd.to_datetime(end_date)]

        elif self.arctic_available:
            version_store = self.store[lib_name]  # type: VersionStore
            version_item = version_store.read(value_name, date_range=DateRange(start_date, end_date))
            d = version_item.metadata
            values = version_item.data  # type: Union[pd.Series, pd.DataFrame]
        else:
            raise Exception('Arctic not avaiabl')

        return values, d

    def synchronize_local_from_arctic(self):
        if self.arctic_available:
            for c in self.store.list_libraries():
                if c.startswith(self.context_name):
                    save_folder = os.path.join(self.local_path, c)
                    meta_folder = os.path.join(self.metadata_path, c)
                    os.makedirs(save_folder, exist_ok=True)
                    os.makedirs(meta_folder, exist_ok=True)
                    version_store = self.store[c]
                    for name in self.store[c].list_symbols():
                        version_item = version_store.read(name)
                        d = version_item.metadata
                        values = version_item.data
                        save_path = os.path.join(save_folder, name + '.parquet')
                        meta_save_path = os.path.join(meta_folder, name + '.pickle')
                        values.to_parquet(save_path)
                        save_pickle(d, meta_save_path)
                        print('synchronized {}/{} from arctic'.format(c, name))

        else:
            print('Fail. Arctic not available')

    def synchronize_arctic_from_local(self):
        if self.arctic_available:
            for c in os.listdir(self.local_path):
                if c == 'metadata':
                    continue
                save_folder = os.path.join(self.local_path, c)
                meta_folder = os.path.join(self.metadata_path, c)
                lib_name = '{}.{}'.format(self.context_name, c)
                if not self.store.library_exists(lib_name):
                    self.store.initialize_library(lib_name)

                for name in os.listdir(save_folder):
                    save_path = os.path.join(save_folder, name)
                    meta_save_path = os.path.join(meta_folder, name.replace('parquet', 'pickle'))
                    values = pd.read_parquet(save_path)
                    d = read_pickle(meta_save_path)
                    version_store = self.store[lib_name]  # type: VersionStore
                    version_store.write(name.replace('.parquet', ''), values, d)
                    print('synchronized to arctic {}/{} from local'.format(c, name))

        else:
            print('Fail. Arctic not available')

    def delete_factor(self, category: str, value_name: str):
        save_folder = os.path.join(self.local_path, category)
        meta_folder = os.path.join(self.metadata_path, category)
        save_path = os.path.join(save_folder, value_name + '.parquet')
        meta_save_path = os.path.join(meta_folder, value_name + '.pickle')
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(meta_save_path):
            os.remove(meta_save_path)

        if self.arctic_available:
            lib_name = '{}.{}'.format(self.context_name, category)
            try:
                version_store = self.store[lib_name]  # type: VersionStore
                version_store.delete(value_name)
            except (LibraryNotFoundException, NoDataFoundException):
                pass

    def list_categories(self, local=True):
        if local:
            return [category for category in os.listdir(self.metadata_path) if
                    os.path.isdir(os.path.join(self.metadata_path, category))]
        elif self.arctic_available:
            self.store.reload_cache()
            return [c for c in self.store.list_libraries() if c.startswith(self.context_name)]
        return []

    def list_value_name(self, category, local=True):
        if local:
            meta_folder = os.path.join(self.metadata_path, category)
            return [factor_name.replace('.pickle', '') for factor_name in os.listdir(meta_folder)]
        elif self.arctic_available:
            lib_name = '{}.{}'.format(self.context_name, category)
            return self.store[lib_name].list_symbols()
        return []

    def infer_freq(self, idx):
        idx = idx.drop_duplicates().sort_values()
        delta = (idx[1:] - idx[:-1]).median().days
        if delta == 0:
            td = (idx[1:] - idx[:-1]).median()
            seconds = td.seconds
            minutes = int(seconds / 60)
            if minutes >= 60:
                hours = int(minutes / 60)
                freq = '{}h'.format(hours)
            elif minutes == 0:
                freq = '{}s'.format(seconds)
            else:
                freq = '{}m'.format(minutes)
        elif delta == 1:
            freq = 'daily'
        elif 5 <= delta < 10:
            freq = 'weekly'
        elif 10 <= delta < 15:
            freq = 'half-month'
        elif 20 <= delta < 35:
            freq = 'monthly'
        else:
            freq = '{} days'.format(delta)
        return freq
