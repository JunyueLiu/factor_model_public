import configparser
import datetime
import hashlib
import os
from typing import Dict, List, Tuple

import pandas as pd
import tqdm

from arctic import Arctic, VERSION_STORE
from arctic.store.versioned_item import VersionedItem
from data_management.dataIO.component_data import IndexTicker, get_index_component, get_north_connect_component
from data_management.dataIO.utils import get_arctic_store, read_json
from data_management.keeper.LabelGuardian import LabelGuardian
from data_management.keeper.ZooKeeper import ZooKeeper
from data_management.pandas_utils.cache import panel_df_join
from factor_testing.performance import cal_ic, quantile_backtesting, quantize_factor
from factor_zoo.utils import market_filter_in


class PerformanceCoach:

    def __init__(self, config_path: str,
                 zookeeper: ZooKeeper,
                 label_guardian: LabelGuardian,
                 setting: Dict,
                 ):

        self.config_path = config_path
        config = configparser.ConfigParser()
        config.read(os.path.abspath(self.config_path))
        store: Arctic = get_arctic_store(config)
        self.config = config
        self.context = self.config['Context']['name']
        self.data_input = os.path.join(self.config['Info']['project_path'],
                                       self.config['Info']['data_input'])
        self.store = store
        self.zoo_keeper = zookeeper
        self.label_guardian = label_guardian
        self.setting = setting
        self.subscribe_factors = []
        self.subscribe_label = []
        self.universe_component = {}
        self.freq_priority = {'daily': 0, 'weekly': 1, 'monthly': 3, '4h': 0, 'intraday': 0}
        self._parse_setting()

    def _parse_setting(self):
        project_path = self.config['Info']['project_path']
        sub_factor_path = self.config['Info']['subscribe_factor_json_path']
        sub_label_path = self.config['Info']['subscribe_label_json_path']
        d = read_json(os.path.join(project_path, sub_factor_path))
        for category, factors in d.items():
            for f in factors:
                self.subscribe_factors.append((category, f))

        d1 = read_json(os.path.join(project_path, sub_label_path))
        for category, factors in d1.items():
            for f in factors:
                self.subscribe_label.append((category, f))

    def _check_factor_need_cal(self, stats_name: str,
                               version_store, factor_meta) -> bool:
        need_cal = True
        if version_store.has_symbol(stats_name):
            d = version_store.read_metadata(stats_name).metadata
            if d['factor_hash'] == factor_meta['values_hash']:
                need_cal = False
        return need_cal

    def _check_need_cal(self, stats_name: str,
                        version_store, label_meta, factor_meta, force_cal=False) -> bool:
        if force_cal:
            return True
        need_cal = True
        if version_store.has_symbol(stats_name):
            d = version_store.read_metadata(stats_name).metadata
            if d['label_hash'] == label_meta['values_hash'] and \
                    d['factor_hash'] == factor_meta['values_hash']:
                need_cal = False
        return need_cal

    def _get_universe_data(self, data, universe):
        if universe == 'all':
            new_data = data.dropna()
        else:
            if universe in IndexTicker._member_names_:
                if universe not in self.universe_component:
                    component = get_index_component(IndexTicker[universe], config_path=self.data_input)
                    self.universe_component[universe] = component
                else:
                    component = self.universe_component[universe]
            elif universe == 'market_connect':
                if universe not in self.universe_component:
                    component = get_north_connect_component(config_path=self.data_input)
                    self.universe_component[universe] = component
                else:
                    component = self.universe_component[universe]
            else:
                raise ValueError()
            new_data = market_filter_in(data.dropna(), component)
        return new_data

    def _get_name(self, label_cat, label_name,
                  factor_cap, factor_name):
        name = hashlib.sha256(
            '{}_{}_{}_{}'.format(label_cat, label_name,
                                 factor_cap, factor_name).encode('utf-8')).hexdigest()
        return name[:20]

    def _cal_ic(self, version_store: VERSION_STORE,
                data: pd.DataFrame,
                factor_name: str,
                metadata: Dict, force_cal=False):
        """

        :param version_store:
        :param data:
        :param factor_name:
        :param metadata:
        :return:
        """
        universes = self.config['IC']['universe'].replace(' ', '').split(',')
        if not self._check_need_cal('IC', version_store, metadata['label_meta'],
                                    metadata['factor_meta'], force_cal):
            print('IC {} is already tested, skip'.format(factor_name))
            return
        all_ic = []
        for universe in universes:
            new_data = self._get_universe_data(data, universe)
            ic = cal_ic(new_data, factor_name, False, False)
            ic.columns = pd.MultiIndex.from_product([ic.columns, [universe]])
            all_ic.append(ic)
        ic = pd.concat(all_ic, axis=1)
        version_store.write('IC', ic, metadata=metadata)

    def _cal_group_turnover(self, data, factor_name, _bin, _quantiles):
        group = quantize_factor(data, factor_name,
                                quantiles=_quantiles, bins=_bin)
        all_turnover = []
        groups = group.dropna().sort_values().unique().tolist()
        for g in groups:
            to_replace = groups.copy()
            to_replace.remove(g)
            group1 = group.replace(to_replace, 0).unstack()
            turnover = group1.diff().abs().sum(axis=1) / group1.sum(axis=1)
            turnover.name = str(g)
            all_turnover.append(turnover)

        # long short
        to_replace = groups[1:-1]
        group1 = group.replace(to_replace, 0).replace(1, -1).replace(groups[-1], 1).unstack()
        turnover = group1.diff().abs().sum(axis=1) / group1.abs().sum(axis=1)
        turnover.name = '{}-{}'.format(groups[-1], groups[0])
        all_turnover.append(turnover)
        # short long
        group1 = group.replace(to_replace, 0).replace(groups[-1], -1).unstack()
        turnover = group1.diff().abs().sum(axis=1) / group1.abs().sum(axis=1)
        turnover.name = '{}-{}'.format(groups[0], groups[-1])
        all_turnover.append(turnover)

        turnover = pd.concat(all_turnover, axis=1)
        return turnover

    def _cal_summary_stats(self, version_store: VERSION_STORE,
                           factor_data: pd.DataFrame, factor_name, factor_meta):
        basic_stats = self.config['FactorSummary']['stats'].replace(' ', '').split(',')
        summary = factor_data.groupby(level=0).agg(basic_stats)
        summary['positive_rate'] = factor_data.groupby(level=0).aggregate(lambda x: (x > 0).sum() / len(x))
        version_store.write('summary', summary, metadata={'factor_name': factor_name,
                                                          'factor_hash': factor_meta['values_hash'],
                                                          'factor_freq': factor_meta['freq'],
                                                          'factor_meta': factor_meta,
                                                          'upload_time': datetime.datetime.now()})

    def _cal_quantile_backtest(self, version_store: VERSION_STORE,
                               data: pd.DataFrame,
                               factor_name: str,
                               metadata: Dict, force_cal=False):

        universes = self.config['IC']['universe'].replace(' ', '').split(',')
        if 'bins' in self.config['QuantileBacktest']:
            bins = eval(self.config['QuantileBacktest']['bins'])
            for _bin in bins:
                if not self._check_need_cal('mean_ret_bin_{}'.format(_bin), version_store, metadata['label_meta'],
                                            metadata['factor_meta'], force_cal
                                            ):
                    print('{} {} is already tested, skip'.format(factor_name, _bin))
                    continue
                all_mean_ret = []
                all_count_stat = []
                all_turnover = []
                for universe in universes:
                    try:
                        new_data = self._get_universe_data(data, universe)
                        cum_ret, mean_ret, count_stat, std_error_ret = \
                            quantile_backtesting(new_data, factor_name, quantiles=None, bins=_bin, demeaned=False)

                        turnover = self._cal_group_turnover(new_data, factor_name, _bin, None)

                        mean_ret.columns = pd.MultiIndex.from_tuples([(universe, t[1])
                                                                      for t in mean_ret.columns.to_list()])
                        count_stat.columns = pd.MultiIndex.from_tuples([(universe, t[1])
                                                                        for t in count_stat.columns.to_list()])
                        turnover.columns = pd.MultiIndex.from_product([[universe], turnover.columns])
                        if len(mean_ret.columns) < _bin:
                            continue

                        all_mean_ret.append(mean_ret)
                        all_count_stat.append(count_stat)
                        all_turnover.append(turnover)
                    except:
                        print('cannot do quantile backtest')
                if len(all_mean_ret) > 0:
                    mean_ret = pd.concat(all_mean_ret, axis=1)
                    count_stat = pd.concat(all_count_stat, axis=1)
                    turnover = pd.concat(all_turnover, axis=1)
                    version_store.write('mean_ret_bin_{}'.format(_bin), mean_ret, metadata=metadata)
                    version_store.write('count_stat_bin_{}'.format(_bin), count_stat, metadata=metadata)
                    version_store.write('turnover_bin_{}'.format(_bin), turnover, metadata=metadata)

        if 'quantiles' in self.config['QuantileBacktest']:
            quantiles = eval(self.config['QuantileBacktest']['quantiles'])
            for _q in quantiles:
                all_mean_ret = []
                all_count_stat = []
                for universe in universes:
                    new_data = self._get_universe_data(data, universe)
                    cum_ret, mean_ret, count_stat, std_error_ret = \
                        quantile_backtesting(new_data, factor_name, quantiles=_q, bins=None, demeaned=False)
                    mean_ret.columns = pd.MultiIndex.from_product([mean_ret.columns, [universe]])
                    count_stat.columns = pd.MultiIndex.from_product([count_stat.columns, [universe]])
                    all_mean_ret.append(mean_ret)
                    all_count_stat.append(count_stat)
                mean_ret = pd.concat(all_mean_ret, axis=1)
                count_stat = pd.concat(all_count_stat, axis=1)
                version_store.write('mean_ret_q_{}'.format(_q), mean_ret, metadata=metadata)
                version_store.write('count_stat_q_{}'.format(_q), count_stat, metadata=metadata)

    def test_all_factors(self):
        for factor_cap, factor_name in tqdm.tqdm(self.subscribe_factors):
            factor, _ = self.zoo_keeper.get_factor_values(factor_cap, factor_name)
            factor_meta = self.zoo_keeper.get_metadata(factor_cap, factor_name)

            if 'FactorSummary' in self.config:
                lib_name = '{}.{}'.format(self.context, 'sum_' + factor_meta['values_hash'][10:])
                if not self.store.library_exists(lib_name):
                    self.store.initialize_library(lib_name, check_library_count=False)
                version_store = self.store.get_library(lib_name)
                if self._check_factor_need_cal('summary', version_store, factor_meta):
                    self._cal_summary_stats(version_store, factor, factor_name, factor_meta)
                else:
                    print("FactorSummary already calculated for {}... Skip".format(factor_name))

            for label_cat, label_name in self.subscribe_label:
                l = self.label_guardian.get_label_values(label_cat, label_name)
                label_meta = self.label_guardian.get_metadata(label_cat, label_name)

                if self.freq_priority[label_meta['freq']] < self.freq_priority[factor_meta['freq']]:
                    print('label frequency is smaller than factor, skip')
                    continue
                name = self._get_name(label_cat, label_name, factor_cap, factor_name)
                # data = l.to_frame().dropna().join(factor)
                data = panel_df_join(l.to_frame(), factor.to_frame())
                data = data[~data[l.name].isna()]
                lib_name = '{}.{}'.format(self.context, name)
                if not self.store.library_exists(lib_name):
                    self.store.initialize_library(lib_name, check_library_count=False)
                version_store = self.store.get_library(lib_name)  # type: VERSION_STORE
                metadata = {
                    'label_category': label_cat,
                    'label_name': label_name,
                    'label_hash': label_meta['values_hash'],
                    'label_freq': label_meta['freq'],
                    'label_meta': label_meta,
                    'factor_category': factor_cap,
                    'factor_name': factor_name,
                    'factor_hash': factor_meta['values_hash'],
                    'factor_freq': factor_meta['freq'],
                    'factor_meta': factor_meta,
                    'upload_time': datetime.datetime.now()
                }

                if 'IC' in self.config:
                    self._cal_ic(version_store, data, factor_name, metadata)

                if 'QuantileBacktest' in self.config:
                    self._cal_quantile_backtest(version_store, data, factor_name, metadata)

    def _cal_single_factor_ic(self, factor_cap, factor_name,
                              label_cat, label_name
                              ):
        factor, _ = self.zoo_keeper.get_factor_values(factor_cap, factor_name)
        factor_meta = self.zoo_keeper.get_metadata(factor_cap, factor_name)
        l = self.label_guardian.get_label_values(label_cat, label_name)
        label_meta = self.label_guardian.get_metadata(label_cat, label_name)
        if self.freq_priority[label_meta['freq']] < self.freq_priority[factor_meta['freq']]:
            print('label frequency is smaller than factor, skip')
            return
        name = self._get_name(label_cat, label_name, factor_cap, factor_name)
        data = l.to_frame().dropna().join(factor)
        lib_name = '{}.{}'.format(self.context, name)
        if not self.store.library_exists(lib_name):
            self.store.initialize_library(lib_name)
        version_store = self.store.get_library(lib_name)  # type: VERSION_STORE
        metadata = {
            'label_category': label_cat,
            'label_name': label_name,
            'label_hash': label_meta['values_hash'],
            'label_freq': label_meta['freq'],
            'label_meta': label_meta,
            'factor_category': factor_cap,
            'factor_name': factor_name,
            'factor_hash': factor_meta['values_hash'],
            'factor_freq': factor_meta['freq'],
            'factor_meta': factor_meta,
            'upload_time': datetime.datetime.now()
        }
        if 'IC' in self.config:
            self._cal_ic(version_store, data, factor_name, metadata)

    def _cal_single_factor_quantile_backtest(self, factor_cap, factor_name,
                                             label_cat, label_name):
        factor = self.zoo_keeper.get_factor_values(factor_cap, factor_name)
        factor_meta = self.zoo_keeper.get_metadata(factor_cap, factor_name)
        l = self.label_guardian.get_label_values(label_cat, label_name)
        label_meta = self.label_guardian.get_metadata(label_cat, label_name)
        if self.freq_priority[label_meta['freq']] < self.freq_priority[factor_meta['freq']]:
            print('label frequency is smaller than factor, skip')
            return
        name = self._get_name(label_cat, label_name, factor_cap, factor_name)
        data = panel_df_join(l.to_frame().dropna(), factor)
        lib_name = '{}.{}'.format(self.context, name)
        if not self.store.library_exists(lib_name):
            self.store.initialize_library(lib_name)
        version_store = self.store.get_library(lib_name)  # type: VERSION_STORE
        metadata = {
            'label_category': label_cat,
            'label_name': label_name,
            'label_hash': label_meta['values_hash'],
            'label_freq': label_meta['freq'],
            'label_meta': label_meta,
            'factor_category': factor_cap,
            'factor_name': factor_name,
            'factor_hash': factor_meta['values_hash'],
            'factor_freq': factor_meta['freq'],
            'factor_meta': factor_meta,
            'upload_time': datetime.datetime.now()
        }
        if 'QuantileBacktest' in self.config:
            self._cal_quantile_backtest(version_store, data, factor_name, metadata)

    def get_ic(self,
               factor_category: str, factor_name: str,
               label_category: str, label_name: str,
               ):
        name = self._get_name(label_category, label_name, factor_category, factor_name)
        lib_name = '{}.{}'.format(self.context, name)
        if not self.store.library_exists(lib_name) or (not self.store[lib_name].has_symbol('IC')):
            self._cal_single_factor_ic(factor_category, factor_name, label_category, label_name)
        versioned_item = self.store[lib_name].read('IC')  # type: VersionedItem
        ic = versioned_item.data
        meta_data = versioned_item.metadata
        return ic, meta_data

    def get_quantile_backtest(self,
                              factor_category: str, factor_name: str,
                              label_category: str, label_name: str,
                              bins
                              ):
        name = self._get_name(label_category, label_name, factor_category, factor_name)
        lib_name = '{}.{}'.format(self.context, name)
        mean_ret_symbol_name = 'mean_ret_bin_{}'.format(bins)
        count_stat_name = 'count_stat_bin_{}'.format(bins)
        turnover_symbol_name = 'turnover_bin_{}'.format(bins)

        if not self.store.library_exists(lib_name) or (not self.store[lib_name].has_symbol(mean_ret_symbol_name)):
            self._cal_single_factor_quantile_backtest(factor_category, factor_name, label_category, label_name)
        versioned_item = self.store[lib_name].read(mean_ret_symbol_name)  # type: VersionedItem
        mean_ret = versioned_item.data
        meta_data = versioned_item.metadata
        count_stat = self.store[lib_name].read(count_stat_name).data
        turnover = self.store[lib_name].read(turnover_symbol_name).data

        return mean_ret, count_stat, turnover, meta_data

    def get_info(self):
        libs = [l for l in self.store.list_libraries() if l.startswith(self.context)]
        results = []
        for l in libs:
            if 'sum' in l:
                continue
            infos = self.store[l].list_symbols()
            for s in infos:
                metadata = self.store[l].read_metadata(s).metadata
                d = metadata.copy()
                del d['label_meta']
                del d['factor_meta']
                del d['label_hash']
                del d['factor_hash']
                d['stats'] = ','.join(infos)
                results.append(d)
                break
        results = pd.DataFrame(results)
        results = results.sort_values(['label_category', 'label_name',
                                       'factor_category', 'factor_name'
                                       ])
        results['factor_cat_name'] = results.apply(lambda x: x['factor_category'] + '/' + x['factor_name'], axis=1)
        results['label_cat_name'] = results.apply(lambda x: x['label_category'] + '/' + x['label_name'], axis=1)
        return results

    def get_ic_comparison(self,
                          factor_to_compare: List[Tuple[str, str]],
                          label_category: str, label_name: str,
                          start_date=None,
                          end_date=None):
        to_compare = []
        for t in factor_to_compare:
            ic, metadata = self.get_ic(t[0], t[1], label_category, label_name)
            ic.columns = pd.MultiIndex.from_product([['{}_{}'.format(t[0], t[1])]] + ic.columns.levels,
                                                    names=['factor', 'label', 'universe'])
            to_compare.append(ic)
        to_compare = pd.concat(to_compare, axis=1)
        if start_date is not None:
            to_compare = to_compare.loc[pd.to_datetime(start_date):]

        if end_date is not None:
            to_compare = to_compare.loc[: pd.to_datetime(end_date)]
        return to_compare

    def remove_factor_data(self):
        libs = [l for l in self.store.list_libraries() if l.startswith(self.context)]
        results = []
        for l in libs:
            if 'sum' in l:
                continue
            infos = self.store[l].list_symbols()
            for s in infos:
                metadata = self.store[l].read_metadata(s).metadata


if __name__ == '__main__':
    local_cfg_path = '../../cfg/factor_keeper_setting.ini'
    keeper = ZooKeeper(local_cfg_path)
    guardian = LabelGuardian('../../cfg/label_guardian_setting.ini')
    coach = PerformanceCoach('../../cfg/performance_coach_setting.ini', keeper, guardian, {})
    # coach.get_info()
    # coach.test_all_factors()