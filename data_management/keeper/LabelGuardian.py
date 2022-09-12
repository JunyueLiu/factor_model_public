from typing import Union

from data_management.keeper.Keeper import Keeper

import pandas as pd


class LabelGuardian(Keeper):
    def __init__(self, config_path):
        super(LabelGuardian, self).__init__(config_path)

    def get_label_values(self, category: str, label_name: str, start_date=None, end_date=None, local_factor=True):
        label_values, d = self.get_values_and_meta(category, label_name, start_date, end_date, local_factor)
        value_type = d['values_type']
        if value_type == 'cross-section':
            label_values = label_values.stack()
            label_values.name = label_name
        return label_values

    def save_label_values(self, category: str,
                          label_values: pd.Series,
                          to_arctic: bool = True, **meta_kwargs):

        if label_values.name is None:
            raise ValueError('Must provide factor name. Use factor_name parameter or set Series name')
        else:
            factor_name = str(label_values.name)

        self.save_values_and_meta(category, factor_name, label_values, to_arctic, **meta_kwargs)


if __name__ == '__main__':
    guardian = LabelGuardian('../../cfg/label_guardian_setting.ini')
    l = guardian.get_label_values('regression', 'forward_1_M_hold_return_eliminate_paused_and_limit_ret')
