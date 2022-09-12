import configparser
import datetime
import os

import jqdatasdk
from arctic import Arctic

from data_management.dataIO.utils import get_arctic_store
from data_management.tokens import jq_user, jq_password


def build_trading_calendar(arctic_store: Arctic):
    jqdatasdk.auth(jq_user, jq_password)
    trading_date = jqdatasdk.get_all_trade_days()
    last_update = datetime.datetime.now()
    if arctic_store.library_exists('trading_calendar'):
        arctic_store['trading_calendar'].delete('AShares')
        arctic_store['trading_calendar'].write('AShares', trading_date,
                                               metadata={'last_update': last_update, 'source': 'joinquant',
                                                         })
    else:
        arctic_store.initialize_library('trading_calendar')
        arctic_store['trading_calendar'].write('AShares', trading_date,
                                               metadata={'last_update': last_update, 'source': 'joinquant',
                                                         })

if __name__ == '__main__':
    store = Arctic('localhost')
    arctic_config_path = '../../cfg/arctic_hk.ini'
    arctic_config = configparser.ConfigParser()
    arctic_config.read(os.path.abspath(arctic_config_path))
    store = get_arctic_store(arctic_config)
    build_trading_calendar(store)