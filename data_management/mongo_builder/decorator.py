import datetime
import time
from functools import wraps
import traceback

import jqdatasdk
import tushare as ts
from requests import ConnectTimeout
from thriftpy2.transport import TTransportException

from data_management.tokens import jq_user, jq_password, token


def joinquant_retry(f):
    @wraps(f)
    def f_retry(*args, **kwargs):
        while True:
            try:
                jqdatasdk.auth(jq_user, jq_password)
                break
            except:
                pass
        i = 0
        while True:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if '您的账号最多只能开启' in str(e):
                    time.sleep(1)
                    while True:
                        try:
                            jqdatasdk.auth(jq_user, jq_password)
                            break
                        except:
                            pass
                elif isinstance(e, TTransportException) and e.type == 4:
                    # if e.type == 4
                    print(e, args)
                    # traceback.print_tb(e.__traceback__)
                    return None
                elif 'only a table is allowed to query every time' in str(e):
                    time.sleep(1)
                    while True:
                        try:
                            jqdatasdk.auth(jq_user, jq_password)
                            break
                        except:
                            pass
                elif 'count 必须是大于0的整数' in str(e):
                    return None
                else:
                    print(e, args)
                    traceback.print_tb(e.__traceback__)
                    if i == 10:
                        print('retry 10 times, return None')
                        return None
                    i += 1

    return f_retry


def tushare_retry(f):
    @wraps(f)
    def f_retry(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except ConnectTimeout:
                time.sleep(3)
            except Exception as e:
                if '抱歉' in str(e):
                    print(e)
                    now = datetime.datetime.now()
                    sec = 60 - now.second
                    print('so sleep {} sec'.format(sec))
                    time.sleep(sec)
                else:
                    print(e)
                    # break

    return f_retry
