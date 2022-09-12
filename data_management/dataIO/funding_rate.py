from arctic import Arctic
from data_management.dataIO.utils import read_arctic_version_store


def get_funding_rates(store: Arctic, exchange, symbol_type='u_perp'):
    data = read_arctic_version_store(store, '{}_{}.funding_rate'.format(exchange, symbol_type), None)
    data['funding_rate'] = data['funding_rate'].astype(float)
    data['exchange'] = exchange
    return data

if __name__ == '__main__':
    df = get_funding_rates(Arctic('localhost'), 'okx')
    print(df)