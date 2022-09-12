from enum import Enum


class SymbolType(Enum):
    SPOT = 'spot'
    USDT_PERP = 'u_perp'
    INVERSE_PERP = 'c_perp'
    FUTURES = 'futures'
    USD_PERP = 'u_perp'


class BarTimeFrame(Enum):
    m1 = '1m'
    m3 = '3m'
    m5 = '5m'
    m15 = '15m'
    m30 = '30m'
    h1 = '1h'
    h2 = '2h'
    h4 = '4h'
    h6 = '6h'
    h8 = '8h'
    d1 = '1d'
