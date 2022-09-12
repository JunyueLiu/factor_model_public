from enum import Enum
from typing import Optional

import pandas as pd
from attr import dataclass


class OrderStatus(Enum):
    PENDING = 'PENDING'
    FILL = 'FILL'
    CANCEL = 'CANCEL'

class OrderType(Enum):
    MARKET = "MARKET"
    ALGO = 'ALGO'
    LIMIT = 'LIMIT'


@dataclass
class Order:
    create_time: pd.Timestamp
    update_time: pd.Timestamp
    status: OrderStatus
    order_type: OrderType
    price: float
    volume: float
    execution_price: Optional[float]
    fill_volume: Optional[float]

    def __str__(self) -> str:
        return self.__dict__



