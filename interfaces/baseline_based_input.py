from dataclasses import dataclass
from interfaces.product_information import ProductInformation
from interfaces.review import Review
from typing import List

@dataclass
class BaselineBasedInput:
    product_information: ProductInformation
    reviews: List[Review]