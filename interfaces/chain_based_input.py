from dataclasses import dataclass
from interfaces import ProductInformation

@dataclass
class ChainBasedInput:
    reviews: list[str]
    product_search: str
    product_information: ProductInformation
    previous_evaluation: str