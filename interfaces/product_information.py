from dataclasses import dataclass

@dataclass
class ProductInformation:
    name: str
    description: str
    price: float
    overall_rating: float