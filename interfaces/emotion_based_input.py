from dataclasses import dataclass
from interfaces import ProductInformation, Review
from typing import List
@dataclass
class EmotionBasedInput:
    product_information: ProductInformation
    reviews: List[Review]

"""
product_information: ProductInformation = ProductInformation(
    name="Sample Product",
    description="This is a sample product.",
    price=19.99,
    overall_rating=4.5
)

reviews: List[Review] = [
    Review(
        review="This product is amazing!",
        rating="5",
        emotion="joy"
    ),
    Review(
        review="Not what I expected.",
        rating="2",
        emotion="anger"
    ),
]



ebi = EmotionBasedInput(
    product_information=product_information,
    reviews=reviews
)
"""
