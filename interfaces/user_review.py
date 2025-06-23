from dataclasses import dataclass

@dataclass
class UserReview:
    user_name: str
    product_name: str
    review: str