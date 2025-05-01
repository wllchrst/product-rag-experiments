from dataclasses import dataclass

@dataclass
class Review:
    review: str
    rating: str
    emotion: str