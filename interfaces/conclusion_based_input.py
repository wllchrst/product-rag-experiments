from dataclasses import dataclass
from typing import List
@dataclass
class ConclusionBasedInput:
    conclusions: List[str]
    product_search: str