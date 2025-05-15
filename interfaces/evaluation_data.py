from dataclasses import dataclass
from interfaces import ProductInformation
from typing import List
@dataclass
class EvaluationData:
    """
    Data class for evaluation data.
    """
    product_information: ProductInformation
    prediction_result: str
    ground_truths: str
    rogue: dict
    bert: dict