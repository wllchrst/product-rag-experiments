from evaluate import load
from interfaces import ProductInformation
from typing import List
from handler.logging_handler import LoggingHandler
class EvaluationHandler:
    def __init__(self):
        self.logging_handler = LoggingHandler()

        self.rouge = load("rouge")
        self.bertscore = load("bertscore")

    def evaluate(self, product_information: ProductInformation, ground_truths: List[str], prediction: str):
        for ground_truth in ground_truths:
            print(ground_truth)
            results_rouge = self.rouge.compute(predictions=[prediction], references=[ground_truth])
            results_bert = self.bertscore.compute(predictions=[prediction], references=[ground_truth], lang="en")

            print(f"Rouge: {results_rouge}")
            print(f"BertScore: {results_bert}")

            self.logging_handler.log(
                product_information,
                ground_truths,
                prediction,
                results_rouge,
                results_bert
            )