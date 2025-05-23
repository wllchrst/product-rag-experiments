from evaluate import load
from interfaces import ProductInformation, EvaluationData
from typing import List
from handler.logging_handler import LoggingHandler
class EvaluationHandler:
    def __init__(self):
        self.logging_handler = LoggingHandler()

        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        self.bleu = load("bleu")

    def evaluate(self, product_information: ProductInformation, ground_truths: List[str], prediction: str, product_search: str):
        # Rouge and Bleu simply does a comparation between the prediction and the ground truth without using any knowledge like BERT does.
        results_rouge = self.rouge.compute(predictions=[prediction], references=[ground_truths])
        results_bleu = self.bleu.compute(predictions=[prediction], references=[ground_truths])
        results_bert = self.bertscore.compute(predictions=[prediction], references=[ground_truths], lang="en")

        print(f"Rouge: {results_rouge}")
        print(f"BertScore: {results_bert}")
        print(f'Bleu: {results_bleu}')

        evaluation_data = EvaluationData(
            product_information=product_information,
            ground_truths=ground_truths,
            prediction_result=prediction,
            rogue=results_rouge,
            bert=results_bert,
            bleu=results_bleu,
            product_search=product_search
        )

        self.logging_handler.log(
            evaluation_data=evaluation_data
        )