from interfaces import EmotionBasedInput, ConclusionBasedInput
from handler.data_handler import DataHandler
from handler.classification_handler import ClassificationHandler
from handler.evaluation_handler import EvaluationHandler
from agents import EmotionBasedAgent, ConclusionAgent, WebAgent
from dataclasses import asdict
from collections import defaultdict
from interfaces.product_information import ProductInformation

gts = [
    """- The packaging needs to be improved because there are still many damaged ones
- The product needs to be checked again or improved because there are many malfunctions
- The seller needs to pay more attention to the products sent, to prevent shipping errors
- The placement of the warranty card needs to be improved, because the current one makes it prone to damage""",

    """- inconsistent delivery, some are slow, some are fast
- unsafe packaging, so the box is dented and dirty
- wrong product sent (color)
- placement of the warranty card needs to be reconsidered so that it is not easily torn and is outside the box
- product quality is poor and needs to be given more attention"""
]

class AgentHandler:
    def __init__(self):
        self.data_handler = DataHandler()
        self.classification_handler = ClassificationHandler()
        self.evaluation_handler = EvaluationHandler()
        
        self.emotion_based_agent = EmotionBasedAgent(config_key='emotion_based_config')
        self.conclusion_agent = ConclusionAgent(config_key='conclusion_based_config')
        self.web_agent =  WebAgent('web_based_config')

    def test_evaluation_agent(self):
        """
        Test the agents by running them with sample data.
        """
        reviews, product_information  = self.data_handler.get_dummy_data()
        conclusions = []
        
        for review_data in reviews:
            ebi = EmotionBasedInput(
                product_information=product_information,
                reviews=review_data
            )

            conclusion = self.emotion_based_agent.execute_task(data=asdict(ebi))
            conclusions.append(conclusion)
        
        final_answer = self.conclusion_agent.execute_task(data=asdict(ConclusionBasedInput(conclusions=conclusions)))        

        return final_answer
    
    def evaluate_product(self, product_name: str):
        reviews, product_information = self.data_handler.get_data(product_name=product_name)
        reviews = self.classification_handler.assign_emotion(reviews)
        grouped_reviews = defaultdict(list)
        
        for review in reviews:
            grouped_reviews[review.emotion].append(review)
        
        product_search = self.web_agent.execute_task(asdict(product_information))
        
        conclusions = []
        for grouped_review in grouped_reviews.values():
            ebi = EmotionBasedInput(
                product_information=product_information,
                reviews=grouped_review
            )
            
            conclusion = self.emotion_based_agent.execute_task(data=asdict(ebi))
            conclusions.append(conclusion)

        conclusion_input = ConclusionBasedInput(conclusions=conclusions, product_search=product_search)
        
        final_answer = self.conclusion_agent.\
            execute_task(data=asdict(conclusion_input))

        self.evaluation_handler.evaluate(
            product_information=product_information,
            ground_truths=gts,
            prediction=final_answer,
            product_search=product_search
        )

        return final_answer