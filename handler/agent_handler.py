from interfaces import EmotionBasedInput, ConclusionBasedInput, BaselineBasedInput, Review
from handler.data_handler import DataHandler
from handler.classification_handler import ClassificationHandler
from handler.evaluation_handler import EvaluationHandler
from agents import EmotionBasedAgent, ConclusionAgent, WebAgent, ChainAgent, BaselineAgent
from dataclasses import asdict
from collections import defaultdict
from interfaces.product_information import ProductInformation
from interfaces import ChainBasedInput
from agents.chain_agent import ChainAgent
from typing import List

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
        self.chain_agent = ChainAgent(config_key='chain_based_config')
        self.web_agent =  WebAgent('web_based_config')
        self.baseline_agent = BaselineAgent(config_key='baseline_based_config')

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
    
    def evaluate_product(self, product_name: str, type: str='parallel'):
        reviews, product_information = self.data_handler.get_data(product_name=product_name)
        reviews = self.classification_handler.assign_emotion(reviews)
        grouped_reviews = defaultdict(list)
        ground_truths = self.gather_ground_truths(product_name=product_information.name)
        
        for review in reviews:
            grouped_reviews[review.emotion].append(review)
        
        product_search = self.web_agent.execute_task(asdict(product_information))
        
        final_answer = ''
        if type == 'parallel':
            final_answer = self.parallelization(
                product_search=product_search, 
                grouped_reviews=grouped_reviews, 
                product_information=product_information,
                ground_truths=ground_truths
            )
        elif type == 'chaining':
            final_answer = self.prompt_chaining(
                product_search=product_search, 
                grouped_reviews=grouped_reviews, 
                product_information=product_information,
                ground_truths=ground_truths
            )
        elif type == 'baseline':
            final_answer = self.baseline_evaluation(
                product_information=product_information, 
                reviews=reviews, 
                ground_truths=ground_truths
            )

        return final_answer
    
    def gather_ground_truths(self, product_name: str) -> list[str]:
        """
        Gather ground truths for the product.
        """
        user_reviews = self.data_handler.get_user_reviews(product_name=product_name)
        ground_truths = [review.review for review in user_reviews]

        return ground_truths
    
    def baseline_evaluation(self, product_information: ProductInformation, reviews: List[Review], ground_truths: List[str]) -> str:
        """
        Evaluate the product using baseline evaluation.
        """
        baseline_input = BaselineBasedInput(
            product_information=product_information,
            reviews=reviews
        )
        
        result = self.baseline_agent.execute_task(data=asdict(baseline_input))

        self.evaluation_handler.evaluate(
            product_information=product_information,
            ground_truths=ground_truths,
            prediction=result,
            product_search=product_information.name,
            method="baseline"
        )

        return result

    def prompt_chaining(self,
                        product_search: str, 
                        grouped_reviews: defaultdict[any, list], 
                        product_information: ProductInformation,
                        ground_truths: List[str]) -> str:
        previous_evaluation = ''
        evaluations = []
        for grouped_review in grouped_reviews.values():
            cbi = ChainBasedInput(
                reviews=grouped_review,
                product_search=product_search,
                product_information=product_information,
                previous_evaluation=previous_evaluation
            )
            
            result= self.chain_agent.execute_task(data=asdict(cbi))
            previous_evaluation += result
            evaluations.append(result)
        
        final_result = self.chain_agent.format_all_evaluation(evaluations)

        self.evaluation_handler.evaluate(
            product_information=product_information,
            ground_truths=ground_truths,
            prediction=final_result,
            product_search=product_search,
            method="chaining"
        )

        return final_result
    
    def parallelization(self,
                        product_search: str, 
                        grouped_reviews: defaultdict[any, list], 
                        product_information: ProductInformation,
                        ground_truths: list[str]) -> str:
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
            ground_truths=ground_truths,
            prediction=final_answer,
            product_search=product_search,
            method="parallel"
        )
        
        return final_answer