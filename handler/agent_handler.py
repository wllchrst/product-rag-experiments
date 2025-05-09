from interfaces import EmotionBasedInput, ConclusionBasedInput
from handler.data_handler import DataHandler
from handler.classification_handler import ClassificationHandler
from agents import EmotionBasedAgent, ConclusionAgent
from dataclasses import asdict
from handler.classification_handler import ClassificationHandler
from collections import defaultdict
class AgentHandler:
    def __init__(self):
        self.data_handler = DataHandler()
        self.classification_handler = ClassificationHandler()
        
        self.emotion_based_agent = EmotionBasedAgent(config_key='emotion_based_config')
        self.conclusion_agent = ConclusionAgent(config_key='conclusion_based_config')

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
        
        conclusions = []
        for grouped_review in grouped_reviews.values():
            ebi = EmotionBasedInput(
                product_information=product_information,
                reviews=grouped_review
            )
            
            conclusion = self.emotion_based_agent.execute_task(data=asdict(ebi))
            conclusions.append(conclusion)
        
        final_answer = self.conclusion_agent.execute_task(data=asdict(ConclusionBasedInput(conclusions=conclusions)))        

        return final_answer