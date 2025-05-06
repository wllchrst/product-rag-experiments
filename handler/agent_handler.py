from interfaces import EmotionBasedInput, ConclusionBasedInput
from handler.data_handler import DataHandler
from agents import EmotionBasedAgent, ConclusionAgent
from dataclasses import asdict
class AgentHandler:
    def __init__(self):
        self.data_handler = DataHandler()
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