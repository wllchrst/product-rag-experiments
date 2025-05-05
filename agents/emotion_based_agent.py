from agents.base_agent import BaseAgent
from interfaces import EmotionBasedInput

class EmotionBasedAgent(BaseAgent):
    def __init__(self, config_key: str = None):
        super().__init__(config_key=config_key)
    
    def format_input(self, input: EmotionBasedInput):
        product_info = f"""
Name: {input.product_information['name']}
Description: {input.product_information['description']}
Price: {input.product_information['price']}
Rating: {input.product_information['overall_rating']}

Reviews:\n
"""
        product_reviews = ''
        for review in input.reviews:
            product_reviews += f"- {review['review']} (Rating: {review['rating']}, Emotion: {review['emotion']})\n"
        
        product_info += product_reviews
        return product_info

    def execute_task(self, data: dict):
        """
        Executes the task assigned to the agent.
        """
        input = EmotionBasedInput(**data)

        formatted_config = self.format_config()
        formatted_input = self.format_input(input)

        input = formatted_config + formatted_input
        
        answer = self.llm.answer(input)
        return answer
        