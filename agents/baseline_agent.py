from agents.base_agent import BaseAgent
from interfaces import BaselineBasedInput

class BaselineAgent(BaseAgent):
    """
    """
    
    def __init__(self, config_key: str = None):
        super().__init__(config_key)
    
    def format_input(self, input: BaselineBasedInput) -> str:
        """
        Formats the input for the agent based on the BaselineBasedInput structure.
        """
        config = self.format_config()
        reviews = ''
        
        for review in input.reviews:
            reviews += f"- {review['review']}\n"
        
        formatted_input = f"""
{config}
Product Information:
- Name: {input.product_information['name']}
- Description: {input.product_information['description']}
- Price: {input.product_information['price']}

Reviews:
{reviews}
        """
        return formatted_input

    def execute_task(self, data: dict) -> str:
        input = BaselineBasedInput(**data)
        
        formatted_input = self.format_input(input)
        
        return self.llm.answer(formatted_input)