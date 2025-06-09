from agents.base_agent import BaseAgent
from interfaces import ChainBasedInput

class ChainAgent(BaseAgent):
    """
    ChainAgent is responsible for executing a series of tasks in a chain.
    It inherits from BaseAgent and implements the execute_task method.
    """
    
    def __init__(self, config_key: str = None):
        super().__init__(config_key)
    
    def format_input(self, input: ChainBasedInput) -> str:
        """
        Formats the input for the agent based on the ChainBasedInput structure.
        """
        config = self.format_config()
        reviews = ''
        
        for review in input.reviews:
            reviews += f"- {review['review']}\n"
        
        previous_evaluation = input.previous_evaluation if input.previous_evaluation == '' else f"Previous Evaluation\n{input.previous_evaluation}"
        
        formatted_input = f"""
{config}
Product Name: {input.product_information['name']}

Product Information:
{input.product_search}

{previous_evaluation}

Previous Evaluation:
{input.previous_evaluation}

Reviews:
{reviews}
        """

        return formatted_input
    
    def execute_task(self, data: dict):
        """
        Executes the task assigned to the agent by chaining multiple agents.
        """
        chain_data = ChainBasedInput(**data)
        
        formatted_input = self.format_input(chain_data)

        print("-"*50)
        print(formatted_input)
        return self.llm.answer(formatted_input)