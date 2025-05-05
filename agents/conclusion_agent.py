from agents.base_agent import BaseAgent
from interfaces import ConclusionBasedInput

class ConclusionAgent (BaseAgent):
    def __init__(self, config_key: str):
        super().__init__(config_key=config_key)
        
    def format_input(self, input: ConclusionBasedInput) -> str:
        formatted_config = self.format_config()

        formatted_input = f"""
{formatted_config}

Below are the conclusions based on the reviews provided:
"""
        for i, conclusion in enumerate(input.conclusions):
            formatted_input += f"""
Conclusion {i + 1}:
{conclusion}
"""

        return formatted_input

    def execute_task(self, data: dict) -> str:
        """
        Execute the task of generating a conclusion based on the provided data
        """
        input = ConclusionBasedInput(**data)
        formatted_input = self.format_input(input)

        answer = self.llm.answer(formatted_input)
        
        return answer