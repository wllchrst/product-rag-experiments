from abc import ABC, abstractmethod
from llm import GeminiLLM
from interfaces import AgentConfig
from helpers import configuration_helper

class BaseAgent(ABC):
    def __init__(self, config_key: str = None):
        super().__init__()
        self.llm = GeminiLLM()
        if config_key is None:
            raise ValueError("config_key cannot be None")

        configuration = configuration_helper.configs[config_key]
        self.config = AgentConfig(**configuration)
    
    def format_config(self):
        format = f"""
Role: {self.config.role}
Goal: {self.config.goal}
Backstory: {self.config.backstory}

"""
        return format

    @abstractmethod
    def execute_task(self, data: dict):
        """
        Executes the task assigned to the agent.
        """
        pass