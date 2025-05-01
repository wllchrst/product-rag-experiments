from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def execute_task(self):
        """
        Executes the task assigned to the agent.
        """
        pass