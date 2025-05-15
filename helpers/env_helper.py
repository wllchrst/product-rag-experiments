import os
from dotenv import load_dotenv

ENVS = ["GEMINI_API_KEY", "MODEL_PATH", "MLFLOW_URL"]

class EnvHelper:
    """Class for gathering and saving all env for the application """
    def __init__(self):
        load_dotenv(dotenv_path='.env')
        self.envs = {}

        self.gather_envs()
        self.assign_env()

    def gather_envs(self) -> bool:
        """Gather All env for the application if there is a missing value throws error

        Returns:
            bool: _description_
        """
        for env in ENVS:
            env_value = os.getenv(env)
            if env_value is None:
                raise ValueError(f'{env} has value None')

            self.envs[env] = os.getenv(env)

        return True
    
    def assign_env(self):
        self.GEMINI_API_KEY = self.envs[ENVS[0]]
        self.MODEL_PATH = self.envs[ENVS[1]]
        self.MLFLOW_URL = self.envs[ENVS[2]]

env_helper = EnvHelper()