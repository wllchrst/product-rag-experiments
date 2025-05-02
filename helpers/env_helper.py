import os
from dotenv import load_dotenv

ENVS = ["GEMINI_API_KEY"]

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
        print(self.GEMINI_API_KEY)

env_helper = EnvHelper()