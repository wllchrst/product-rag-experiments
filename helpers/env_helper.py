import os
from dotenv import load_dotenv

ENVS = ["GEMINI_API_KEY"]

class EnvHelper:
    """Class for gathering and saving all env for the application """
    def __init__(self):
        load_dotenv()
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
        self.SESSION_PPT_COLLECTION = self.envs[ENVS[0]]

env_helper = EnvHelper()