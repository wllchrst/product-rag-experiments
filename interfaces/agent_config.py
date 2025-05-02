from dataclasses import dataclass
@dataclass
class AgentConfig:
    role: str
    goal: str
    backstory: str