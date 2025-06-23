from llm.base_llm import BaseLLM
# Use a pipeline as a high-level helper
from transformers import pipeline

class DeepseekLLM(BaseLLM):
    def __init__(self):
        super().__init__()
        self.pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)

    def answer(self, query: str):
        messages = [
            {"role": "user", "content": query},
        ]
        
        return self.pipe(messages)