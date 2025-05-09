from transformers import AutoModelForSequenceClassification, AutoTokenizer
from helpers import env_helper
from interfaces import Review
from typing import List
import torch
class ClassificationHandler:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(env_helper.MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(env_helper.MODEL_PATH)
        self.mapping = {
            0: "love",
            1: "anger",
            2: "happy",
            3: "fear",
            4: "sadness",
        }

        self.move_model()

    def move_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax().item()
            # print(f'{text} - {self.mapping[predicted_class]}')
            return self.mapping[predicted_class]
    
    def assign_emotion(self, reviews: List[Review]) -> List[Review]:
        assigned_reviews = []
        for review in reviews:
            review.emotion = self.predict(review.review)
            assigned_reviews.append(review)
        
        return assigned_reviews