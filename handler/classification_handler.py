from transformers import AutoModelForSequenceClassification, AutoTokenizer
from helpers import env_helper
import torch
class ClassificationHandler:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(env_helper.MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(env_helper.MODEL_PATH)
        self.mapping = {
            0: "love",
            1: "anger",
            2: "happy",
            3: "feat",
            4: "sadness",
        }

        self.move_model()

    def move_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax().item()
            print("Predicted class:", self.mapping[predicted_class])