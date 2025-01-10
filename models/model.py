from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
MAX_TOKENS=2048
MODEL_NAME = "your_model_name"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def classify_claim(claim, top_k_claims):
    inputs = tokenizer([claim] * len(top_k_claims), 
                       top_k_claims, 
                       max_length=MAX_TOKENS, padding=True, 
                       truncation=True, return_tensors="pt")
    # Move inputs to the device of the model (e.g., GPU)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1].tolist()  # Probability of class 1
    # probabilities = [0.99] * len(top_k_claims)
    return list(zip(top_k_claims, probabilities))
