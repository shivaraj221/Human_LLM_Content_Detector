# Runs at build time so the model is cached before the service starts.
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "ShantanuT01/gradient-ai-text-detector"
AutoTokenizer.from_pretrained(MODEL_ID)
AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
print("Model cached.")
