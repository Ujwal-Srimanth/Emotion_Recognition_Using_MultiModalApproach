import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_type = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModelForSequenceClassification.from_pretrained("saved_model", trust_remote_code=True)

class_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    sentiment = class_labels[predicted_class]
    return sentiment

st.title("Sentiment Analysis with BERT")
st.write("This application classifies text into one of six sentiment categories: sadness, joy, love, anger, fear, or surprise.")

text = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if text.strip():
        sentiment = predict_sentiment(text)
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.write("Please enter some text to analyze.")
