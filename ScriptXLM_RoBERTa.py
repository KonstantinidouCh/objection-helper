from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from CustomXLMRoberta import CustomXLMRoberta
from transformers import XLMRobertaModel

# ================================
# Load Data
# ================================
data = pd.read_csv('data/Laws.csv')

# Process the "Law Used" column to extract full legal articles
data['Laws Used'] = data['Laws Used'].apply(lambda x: eval(x) if isinstance(x, str) else [])
data['Laws Used'] = data['Laws Used'].apply(lambda x: [law.strip() for law in x])

# Combine all unique laws into a list
unique_laws = list(set([law for laws in data['Laws Used'] for law in laws]))
unique_laws = sorted(unique_laws)  # Optional: Sort alphabetically


# Tokenize and predict using the loaded model
def predict_with_loaded_model(text):
    """
    Predicts the label for a given text using the loaded model and tokenizer.
    """
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        dictum_logits, laws_logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
        
        # Get probabilities using softmax
        probabilities = torch.softmax(dictum_logits, dim=1).cpu().numpy()[0]
        
        # Get predicted label (class with highest probability)
        predicted_label = torch.argmax(torch.tensor(probabilities)).item()
        
        return predicted_label, laws_logits, probabilities

    


def tokenize_function(text):
    """
    Tokenizes a text input using the BERT tokenizer with padding and truncation to handle long text.
    """
    return tokenizer(
        text, padding='max_length', truncation=True, max_length=512, return_tensors="pt"
    )
    

# ================================
# Prediction with Top 5 Laws (Interpretability (1))
# ================================
def predict_with_top_5_laws(text):
    inputs = tokenize_function(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        dictum_logits, laws_logits, _ = model(inputs['input_ids'], inputs['attention_mask'])
        dictum_prediction = torch.argmax(dictum_logits, dim=1).item()
        laws_probabilities = torch.sigmoid(laws_logits).cpu().numpy()[0]
        top_5_indices = np.argsort(laws_probabilities)[-5:][::-1]
        top_5_probs = laws_probabilities[top_5_indices]
        
        # Normalize the top 5 probabilities so they sum up to 1
        top_5_probs /= top_5_probs.sum()
        
        top_5_laws = [unique_laws[i] for i in top_5_indices]
        top_5 = list(zip(top_5_laws, top_5_probs))
    return dictum_prediction, top_5



# ================================
# Prediction with Top 5 Words (Interpretability (2))
# Prediction with Top 3 Sentences (Interpretability (3))

# The attention score is a numerical value that reflects how much focus or
# "attention" a model assigns to specific tokens (words, subwords, or symbols)
# in an input sequence when generating an output or making a prediction.
# ================================
def predict_with_top_5_words_and_sentences(text):
    inputs = tokenize_function(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        dictum_logits, laws_logits, attentions = model(inputs['input_ids'], inputs['attention_mask'])
        dictum_prediction = torch.argmax(dictum_logits, dim=1).item()
        laws_probabilities = torch.sigmoid(laws_logits).cpu().numpy()[0]
        top_5_indices = np.argsort(laws_probabilities)[-5:][::-1]
        top_5_probs = laws_probabilities[top_5_indices]

        # Normalize top_5_probs so they sum to 1
        top_5_probs /= top_5_probs.sum()

        top_5_laws = [unique_laws[i] for i in top_5_indices]
        top_5_laws_with_probs = list(zip(top_5_laws, top_5_probs))
        
        # Extract attention scores from the last layer
        last_layer_attention = attentions[-1]  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = last_layer_attention.mean(dim=1)  # Average over all heads: (batch_size, seq_len, seq_len)
        cls_attention_scores = attention_scores[:, 0, :]  # CLS token's attention: (batch_size, seq_len)
        
        # Map attention scores to words and filter special tokens
        input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        word_attention_scores = cls_attention_scores[0].cpu().numpy()
        word_attention_pairs = [
            (token, score) for token, score in zip(input_tokens, word_attention_scores) 
            if token not in ["<s>", "</s>", "<pad>", ".", ",", "(", ")", "!", "?"] and not any(char.isdigit() for char in token)
        ]
        
        # Combine tokens with the same word (case insensitive)
        word_attention_dict = {}
        for token, score in word_attention_pairs:
            token_lower = token.lower()
            if token_lower in word_attention_dict:
                word_attention_dict[token_lower] += score
            else:
                word_attention_dict[token_lower] = score
        
        # Sort words by attention score
        top_5_words = sorted(word_attention_dict.items(), key=lambda x: x[1], reverse=True)[:5]

        # Normalize word_attention_scores so they sum to 1
        total_word_attention = sum(score for _, score in top_5_words)
        top_5_words = [(word, score / total_word_attention) for word, score in top_5_words]

        # Compute attention scores for sentences
        sentences = text.split(".")  # Simple sentence splitting by periods
        sentence_scores = []
        for sentence in sentences:
            if sentence.strip():
                tokens = tokenizer.tokenize(sentence.strip())
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                sentence_attention = np.mean([
                    word_attention_scores[i]
                    for i, token_id in enumerate(inputs['input_ids'][0].cpu().numpy())
                    if token_id in token_ids
                ])
                sentence_scores.append((sentence.strip(), sentence_attention))

        # Normalize sentence_scores so they sum to 1
        total_sentence_attention = sum(score for _, score in sentence_scores)
        sentence_scores = [(sentence, score / total_sentence_attention) for sentence, score in sentence_scores]
        
        # Sort sentences by attention scores
        top_3_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:3]
    
    return dictum_prediction, top_5_laws_with_probs, top_5_words, top_3_sentences

# ================================
# Set device (GPU if available, otherwise CPU)
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Load XLM-RoBERTa model and tokenizer
# ================================
# Load the model state dictionary
import os
import gdown
import streamlit as st

# File details
model_load_path = "XLM-RoBERTa.pth"

@st.cache_resource
def download_and_load_model():
    # Check if the file already exists
    if not os.path.isfile(model_load_path):
        url = f"https://drive.google.com/uc?id=1Yj-TSKZNkQwCOLqWId2Ehgu-JtC9viLw"
        gdown.download(url, model_load_path, quiet=False)
    
    # Load the model
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    return model


model = CustomXLMRoberta("xlm-roberta-base", len(unique_laws))
# Load model
model = download_and_load_model()
model.to(device)
print("Model loaded successfully.")

# Load the tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("XLMRobertaTokenizer.json")
print("Tokenizer loaded successfully.")


model.to(device)

# ================================
# Predict with the loaded model
# ================================

# ----------------------------------------------------------------
# This can be modified to use the loaded model for prediction
# ----------------------------------------------------------------
NEW_OBJECTION = """Hierbij maak ik bezwaar tegen de mij opgelegde boete van vijftig euro voor het verkeerd aanbieden van afval op vijftien januari 2024.

Op de ochtend van vijftien januari heb ik een grote doos karton naast de papiercontainer gezet, omdat deze vol was. Ik heb de doos netjes naast de container geplaatst en was van plan om hem later op de dag weg te brengen als de container geleegd was. Helaas heb ik niet op een bordje gelet dat aangaf dat het afval op die plek niet geplaatst mocht worden.

Ik vind de boete onterecht, omdat ik ervan uitging dat het geen probleem was om de doos tijdelijk naast de volle container te zetten. Ik heb altijd mijn best gedaan om mijn afval goed te scheiden en heb nog nooit eerder een boete gehad.

Bijgevoegd vindt u een foto van de volle papiercontainer op de dag van de overtreding.

Ik verzoek u vriendelijk de boete te annuleren."""

# Test the model with a new objection
predicted_label, predicted_laws, predicted_probabilities = predict_with_loaded_model(NEW_OBJECTION)
print(f"Predicted label: {predicted_label} (1: Gegrond, 0: Ongegrond)")
print(f"Probabity (Ongegrond): {predicted_probabilities[0]}\nProbabity (Gegrond): {predicted_probabilities[1]}")

prediction, top_5_laws, top_5_words, top_3_sentences = predict_with_top_5_words_and_sentences(NEW_OBJECTION)

print(f"Predicted label: {prediction} (1: Gegrond, 0: Ongegrond)")
print("Top 5 Laws:")
for law, prob in top_5_laws:
    print(f"{law}: {prob:.4f}")

print("\nTop 5 Words by Attention (Filtered):")
for word, score in top_5_words:
    print(f"{word}: {score:.4f}")

print("\nTop 3 Sentences by Attention:")
for sentence, score in top_3_sentences:
    print(f"'{sentence}': {score:.4f}")
