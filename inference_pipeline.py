import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained('./bert_classification_text')

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('./label_classes.npy', allow_pickle=True)  # Load saved classes

# Function for prediction
def predict_new_text(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize input and move to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to numpy and find the predicted class
    predicted_class = np.argmax(outputs.logits.detach().cpu().numpy())
    
    # Return the corresponding label
    return label_encoder.inverse_transform([predicted_class])[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text classification using a BERT model")
    parser.add_argument("--text", type=str, required=True, help="Input text for classification")
    args = parser.parse_args()

    predicted_specialty = predict_new_text(args.text)
    print(f"Predicted medical specialty: {predicted_specialty}")