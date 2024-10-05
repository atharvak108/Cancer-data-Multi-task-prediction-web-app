import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from reg_pred import predict_from_df
from predictions3 import get_prediction_result
# from predict_segmentation import predict_tumour, visualize_prediction
from classi_pred import predict_from_csv
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np
from textgenusingtransformer import rag_pipeline, load_text_generation_models

# Load models
tabular_model = load_model('model.h5')
image_classification_model = load_model('vgg16_functional_model.h5')
# segmentation_model = load_model('unet_model.h5')

# Load model, tokenizer, and label encoder for text classification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained('bert_classification_text')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load text generation models and data
df, text_model, index, generator_model, gen_tokenizer = load_text_generation_models()

# Streamlit app layout
st.title('Multi-task Prediction App')

#"Text Generation"
task = st.selectbox(
    "Select a Task",
    ("Tabular Regression Prediction", "Tabular Classification", "Image Classification", "Image Segmentation", "Text Classification", "Text Generation")
)

if task == "Tabular Regression Prediction":
    st.header("Tabular Regression Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        predictions = predict_from_df(df)
        st.write("Predictions:")
        st.write(predictions)
        st.download_button(
            label="Download Predictions as CSV",
            data=pd.DataFrame(predictions, columns=['Predictions']).to_csv(index=False),
            file_name='predictions_output.csv',
            mime='text/csv'
        )

elif task == "Tabular Classification":
    st.header("Tabular Classification")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        predictions = predict_from_csv(uploaded_file)
        st.write("Predictions:")
        st.write(predictions)
        st.download_button(
            label="Download Predictions as CSV",
            data=predictions.to_csv(index=False),
            file_name='predictions_output.csv',
            mime='text/csv'
        )

elif task == "Image Classification":
    st.header('Image Classification')
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        predictions, result = get_prediction_result(image_classification_model, uploaded_file)
        st.write(f"Prediction Result: {result}")
        st.write(f"Raw Predictions: {predictions}")

# elif task == "Image Segmentation":
#     st.header("Image Segmentation")
#     uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         pred_mask = predict_tumour(uploaded_file, segmentation_model)
#         visualize_prediction(uploaded_file, pred_mask)
#         st.image('predicted_mask_output.png')

elif task == "Text Classification":
    st.header("Text Classification")
    user_input = st.text_area("Enter the text for classification")

    if st.button("Classify Text"):
        if user_input:
            # Tokenize input and move to device
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

            # Get model output
            with torch.no_grad():
                outputs = model(**inputs)

            # Convert logits to numpy and find the predicted class
            predicted_class = np.argmax(outputs.logits.detach().cpu().numpy())

            # Return the corresponding label
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            st.write(f"Predicted medical specialty: {predicted_label}")

elif task == "Text Generation":
    st.header("Text Generation")
    user_query = st.text_area("Enter the query for generating the response")

    if st.button("Generate Response"):
        if user_query:
            answer = rag_pipeline(user_query, df, text_model, index, generator_model, gen_tokenizer)
            st.write(f"Generated Answer: {answer}")
