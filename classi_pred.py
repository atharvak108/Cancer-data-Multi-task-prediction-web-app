import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

dummy_cols = [
    'Age', 'Time to Recurrence (months)', 'Survival Time (months)', 
    'Gender_Male', 'Tumor Type_Glioblastoma', 'Tumor Type_Meningioma', 
    'Tumor Grade_II', 'Tumor Grade_III', 'Tumor Grade_IV', 
    'Tumor Location_Occipital lobe', 'Tumor Location_Parietal lobe', 
    'Tumor Location_Temporal lobe', 'Treatment_Chemotherapy + Radiation', 
    'Treatment_Radiation', 'Treatment_Surgery', 
    'Treatment_Surgery + Chemotherapy', 
    'Treatment_Surgery + Radiation', 
    'Treatment_Surgery + Radiation therapy', 
    'Recurrence Site_Occipital lobe', 'Recurrence Site_Parietal lobe', 
    'Recurrence Site_Temporal lobe'
]

# Load scaler and means
with open('scaler_classi.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_means_classi.pkl', 'rb') as f:
    means = pickle.load(f)

def preprocess_classification_data(df, dummy_cols=dummy_cols, scaler=scaler, means=means):
    # Handle categorical columns
    categorical_columns = ['Gender', 'Tumor Type', 'Tumor Grade', 'Tumor Location', 'Treatment', 'Recurrence Site']
    
    # One-hot encode the categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df = df.reindex(columns=dummy_cols, fill_value=0)

    # Handle 'Time to Recurrence (months)' and 'Survival Time (months)' for NaN values
    df['Time to Recurrence (months)'].fillna(df['Time to Recurrence (months)'].mean(), inplace=True)
    df['Survival Time (months)'].fillna(df['Survival Time (months)'].mean(), inplace=True)

    # Fill missing values in numerical columns with means
    if means is not None:
        for column in df.columns:
            if column in means:
                df[column].fillna(means[column], inplace=True)
    
    df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

    # Replace remaining missing values for object (categorical) columns
    df.fillna(value={col: 'Unknown' for col in df.select_dtypes(include='object').columns}, inplace=True)

    return df

model = load_model('model_classi.h5')

def predict_from_csv(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Preprocess the data
    X = preprocess_classification_data(df)

    # Scale the input features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Convert predictions to class labels
    predictions_classes = np.argmax(predictions, axis=1)

    # Add predictions to the DataFrame with interpretation
    class_mapping = {
        0: "Complete response", 
        1: "Partial Response", 
        2: "Progressive disease", 
        3: "Stable disease"
    }
    df['Predictions'] = predictions_classes
    df['Prediction Interpretation'] = df['Predictions'].map(class_mapping)
    
    # Optionally, save the predictions to a CSV file
    df.to_csv('predictions_output.csv', index=False)
    
    print(f'Predictions saved to predictions_output.csv')
    return df[['Predictions', 'Prediction Interpretation']]

if __name__ == "__main__":
    input_csv = 'classi_pred.csv'  # Replace with your CSV file path
    predictions = predict_from_csv(input_csv)
    print(predictions)