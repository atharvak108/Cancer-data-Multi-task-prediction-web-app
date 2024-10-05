import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import re
import pickle

dummy_cols=['avganncount', 'avgdeathsperyear', 'incidencerate',
       'medincome', 'popest2015', 'povertypercent', 'studypercap', 'binnedinc',
       'medianage', 'medianagemale', 'medianagefemale', 'percentmarried',
       'pctnohs18_24', 'pcths18_24', 'pctsomecol18_24', 'pctbachdeg18_24',
       'pcths25_over', 'pctbachdeg25_over', 'pctemployed16_over',
       'pctunemployed16_over', 'pctprivatecoverage', 'pctprivatecoveragealone',
       'pctempprivcoverage', 'pctpubliccoverage', 'pctpubliccoveragealone',
       'pctwhite', 'pctblack', 'pctasian', 'pctotherrace',
       'pctmarriedhouseholds', 'birthrate', 'geography_Alaska',
       'geography_Arizona', 'geography_Arkansas', 'geography_California',
       'geography_Colorado', 'geography_Connecticut', 'geography_Delaware',
       'geography_District of Columbia', 'geography_Florida',
       'geography_Georgia', 'geography_Hawaii', 'geography_Idaho',
       'geography_Illinois', 'geography_Indiana', 'geography_Iowa',
       'geography_Kansas', 'geography_Kentucky', 'geography_Louisiana',
       'geography_Maine', 'geography_Maryland', 'geography_Massachusetts',
       'geography_Michigan', 'geography_Minnesota', 'geography_Mississippi',
       'geography_Missouri', 'geography_Montana', 'geography_Nebraska',
       'geography_Nevada', 'geography_New Hampshire', 'geography_New Jersey',
       'geography_New Mexico', 'geography_New York',
       'geography_North Carolina', 'geography_North Dakota', 'geography_Ohio',
       'geography_Oklahoma', 'geography_Oregon', 'geography_Pennsylvania',
       'geography_Rhode Island', 'geography_South Carolina',
       'geography_South Dakota', 'geography_Tennessee', 'geography_Texas',
       'geography_Utah', 'geography_Vermont', 'geography_Virginia',
       'geography_Washington', 'geography_West Virginia',
       'geography_Wisconsin', 'geography_Wyoming']

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_means.pkl', 'rb') as f:
    means = pickle.load(f)

def preprocess_data(df,dummy_cols=dummy_cols,scaler=scaler,means=means):
    # Handle 'binnedinc' column: convert ranges to mean values
    def convert_range_to_mean(value):
        if pd.isna(value):
            return value  # Keep NaN as is
        match = re.findall(r'([\d.]+),\s*([\d.]+)', value)
        if match:
            low, high = map(float, match[0])
            return (low + high) / 2
        return value

    df['binnedinc'] = df['binnedinc'].apply(convert_range_to_mean)

    # Handle 'geography' column: extract state
    df['geography'] = df['geography'].apply(lambda x: x.split(', ')[-1] if isinstance(x, str) else x)

    # Fill missing values
    df = handle_missing_values(df)

    # One-hot encode the 'geography' column
    df = pd.get_dummies(df, columns=['geography'], drop_first=True)
    df = df.reindex(columns=dummy_cols, fill_value=0)

    if means is not None:
        # Fill missing values in numerical columns with the mean/median values
        for column in df.columns:
            if column in means:
                df[column].fillna(means[column], inplace=True)

    if scaler:
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df

def handle_missing_values(df):
    # Interpolate missing values for numerical columns
    df.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

    # Replace missing values for object (categorical) columns
    df.fillna(value={col: 'Unknown' for col in df.select_dtypes(include='object').columns}, inplace=True)

    return df

model = load_model('model.h5')

def predict_from_df(df):
    X = preprocess_data(df)
    predictions = model.predict(X)
    return predictions