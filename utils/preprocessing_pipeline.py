# utils/preprocessing_pipeline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_column=None):
    """
    Cleans and preprocesses input dataframe:
    - Imputes missing values
    - Encodes categorical features
    - Scales numerical features
    """
    df = df.copy()

    # Separate features and target if specified
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column]
        df = df.drop(columns=[target_column])

    # Handle categorical variables
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_scaled, y