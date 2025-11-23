import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_preprocess(path, save_encoders=True):
    """
    Load and preprocess the credit data
    
    Parameters:
    -----------
    path : str
        Path to the CSV file
    save_encoders : bool
        If True, save the label encoders for later use
    
    Returns:
    --------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    """
    # Handle relative paths
    if not os.path.isabs(path):
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        path = os.path.join(project_root, path)
    
    df = pd.read_csv(path)
    df2 = df.copy()
    
    encoders = {}
    
    for col in df2.columns:
        if df2[col].dtype == object:
            le = LabelEncoder()
            df2[col] = le.fit_transform(df2[col])
            encoders[col] = le
    
    # Save encoders if requested
    if save_encoders:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        encoders_path = os.path.join(models_dir, "label_encoders.pkl")
        joblib.dump(encoders, encoders_path)
        print(f"✓ Label encoders saved to {encoders_path}")
    
    X = df2.drop("status", axis=1)
    y = df2["status"]
    
    return X, y


def preprocess_new_data(df, encoders_path="models/label_encoders.pkl"):
    """
    Preprocess new data using saved encoders
    
    Parameters:
    -----------
    df : DataFrame
        New data to preprocess
    encoders_path : str
        Path to saved label encoders
    
    Returns:
    --------
    df_processed : DataFrame
        Preprocessed dataframe
    """
    df_processed = df.copy()
    
    try:
        encoders = joblib.load(encoders_path)
        
        for col in df_processed.columns:
            if col in encoders and col != 'status':
                # Handle unseen categories
                le = encoders[col]
                df_processed[col] = df_processed[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Remove target column if present
        if 'status' in df_processed.columns:
            df_processed = df_processed.drop('status', axis=1)
            
    except FileNotFoundError:
        print("Warning: Encoders not found. Using new LabelEncoder for each column.")
        for col in df_processed.columns:
            if df_processed[col].dtype == object:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
        
        if 'status' in df_processed.columns:
            df_processed = df_processed.drop('status', axis=1)
    
    return df_processed