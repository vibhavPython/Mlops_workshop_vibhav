import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.preprocessing import LabelEncoder
from collections.abc import Iterator

def split_data(data_df):
    """Split a dataframe into training and validation datasets"""
    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])
    
    # Remove 'ps_ind_04_cat' column if it exists
    if 'ps_ind_04_cat' in features.columns:
        features = features.drop(columns=['ps_ind_04_cat'])

    features_train, features_valid, labels_train, labels_valid = \
        train_test_split(features, labels, test_size=0.2, random_state=0)
    
    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(features_valid, label=labels_valid, free_raw_data=False)
    return (train_data, valid_data)

def train_model(data, parameters):
    """Train a model with the given datasets and parameters"""
    # Access train_data with data[0] and valid_data with data[1]
    train_data = data[0]
    valid_data = data[1]

    # Create a callback for early stopping
    early_stopping_callback = lightgbm.early_stopping(stopping_rounds=20, verbose=True)

    model = lightgbm.train(
        parameters,
        train_data,
        valid_sets=[valid_data],  # Ensure this is a list of valid sets
        num_boost_round=500,
        callbacks=[early_stopping_callback]  # Include early stopping callback
    )
    return model

def get_model_metrics(model, data):
    """Construct a dictionary of metrics for the model"""
    print(data[1].data)
    predictions = model.predict(data[1].data)
    fpr, tpr, thresholds = metrics.roc_curve(data[1].label, predictions)
    auc = metrics.auc(fpr, tpr)
    model_metrics = {"auc": auc}
    print(model_metrics)
    return model_metrics

# Example usage would involve creating a DataFrame `data_df` that includes the columns 'target', 'id', and 'ps_ind_04_cat' 
# and other necessary features, then calling these functions with appropriate parameters.
