import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (GradientBoostingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np


# Load features from a single file
def load_combined_features(feature_file):
    """
    Load all features and labels from a single CSV file.

    Parameters:
    -----------
    feature_file : str
        Path to the combined feature file.

    Returns:
    --------
    pd.DataFrame
        Combined features with labels.
    """
    return pd.read_csv(feature_file)


# Define hyperparameter grids for each model
parameter_grids = {
    'SVM': {
        'C': [1e-3, 1e-2, 1e-1, 1, 10, 100],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 1.0],
        'subsample': [0.8, 1.0]
    },
    'ExtraTrees': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'DecisionTree': {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'max_depth': [None, 10, 20, 30]
    },
    'LogisticRegression': {
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'MLP': {
        'hidden_layer_sizes': [(100,), (50, 50), (100, 100)],
        'solver': ['adam', 'lbfgs', 'sgd']
    },
    'NaiveBayes': {}  # No hyperparameters for tuning
}

# Define model classes
models = {
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(),
    'AdaBoost': AdaBoostClassifier(),
    'MLP': MLPClassifier(),
    'NaiveBayes': GaussianNB()
}


# Patient-level aggregation
def aggregate_patient_predictions(data, predictions, test_data, method='majority_vote'):
    """
    Aggregate channel-level predictions to patient-level predictions.

    Parameters:
    -----------
    data : pd.DataFrame
        The input data containing 'participant_id' and channel-level predictions.
    predictions : np.ndarray
        The channel-level predictions for the test set.
    test_data : pd.DataFrame
        The test dataset with 'participant_id' for correct aggregation.
    method : str
        Aggregation method ('majority_vote' or 'average_probability').

    Returns:
    --------
    pd.DataFrame
        Patient-level predictions with participant IDs.
    """
    # Only use participant_ids from the test set
    test_data['prediction'] = predictions
    patient_results = test_data.groupby('participant_id').agg({'prediction': 'mean'}).reset_index()

    if method == 'majority_vote':
        # Convert probabilities into binary predictions based on threshold
        patient_results['final_prediction'] = (patient_results['prediction'] > 0.5).astype(int)
    elif method == 'average_probability':
        patient_results['final_prediction'] = patient_results['prediction']
    else:
        raise ValueError("Invalid aggregation method.")

    return patient_results[['participant_id', 'final_prediction']]


# Modify the model evaluation to use only test set predictions
def train_and_evaluate_models(data, models, parameter_grids):
    """
    Train and evaluate models with hyperparameter tuning.

    Parameters:
    -----------
    data : pd.DataFrame
        Combined dataset with features and labels.
    models : dict
        Dictionary of model instances.
    parameter_grids : dict
        Hyperparameter grids for each model.

    Returns:
    --------
    None
    """
    X = data.drop(columns=['Autistic', 'participant_id', 'channel'])
    y = data['Autistic']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Combine test set data for aggregation
    test_data = data.loc[X_test.index, ['participant_id']]  # Get the corresponding participant_ids

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        if parameter_grids.get(model_name):
            # Perform grid search for hyperparameter tuning
            grid_search = GridSearchCV(estimator=model, param_grid=parameter_grids[model_name],
                                       scoring='accuracy', cv=5, verbose=2)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            # Train model without hyperparameter tuning
            best_model = model
            best_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

        # Aggregating patient-level predictions
        patient_results = aggregate_patient_predictions(data, y_pred, test_data, method='majority_vote')

        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        if y_prob is not None:
            print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob)}")
        print("Patient-Level Results:\n", patient_results)


# Main function
def main():
    """
    Main function to load features, train models, and evaluate them.
    """
    feature_file = '/features_combined.csv'
    data = load_combined_features(feature_file)
    train_and_evaluate_models(data, models, parameter_grids)


if __name__ == '__main__':
    main()
