"""
Wine Quality Prediction - Original Tutorial Code
Source: https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn

This is the complete code from the EliteDataScience tutorial.
We'll use this as the starting point for MLOps refactoring.
"""

# 1. Import libraries and modules
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 2. Load red wine data
print("Loading wine quality dataset...")
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(dataset_url, sep=";")

# Check the data
print(f"\nDataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset info:")
print(data.describe())

# 3. Split data into training and test sets
print("\n" + "=" * 50)
print("Splitting data into train/test sets...")
y = data.quality
X = data.drop("quality", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# 4. Declare data preprocessing steps in a pipeline
print("\n" + "=" * 50)
print("Creating preprocessing + model pipeline...")

pipeline = make_pipeline(
    preprocessing.StandardScaler(),
    RandomForestRegressor(n_estimators=100, random_state=123),
)

print("Pipeline created:")
print(pipeline)

# 5. Declare hyperparameters to tune
print("\n" + "=" * 50)
print("Setting up hyperparameter grid...")

hyperparameters = {
    "randomforestregressor__max_features": ["auto", "sqrt", "log2"],
    "randomforestregressor__max_depth": [None, 5, 3, 1],
}

print(f"Hyperparameters to tune: {hyperparameters}")
print(f"Total combinations: {3 * 4} = 12 combinations")
print(f"With 10-fold CV: {12 * 10} = 120 model fits")

# 6. Tune model using cross-validation pipeline
print("\n" + "=" * 50)
print("Starting GridSearchCV (this may take 1-2 minutes)...")

clf = GridSearchCV(pipeline, hyperparameters, cv=10, verbose=1, n_jobs=-1)
clf.fit(X_train, y_train)

print("\n" + "=" * 50)
print("GridSearchCV completed!")
print("\nBest parameters found:")
for param, value in clf.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation score: {clf.best_score_:.4f}")

# 7. Refit is automatic (refit=True by default)
print("\n" + "=" * 50)
print("Model automatically refitted on entire training set")
print(f"Refit status: {clf.refit}")

# 8. Evaluate model pipeline on test data
print("\n" + "=" * 50)
print("Evaluating model on test set...")

y_pred = clf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nTest Set Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {np.sqrt(mse):.4f}")

# 9. Save model for future use
print("\n" + "=" * 50)
print("Saving trained model...")

model_filename = "rf_regressor.pkl"
joblib.dump(clf, model_filename)

print(f"Model saved as '{model_filename}'")
print(f"File size: {os.path.getsize(model_filename) / 1024:.2f} KB")

# 10. Test loading the model
print("\n" + "=" * 50)
print("Testing model loading...")

clf_loaded = joblib.load(model_filename)
y_pred_loaded = clf_loaded.predict(X_test)

# Verify predictions are identical
assert np.allclose(y_pred, y_pred_loaded), "Loaded model predictions don't match!"
print("✓ Model loaded successfully and predictions match!")

# 11. Example prediction
print("\n" + "=" * 50)
print("Example prediction on a single wine sample:")

# Take the first sample from test set
sample = X_test.iloc[0:1]
actual_quality = y_test.iloc[0]
predicted_quality = clf_loaded.predict(sample)[0]

print("\nWine features:")
for feature, value in sample.iloc[0].items():
    print(f"  {feature}: {value:.3f}")

print(f"\nActual quality: {actual_quality}")
print(f"Predicted quality: {predicted_quality:.2f}")
print(f"Prediction error: {abs(actual_quality - predicted_quality):.2f}")

print("\n" + "=" * 50)
print("Tutorial completed successfully! 🍷")
print("\nNext steps for MLOps:")
print("1. Refactor code into modular structure (data/, src/, tests/)")
print("2. Add MLflow tracking")
print("3. Create pytest test cases")
print("4. Build REST API with FastAPI")
print("5. Containerize with Docker")
