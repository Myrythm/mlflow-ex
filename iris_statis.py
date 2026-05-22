from preprocess import preprocess_data
from sklearn.datasets import load_iris
from pathlib import Path
import os
import sys
import tempfile
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier # Algoritma dapat disesuaikan
from sklearn.model_selection import GridSearchCV # Optimasi dapat disesuaikan

local_tmp = Path("tmp").resolve()
local_tmp.mkdir(exist_ok=True)
os.environ["TMP"] = str(local_tmp)
os.environ["TEMP"] = str(local_tmp)
tempfile.tempdir = str(local_tmp)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data.head()
input_example = data[0:5]
 
# Contoh Penggunaan
X_train, X_test, y_train, y_test = preprocess_data(data, 'target', 'preprocessor_pipeline.joblib', 'data.csv')

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
 
# Create a new MLflow Experiment
mlflow.set_experiment("Latihan Model Statis")

with mlflow.start_run():
    run = mlflow.active_run()
    print(f"MLflow run_id: {run.info.run_id}")
    mlflow.log_param("status", "started")
    # Parameter Grid untuk GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    # Inisialisasi dan Grid Search
    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
 
    # Perform grid search
    total_fits = (
        len(param_grid['n_estimators'])
        * len(param_grid['max_depth'])
        * len(param_grid['min_samples_split'])
        * len(param_grid['min_samples_leaf'])
        * 5
    )
    print(f"Mulai GridSearchCV: {total_fits} fit")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
 
    # Mendapatkan Parameter dan Model Terbaik
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
 
    # Log best parameters
    mlflow.log_params(best_params)
 
    # Train the best model on the training set
    best_model.fit(X_train, y_train)
 
    # Evaluate the model on the test set and log accuracy
    accuracy = best_model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")
