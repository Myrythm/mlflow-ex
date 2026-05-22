from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
 
 
# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
 
 
# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
 
 
# Create a new MLflow Experiment
mlflow.set_experiment("Online Training Iris")
 
 
# Model Online Learning: SGDClassifier
model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, max_iter=10000)
 
 
# Kelas target (diperlukan untuk partial_fit)
classes = data['target'].unique()
 
# Auto-increment run counter dari MLflow
experiment = mlflow.get_experiment_by_name("Online Training Iris")
if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    i = len(runs) + 1
else:
    i = 1

with mlflow.start_run(run_name=f"initonline {i}"):
    mlflow.autolog()  # Melakukan logging otomatis
 
 
    # Preprocessing data untuk batch
    X_batch = data.drop(columns=['target'])
    y_batch = data['target']
 
 
    # Initial fit untuk batch pertama
    model.partial_fit(X_batch, y_batch, classes=classes)
 
 
    # Log metrik setelah setiap batch
    accuracy = model.score(X_batch, y_batch)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix & classification report
    y_pred = model.predict(X_batch)
    conf_matrix = confusion_matrix(y_batch, y_pred)
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    # Plot confusion matrix sebagai heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    mlflow.log_artifact("confusion_matrix.png")
    plt.close(fig)

    class_report = classification_report(y_batch, y_pred, target_names=iris.target_names)
    mlflow.log_text(class_report, "classification_report.txt")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    # Simpan model ke file lokal
    dump(model, "online_model.joblib")
 
 
    # Log file model sebagai artefak ke MLflow
    mlflow.log_artifact("online_model.joblib", artifact_path="model_artifacts")
 
 
    # Log model setelah selesai online training
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="online_model",
        input_example=X_batch.iloc[:5]
    )