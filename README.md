# MLflow Experiment Tracking

This project explores Machine Learning experimentation using **MLflow** for tracking, logging, and model registry. It covers two main scenarios: **Static Training** (Iris + RandomForest) and **Online/Incremental Training** (Iris + SGDClassifier), as well as **Deep Learning** (MNIST + CNN/Keras).

## 📁 Project Structure

```
├── preprocess.py           # Preprocessing pipeline (imputer, scaler, encoder)
├── iris_statis.py          # Static training: RandomForest + GridSearchCV
├── initonline.py           # Online learning: initial fit with SGDClassifier
├── incremental.py          # Online learning: incremental update from latest model
├── deeplearning.py         # Deep learning: CNN on MNIST with MLflow
├── testingdl.py            # Deep learning: CNN on MNIST without MLflow (testing)
├── preprocessor_pipeline.joblib  # Saved preprocessing pipeline
├── data.csv                # Dataset column headers (output from preprocess)
├── pyproject.toml          # Project configuration & dependencies
├── uv.lock                 # Lock file (uv package manager)
└── README.md               # This documentation
```

## ⚙️ Prerequisites

- **Python** >= 3.12
- **uv** (package manager) or **pip**
- **MLflow Server** running at `http://127.0.0.1:5000/`

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Myrythm/mlflow-ex.git

# Install dependencies with uv
uv sync

```

## 🏃 Running the MLflow Server

Start the MLflow server before running any script:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Open the MLflow UI in your browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📜 Script Descriptions

### 1. `preprocess.py` — Preprocessing Pipeline

A utility module used by `iris_statis.py`. The `preprocess_data()` function performs:
- Automatic detection of numeric and categorical features
- Numeric pipeline: `SimpleImputer` (mean) → `StandardScaler`
- Categorical pipeline: `SimpleImputer` (constant) → `OneHotEncoder`
- Train/test split (70/30)
- Saves the pipeline to a `.joblib` file

### 2. `iris_statis.py` — Static Training

Trains a **RandomForestClassifier** with **GridSearchCV** on the Iris dataset.

```bash
python iris_statis.py
```

| Feature | Details |
|---|---|
| Algorithm | RandomForestClassifier |
| Optimization | GridSearchCV (5-fold CV) |
| Experiment | `Latihan Model Statis` |
| Logged metrics | accuracy, best hyperparameters |

### 3. `initonline.py` — Initial Online Training

Performs an initial fit of an **SGDClassifier** using `partial_fit` on the entire Iris dataset. This script should be run **once** to initialize the online learning model.

```bash
python initonline.py
```

| Feature | Details |
|---|---|
| Algorithm | SGDClassifier (log_loss) |
| Experiment | `Online Training Iris` |
| Logged metrics | accuracy |
| Logged artifacts | `online_model.joblib`, `confusion_matrix.txt`, `confusion_matrix.png`, `classification_report.txt` |
| Run name | Auto-increment (`initonline 1`, `initonline 2`, ...) |

### 4. `incremental.py` — Incremental Update

Continues training from the **latest model** stored in MLflow. Simulates data streaming by splitting the dataset into small batches.

```bash
python incremental.py
```

| Feature | Details |
|---|---|
| Algorithm | SGDClassifier (continued from `initonline.py`) |
| Experiment | `Online Training Iris` |
| Model source | Latest run in the experiment (auto-detected) |
| Batch size | 60 samples |
| Logged metrics | batch_accuracy |

### 5. `deeplearning.py` — Deep Learning (MNIST)

Trains a **CNN (Convolutional Neural Network)** on the MNIST dataset using TensorFlow/Keras, with MLflow tracking.

```bash
python deeplearning.py
```

| Feature | Details |
|---|---|
| Architecture | Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dropout → Dense |
| Dataset | MNIST (28x28 grayscale, 10 classes) |
| Experiment | `/mlflow-tf-keras-mnist` |
| Epochs | 20 |
| Logged metrics | test_loss, test_accuracy |
| Logged artifacts | `confusion_matrix.txt`, `classification_report.txt` |

### 6. `testingdl.py` — Deep Learning Testing (No MLflow)

A standalone script for testing the CNN architecture on MNIST **without** MLflow tracking. Used for quick experimentation before integrating with MLflow.

```bash
python testingdl.py
```

---

## 🔄 Online Learning Workflow

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ initonline.py│────▶│  MLflow Server   │◀────│ incremental.py   │
│ (initial fit)│     │  (model stored)  │     │ (load & update)  │
└──────────────┘     └──────────────────┘     └──────────────────┘
       │                      │                        │
       ▼                      ▼                        ▼
  Run: initonline 1     Model Registry         Run: incremental N
  + confusion matrix    + versioning           + batch_accuracy
  + accuracy            + model.pkl            + updated model
```

1. Run `initonline.py` → The initial model is trained and saved to MLflow
2. Run `incremental.py` → The latest model is loaded from MLflow, updated with a new batch, and saved back
3. Repeat step 2 for each new batch of data

---

## 📊 Viewing Results in MLflow UI

1. Open [http://127.0.0.1:5000](http://127.0.0.1:5000)
2. Select the desired experiment
3. Click on a run to view:
   - **Parameters** — Hyperparameters used
   - **Metrics** — Accuracy, loss, etc.
   - **Artifacts** — Model files, confusion matrix (text & image), classification report

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Framework | scikit-learn, TensorFlow/Keras |
| Experiment Tracking | MLflow 3.12+ |
| Preprocessing | scikit-learn Pipeline |
| Visualization | matplotlib, seaborn |
| Package Manager | uv |
| Python | >= 3.12 |
