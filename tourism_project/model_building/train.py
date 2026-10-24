from pathlib import Path
import json
import pandas as pd
import joblib
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "model_building"
DEPLOY_DIR = ROOT / "deployment"
TARGET = "ProdTaken"

X_train = pd.read_csv(BUILD_DIR / "Xtrain.csv")
X_test = pd.read_csv(BUILD_DIR / "Xtest.csv")
y_train = pd.read_csv(BUILD_DIR / "ytrain.csv")[TARGET].astype(int)
y_test = pd.read_csv(BUILD_DIR / "ytest.csv")[TARGET].astype(int)

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_features),
    ("categorical", categorical_pipe, categorical_features)
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1))
])
param_grid = {
    "model__n_estimators": [150, 300],
    "model__max_depth": [None, 12],
    "model__min_samples_leaf": [1, 3]
}
if mlflow is not None:
    # Use SQLite instead of the deprecated local FileStore backend (required by MLflow 3).
    tracking_db = BUILD_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve()}")
    mlflow.set_experiment("tourism-package-prediction")
search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
search.fit(X_train, y_train)
best_model = search.best_estimator_
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, probabilities))
}
if mlflow is not None:
    with mlflow.start_run():
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.sklearn.log_model(best_model, name="model")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, DEPLOY_DIR / "model.joblib")
with open(DEPLOY_DIR / "model_metadata.json", "w") as f:
    json.dump({"target": TARGET, "features": X_train.columns.tolist(),
               "best_params": search.best_params_, "metrics": metrics}, f, indent=2)
print("Best parameters:", search.best_params_)
print("Test metrics:", json.dumps(metrics, indent=2))
print("\\nClassification report:\\n", classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:\\n", confusion_matrix(y_test, predictions))
print("Saved model:", DEPLOY_DIR / "model.joblib")
