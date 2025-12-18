"""
train_model.py - MLflow Experiment Tracking & Model Versioning

This script trains 4 models on the Iris dataset, logs all metrics and artifacts
to MLflow, identifies the best model, and registers it in the MLflow Model Registry.
"""

import os
import json
import warnings
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# Configuration
EXPERIMENT_NAME = "iris-model-zoo"
MODEL_VERSION = "v1.0.0"
REGISTERED_MODEL_NAME = "IrisModel"
OUTPUT_DIR = "app"


def load_data():
    """Load and prepare the Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names, scaler


def get_models():
    """Define the model zoo with 4 different classifiers."""
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=200, random_state=42
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, weights="uniform", algorithm="auto"
        ),
    }
    return models


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute all required metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
    }
    
    # ROC-AUC (multiclass, one-vs-rest)
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            metrics["roc_auc"] = None
    
    return metrics


def save_confusion_matrix(y_true, y_pred, target_names, filepath):
    """Generate and save confusion matrix as an image."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def save_classification_report(y_true, y_pred, target_names, filepath):
    """Save classification report as text file."""
    report = classification_report(y_true, y_pred, target_names=target_names)
    with open(filepath, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)


def train_and_log_model(
    model_name, model, X_train, X_test, y_train, y_test, target_names
):
    """Train a model and log everything to MLflow."""
    
    with mlflow.start_run(run_name=model_name) as run:
        # Log model name and version tags
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("version", MODEL_VERSION)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)
        
        # Log parameters
        params = model.get_params()
        for param_name, param_value in params.items():
            try:
                mlflow.log_param(param_name, param_value)
            except Exception:
                pass  # Skip parameters that can't be logged
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)
        
        # Create temporary directory for artifacts
        artifact_dir = f"artifacts_{model_name}"
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Save and log confusion matrix
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, target_names, cm_path)
        mlflow.log_artifact(cm_path)
        
        # Save and log classification report
        report_path = os.path.join(artifact_dir, "classification_report.txt")
        save_classification_report(y_test, y_pred, target_names, report_path)
        mlflow.log_artifact(report_path)
        
        # Save and log the model
        model_path = os.path.join(artifact_dir, "model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Log model with MLflow's sklearn flavor
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        # Clean up artifact directory
        for f in os.listdir(artifact_dir):
            os.remove(os.path.join(artifact_dir, f))
        os.rmdir(artifact_dir)
        
        print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, F1-macro={metrics['f1_macro']:.4f}")
        
        return {
            "model_name": model_name,
            "model": model,
            "run_id": run.info.run_id,
            "metrics": metrics,
        }


def save_best_model(best_result, scaler):
    """Save the best model and metadata locally."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save model with scaler as a dict
    model_data = {
        "model": best_result["model"],
        "scaler": scaler,
    }
    model_path = os.path.join(OUTPUT_DIR, "model.joblib")
    joblib.dump(model_data, model_path)
    print(f"\nSaved model to: {model_path}")
    
    # Save metadata
    metadata = {
        "best_model": best_result["model_name"],
        "metrics": {
            "accuracy": round(best_result["metrics"]["accuracy"], 4),
            "f1_macro": round(best_result["metrics"]["f1_macro"], 4),
            "precision": round(best_result["metrics"]["precision"], 4),
            "recall": round(best_result["metrics"]["recall"], 4),
        },
        "mlflow_run_id": best_result["run_id"],
        "version": MODEL_VERSION,
    }
    
    if best_result["metrics"].get("roc_auc") is not None:
        metadata["metrics"]["roc_auc"] = round(best_result["metrics"]["roc_auc"], 4)
    
    meta_path = os.path.join(OUTPUT_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_path}")


def register_model(best_result):
    """Register the best model in MLflow Model Registry."""
    model_uri = f"runs:/{best_result['run_id']}/model"
    
    # Register the model
    result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    print(f"\nRegistered model: {REGISTERED_MODEL_NAME} (version {result.version})")
    
    return result


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("MLflow Experiment Tracking - Iris Model Zoo")
    print("=" * 60)
    
    # Set up MLflow experiment
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nExperiment: {EXPERIMENT_NAME}")
    print(f"Model Version: {MODEL_VERSION}")
    
    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_names, target_names, scaler = load_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Get models
    models = get_models()
    print(f"\nTraining {len(models)} models...")
    
    # Train and log each model
    results = []
    for model_name, model in models.items():
        result = train_and_log_model(
            model_name, model, X_train, X_test, y_train, y_test, target_names
        )
        results.append(result)
    
    # Find the best model by F1-macro score
    best_result = max(results, key=lambda x: x["metrics"]["f1_macro"])
    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_result['model_name']}")
    print(f"  F1-macro: {best_result['metrics']['f1_macro']:.4f}")
    print(f"  Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"  Run ID: {best_result['run_id']}")
    print(f"{'=' * 60}")
    
    # Save the best model locally
    save_best_model(best_result, scaler)
    
    # Register the best model in MLflow Model Registry
    register_model(best_result)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Launch MLflow UI:")
    print("   mlflow ui --backend-store-uri ./mlruns --port 5000")
    print("\n2. Run Streamlit app:")
    print("   streamlit run app/app.py")


if __name__ == "__main__":
    main()
