
import os
import json
import time
import requests
import joblib
import numpy as np
import mlflow

from fastapi import FastAPI, HTTPException

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from simulation.pump_simulator import FEATURES


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_URL = os.getenv(
    "DATA_URL",
    "http://data-handler:8001"
)

MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "/models"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Triplex-Pump-Predictive-Maintenance")

app = FastAPI(
    title="ML Maintenance Service",
    version="3.1"
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "service": "ml-maintenance",
        "status": "ok"
    }


# --------------------------------------------------
# RETRAIN MODEL
# --------------------------------------------------

@app.post("/retrain")
def retrain():

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Fetch labelled records
    try:
        response = requests.get(
            DATA_URL + "/readings",
            params={
                "limit": 20000,
                "labelled_only": "true"
            },
            timeout=30
        )

        response.raise_for_status()
        records = response.json()["records"]

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Data handler unavailable: {e}"
        )

    if len(records) < 30:
        raise HTTPException(
            status_code=400,
            detail="At least 30 labelled pump windows are required."
        )

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    try:
        X = np.asarray(
            [[record[k] for k in FEATURES] for record in records],
            dtype=float
        )

        y = np.asarray(
            [record["failure"] for record in records],
            dtype=int
        )

        record_ids = np.asarray(
            [int(record["id"]) for record in records],
            dtype=int
        )

    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid training data: {e}"
        )

    if len(np.unique(y)) < 2:
        raise HTTPException(
            status_code=400,
            detail="Training requires both normal and failure labels."
        )

    if not np.isfinite(X).all():
        raise HTTPException(
            status_code=400,
            detail="Dataset contains NaN or infinite feature values."
        )

    # --------------------------------------------------
    # CHRONOLOGICAL SPLIT WITHIN EACH CLASS
    # --------------------------------------------------

    train_indices = []
    test_indices = []

    for label in np.unique(y):

        class_indices = np.flatnonzero(y == label)

        if len(class_indices) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough records for class {label}."
            )

        # Latest 25% of each class reserved for testing
        n_test = max(
            1,
            int(np.ceil(0.25 * len(class_indices)))
        )

        train_indices.extend(class_indices[:-n_test])
        test_indices.extend(class_indices[-n_test:])

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    Xtr = X[train_indices]
    Xte = X[test_indices]

    ytr = y[train_indices]
    yte = y[test_indices]

    if len(np.unique(ytr)) < 2:
        raise HTTPException(
            status_code=400,
            detail="Training data must contain both classes."
        )

    if len(np.unique(yte)) < 2:
        raise HTTPException(
            status_code=400,
            detail="Test data must contain both normal and failure labels."
        )

    # Save the IDs corresponding to the held-out test records
    test_record_ids = [
        int(record_ids[i])
        for i in test_indices
    ]

    # --------------------------------------------------
    # RANDOM FOREST MODEL
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------
    # TRAIN AND EVALUATE
    # --------------------------------------------------

    try:

        with mlflow.start_run() as run:

            model.fit(Xtr, ytr)

            predictions = model.predict(Xte)

            f1 = float(
                f1_score(
                    yte,
                    predictions,
                    zero_division=0
                )
            )

            accuracy = float(
                accuracy_score(yte, predictions)
            )

            precision = float(
                precision_score(
                    yte,
                    predictions,
                    zero_division=0
                )
            )

            recall = float(
                recall_score(
                    yte,
                    predictions,
                    zero_division=0
                )
            )

            cm = confusion_matrix(
                yte,
                predictions,
                labels=[0, 1]
            ).tolist()

            report = classification_report(
                yte,
                predictions,
                labels=[0, 1],
                target_names=["normal", "failure"],
                zero_division=0,
                output_dict=True
            )

            # --------------------------------------------------
            # MLFLOW LOGGING
            # --------------------------------------------------

            mlflow.log_params({
                "algorithm": "RandomForestClassifier",
                "n_estimators": 200,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "features": len(FEATURES),
                "train_rows": len(Xtr),
                "test_rows": len(Xte),
                "split_method": "chronological_within_class"
            })

            mlflow.log_metrics({
                "test_f1": f1,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall
            })

            # --------------------------------------------------
            # SAVE MODEL
            # --------------------------------------------------

            version = f"pump_rf_{int(time.time())}.joblib"

            model_path = os.path.join(
                MODEL_DIR,
                version
            )

            joblib.dump(model, model_path)

            mlflow.log_artifact(model_path)

            # --------------------------------------------------
            # MODEL REGISTRY
            # --------------------------------------------------

            registry_path = os.path.join(
                MODEL_DIR,
                "registry.json"
            )

            if os.path.exists(registry_path):

                with open(registry_path, "r") as file:
                    history = json.load(file).get(
                        "versions",
                        []
                    )

            else:
                history = []

            # Archive previous production model
            for previous_version in history:

                if previous_version.get("status") == "Production":
                    previous_version["status"] = "Archived"

            # Register new model with its test record IDs
            history.append({
                "version": version,
                "run_id": run.info.run_id,
                "f1": f1,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "test_record_ids": test_record_ids,
                "test_record_count": len(test_record_ids),
                "status": "Production",
                "trained_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            })

            with open(registry_path, "w") as file:

                json.dump(
                    {
                        "production_model": version,
                        "versions": history
                    },
                    file,
                    indent=2
                )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {e}"
        )

    # --------------------------------------------------
    # RESPONSE
    # --------------------------------------------------

    return {
        "message": "Model trained and registered successfully",
        "version": version,
        "test_f1": f1,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "confusion_matrix": {
            "labels": ["normal", "failure"],
            "matrix": cm
        },
        "classification_report": report,
        "training_samples": len(Xtr),
        "test_samples": len(Xte),
        "test_record_count": len(test_record_ids),
        "mlflow_run_id": run.info.run_id
    }
