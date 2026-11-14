
import os
import json
import requests
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.stats import ks_2samp
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
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

REGISTRY_PATH = os.path.join(
    MODEL_DIR,
    "registry.json"
)

app = FastAPI(
    title="ML Operate Service",
    version="3.0"
)


# --------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------

class PumpFeatures(BaseModel):
    fPeak: float
    pLow: float
    pMid: float
    pHigh: float
    pKurtosis: float
    qMean: float
    qVar: float
    qSkewness: float
    qKurtosis: float
    qPeak2Peak: float
    qCrest: float
    qRMS: float
    qMAD: float
    qCSRange: float


# --------------------------------------------------
# MODEL REGISTRY HELPERS
# --------------------------------------------------

def load_registry():

    if not os.path.exists(REGISTRY_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model registry not found."
        )

    try:
        with open(REGISTRY_PATH, "r") as file:
            registry = json.load(file)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read model registry: {e}"
        )

    return registry


def production():

    registry = load_registry()

    version = registry.get("production_model")

    if not version:
        return None

    return os.path.join(
        MODEL_DIR,
        version
    )


def production_metadata():

    registry = load_registry()

    production_version = registry.get(
        "production_model"
    )

    if not production_version:
        raise HTTPException(
            status_code=503,
            detail="No production model registered."
        )

    versions = registry.get("versions", [])

    for item in versions:

        if item.get("version") == production_version:
            return item

    raise HTTPException(
        status_code=500,
        detail="Production model metadata not found in registry."
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "service": "ml-operate",
        "status": "ok"
    }


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(x: PumpFeatures):

    path = production()

    if not path or not os.path.exists(path):

        raise HTTPException(
            status_code=503,
            detail="No production model; train through /retrain."
        )

    try:

        a = np.array(
            [[getattr(x, k) for k in FEATURES]],
            dtype=float
        )

        if not np.isfinite(a).all():
            raise HTTPException(
                status_code=400,
                detail="Input contains NaN or infinite values."
            )

        model = joblib.load(path)

        pred = int(model.predict(a)[0])

        prob = float(
            model.predict_proba(a)[0][1]
        )

        return {
            "failure_prediction": pred,
            "failure_probability": prob,
            "model": os.path.basename(path),
            "note": "Single prediction; input drift is evaluated on windows via /drift/input."
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )


# --------------------------------------------------
# INPUT FEATURE DRIFT
# --------------------------------------------------

@app.get("/drift/input")
def input_drift(
    reference_size: int = 100,
    window_size: int = 100,
    alpha: float = 0.05
):

    if reference_size < 1 or window_size < 1:
        raise HTTPException(
            status_code=400,
            detail="Window sizes must be positive."
        )

    if not 0 < alpha < 1:
        raise HTTPException(
            status_code=400,
            detail="Alpha must be between 0 and 1."
        )

    try:

        response = requests.get(
            DATA_URL + "/readings",
            params={"limit": 20000},
            timeout=20
        )

        response.raise_for_status()

        rows = response.json()["records"]

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=f"Data handler unavailable: {e}"
        )

    if len(rows) < reference_size + window_size:

        raise HTTPException(
            status_code=400,
            detail="Not enough records for requested reference and current windows."
        )

    # Historical reference window
    ref = rows[:reference_size]

    # Recent observation window
    cur = rows[-window_size:]

    results = {}

    for feature in FEATURES:

        a = np.array(
            [r[feature] for r in ref],
            dtype=float
        )

        b = np.array(
            [r[feature] for r in cur],
            dtype=float
        )

        if not np.isfinite(a).all() or not np.isfinite(b).all():

            raise HTTPException(
                status_code=400,
                detail=f"Invalid values found in feature {feature}."
            )

        statistic, p_value = ks_2samp(a, b)

        results[feature] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift": bool(p_value < alpha)
        }

    affected = [
        feature
        for feature, result in results.items()
        if result["drift"]
    ]

    return {
        "detected": bool(affected),
        "method": "two-sample Kolmogorov-Smirnov test",
        "alpha": alpha,
        "reference_records": reference_size,
        "current_records": window_size,
        "affected_features": affected,
        "features": results
    }


# --------------------------------------------------
# OUTPUT MODEL PERFORMANCE EVALUATION
# --------------------------------------------------

@app.get("/drift/output")
def output_drift(threshold: float = 0.70):

    if not 0 <= threshold <= 1:

        raise HTTPException(
            status_code=400,
            detail="Threshold must be between 0 and 1."
        )

    # Get current production model metadata
    metadata = production_metadata()

    model_version = metadata["version"]

    model_path = os.path.join(
        MODEL_DIR,
        model_version
    )

    if not os.path.exists(model_path):

        raise HTTPException(
            status_code=503,
            detail="Production model file not found."
        )

    # Retrieve saved test record IDs
    test_ids = metadata.get("test_record_ids", [])

    if not test_ids:

        raise HTTPException(
            status_code=400,
            detail="No held-out test record IDs found. Retrain the model using the updated ML Maintenance service."
        )

    test_ids = set(
        int(record_id)
        for record_id in test_ids
    )

    # Fetch labelled records from Data Handler
    try:

        response = requests.get(
            DATA_URL + "/readings",
            params={
                "limit": 20000,
                "labelled_only": "true"
            },
            timeout=20
        )

        response.raise_for_status()

        all_records = response.json()["records"]

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=f"Data handler unavailable: {e}"
        )

    # Select only the saved held-out test records
    test_records = [
        record
        for record in all_records
        if int(record["id"]) in test_ids
    ]

    if len(test_records) != len(test_ids):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {len(test_ids)} test records, "
                f"but found {len(test_records)}. "
                "Check whether the original test records still exist."
            )
        )

    if len(test_records) < 2:

        raise HTTPException(
            status_code=400,
            detail="Insufficient held-out test records."
        )

    # Prepare test features and actual labels
    try:

        X_test = np.asarray(
            [
                [record[feature] for feature in FEATURES]
                for record in test_records
            ],
            dtype=float
        )

        y_test = np.asarray(
            [
                record["failure"]
                for record in test_records
            ],
            dtype=int
        )

    except (KeyError, TypeError, ValueError) as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid test record data: {e}"
        )

    if not np.isfinite(X_test).all():

        raise HTTPException(
            status_code=400,
            detail="Test dataset contains NaN or infinite values."
        )

    if len(np.unique(y_test)) < 2:

        raise HTTPException(
            status_code=400,
            detail="Held-out test data must contain both normal and failure labels."
        )

    # Load production model and predict on held-out records
    try:

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Model evaluation failed: {e}"
        )

    # Calculate performance metrics
    current_f1 = float(
        f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    accuracy = float(
        accuracy_score(
            y_test,
            predictions
        )
    )

    precision = float(
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    recall = float(
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).tolist()

    # Compare against configured performance threshold
    performance_degraded = bool(
        current_f1 < threshold
    )

    return {

        "model_version": model_version,

        "evaluation_type": "held_out_test_set_evaluation",

        "method": "F1-score evaluation using saved test record IDs",

        "current_f1": current_f1,

        "registered_test_f1": metadata.get("f1"),

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "threshold": threshold,

        "performance_degraded": performance_degraded,

        "confusion_matrix": {
            "labels": ["normal", "failure"],
            "matrix": cm
        },

        "labelled_test_records": len(test_records),

        "saved_test_record_count": len(test_ids),

        "note": (
            "This endpoint evaluates the production model on its saved "
            "held-out test set. It is a reproducible performance check, "
            "not continuous live concept-drift monitoring."
        )
    }
