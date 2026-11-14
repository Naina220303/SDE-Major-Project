# Research-aligned predictive maintenance prototype

This is a substantial reimplementation inspired by Marchand et al. (2026), not a bit-for-bit reproduction. MATLAB Simscape is not available, so `simulation/pump_simulator.py` approximates the pump signals in Python. It creates 2-second, 1 kHz synthetic windows and extracts the paper's 14 named features.

## Services
- Data & Model Handler: SQLite persistence and operator feedback labels.
- ML Operate: Random Forest inference, two-sample KS input drift on reference/current windows, and labelled-feedback F1 monitoring for concept drift.
- ML Maintenance: Random Forest retraining, model version registry, MLflow experiment logging.
- API Gateway: unified REST interface.
- MLflow UI: http://localhost:5002

## Start
From this folder:
```powershell
docker compose up --build -d
```
Generate a synthetic normal-to-leak dataset:
```powershell
python scripts/generate_pump_experiment.py
```
The CSV is generated locally. Ingest it:
```powershell
python scripts/ingest_pump_csv.py
```
Then train:
```powershell
Invoke-RestMethod -Method Post http://localhost:8000/retrain
```
Swagger: http://localhost:8000/docs

## Important experiment protocol
The Python simulator is not the original MATLAB Simscape Triplex Pump model. The paper's exact physical parameters, raw dataset and complete source implementation are not recreated here. The drift thresholds, KS test choices and generated measurements are explicit implementation choices and must be calibrated/evaluated; do not claim the paper's reported 0.88 F1 or exact timing/features unless reproduced by measured runs.

## Paper-aligned features
`fPeak, pLow, pMid, pHigh, pKurtosis, qMean, qVar, qSkewness, qKurtosis, qPeak2Peak, qCrest, qRMS, qMAD, qCSRange`.
