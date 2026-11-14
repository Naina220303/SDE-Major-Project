# PumpSense Streamlit Dashboard — Setup

This add-on adds a browser-based dashboard to the existing predictive-maintenance project. It does not replace or modify the FastAPI microservices.

## 1. Copy the dashboard folder

Copy the `dashboard` folder from this add-on into the root of your existing `predictive_maintenance_project` folder (the same folder that contains `docker-compose.yml`). You should have:

```text
predictive_maintenance_project/
  dashboard/
    app.py
  docker-compose.yml
  requirements.txt
  services/
  scripts/
  simulation/
```

## 2. Install the dashboard dependency

Open PowerShell in the project root and run:

```powershell
python -m pip install streamlit pandas requests
```

The project requirements already list Streamlit and pandas, so installing the full requirements is also possible:

```powershell
python -m pip install -r requirements.txt
```

## 3. Start the existing backend

```powershell
docker compose up -d
```

Check that the API Gateway is running:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## 4. Launch the dashboard

From the project root:

```powershell
python -m streamlit run dashboard/app.py
```

Open the URL Streamlit prints, normally:

`http://localhost:8501`

The dashboard calls the API Gateway at `http://localhost:8000` by default. You can change this in the dashboard sidebar.

## 5. Suggested demo order

1. Overview: show stored measurements, scenario counts and feature trends.
2. Live prediction: use the prefilled latest measurement and run a prediction.
3. Drift monitoring: run the input feature drift test and inspect the KS p-values.
4. Model performance: evaluate the production model on its saved held-out test set.
5. Model maintenance: trigger retraining only if you intend to change the active model, then open MLflow.

## Important interpretation notes

- The simulator is a Python-generated approximation, not the paper's MATLAB Simscape model.
- Input drift means a change in feature distributions; it does not by itself prove concept drift.
- The output evaluation tab is a reproducible held-out test-set evaluation, not continuous live feedback monitoring.
- Displayed model metrics come from your synthetic experiment, not the paper's reported metrics.
- The dashboard is a presentation layer over existing APIs. It does not implement the missing operator/regulator approval workflow.
