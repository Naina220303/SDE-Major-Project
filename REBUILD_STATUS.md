# Rebuild notes

This archive was upgraded from the user's existing project. Original report/PPT/diagram are retained as historical artifacts and are NOT yet aligned with the new code. They must be regenerated after the experiment is run and outputs verified.

Implemented in this revision:
- Python pump signal simulator with 14 named paper features
- SQLite schema for 14 features and labels
- KS two-sample input drift endpoint using stored windows
- labelled-feedback F1 concept-drift endpoint
- Random Forest training and MLflow run logging
- model file registry and gateway endpoints
- Docker Compose MLflow service

Still required before claiming a complete paper reproduction:
- Inspect/validate pump signal feature definitions against authors' source code/data
- Validate generated time progression and drift progression experimentally
- Add an explicit operator/regulator UI and human approval/staging-to-production workflow
- Add automated retraining trigger and active-learning queue
- Add robust readiness/health checks and service integration tests
- Regenerate report, architecture diagram and slides based on actual measured outputs
