# Telephone Fraud Detection API

This repository contains notebooks, trained models, and logs for a system that detects telephone fraud based on call patterns.  The API code now lives under `src/api_app/` and can be run directly with Uvicorn.

## Running the API

1. Install the package in editable mode (optional):
   ```bash
   pip install -e .
   ```
2. Start the server with `uvicorn`:
   ```bash
   uvicorn api_app.main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

## Directory Structure

- `Notebookes/` - Jupyter notebooks used for experimentation and developing the API.
- `Modelos/` - Saved models (`.pkl` files) and configuration files.
- `Logs_API/` - JSON logs produced by the API.
- `Resultados/` - CSV files with batch results and metrics.
- `Resultados_API/` - CSV files produced when running the API in batch mode.

## Setup

1. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install fastapi uvicorn pandas numpy scikit-learn oracledb
   ```
2. (Optional) Set environment variables for Oracle connection if your configuration differs from the defaults inside the notebook/script:
   - `ORACLE_USER`
   - `ORACLE_PASSWORD`
   - `ORACLE_DSN`

3. Run the API as described above.

