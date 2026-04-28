# HydroTwin: Biological Ground Truth for Autonomous Sensor Validation in Controlled Environment Agriculture

Hydro-Twin is a data-driven system designed to monitor aquaponics environments by predicting expected ammonia behavior and detecting anomalies in real time.

## Developed by Daniela Najmias Lang

## Problem

Aquaponics systems rely heavily on sensor data to monitor water quality, however:

- Sensors degrade over time
- Failures are often detected too late
- Systems often rely on redundant hardware instead of validation

which creates risk for both system stability and biological health!

## Solution

HydroTwin introduces a software-based validation layer built on top of sensor data.

The system works in five stages:

1. Predict expected ammonia using a machine learning model  
2. Compare predicted vs observed values (residuals)  
3. Detect anomalies using:
   - Residual-based detection (sensor-level)
   - Isolation Forest (system-level)  
4. Classify system state  
5. Generate a trust score over time  

This allows operators to distinguish between:
- Sensor drift
- Biological risk
- Critical system failures

## Key Results

- MAE reduced from ~0.061 (baseline) → ~0.033 (final model)
- R² ≈ 0.97
- Clear separation between anomaly types
- Robust performance across multiple ponds

## Repository Structure

- `notebooks/`
  - Full development notebook (exploration and iterations)
  - Clean final pipeline (reproducible version)

- `data/processed/`
  - Model-ready datasets used for analysis and deployment

- `outputs/`
  - Metrics, feature importance, and validation results

- `app/`
  - Streamlit dashboard 

## How to Run

1. Install dependencies: pip install -r requirements.txt
2. Run the app in streamlit: https://hydrotwin.streamlit.app
