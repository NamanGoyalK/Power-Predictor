# Power Predictor Engine: System Architecture & Reference

## Overview
The **Power Predictor Engine** is an industrial ML forecasting system built for Commercial & Industrial (C&I) facilities (specifically high-load steel manufacturing). It is designed to interface with Battery Energy Storage System (BESS) Energy Management Systems (EMS) to prevent utility Maximum Demand penalties.

---

## Evolution Journey: MK1 to MK5

| Iteration | Core Focus | Major Discovery / Bottleneck | Resolution in Next Version |
| :--- | :--- | :--- | :--- |
| **MK1** | Naive BESS dispatch on raw kWh | Discharged on kWh without power factor knowledge; resulted in **₹0 demand savings**. | Made dispatch kVA-aware. |
| **MK2** | kVA-aware BESS Peak Shaving | Saved ₹44,068 by incorporating power factor, but used static split and idealized battery logic. | Moved to rolling ML evaluation. |
| **MK3** | Daily Walk-Forward Quantile Regressor | **Blind Spots Identified**: Data leakage across midnight, high false alarms (precision 14%), and the power factor conversion trap ($kVA = \text{kWh} / pf_{current}$). | Direct kVA prediction & leak-free seam. |
| **MK4** | Direct kVA + Persistence Gating | Fixed data leakage and kVA calculation. Proved that single-step quantile models suffer from high false alarm rates because Q90 predicts risk, not certainty. | Dual-channel multi-step architecture. |
| **MK5 (Final)** | **Dual-Channel Multi-Step Engine** | **The Production Solution**: 4-step forward trajectory (+15m, +30m, +45m, +60m) combined with calibrated 3-tier operational risk classification (GREEN, AMBER, RED). | Packaged into `engine.py`. |

---

## System Architecture

```
                       Industrial Smart Meter Telemetry
                    (kW, kVAR, Power Factor, Load Type)
                                    │
                                    ▼
                     Feature Engineering Pipeline
             (Lags: 15m..24h | Rolling Max/Mean: 1h..24h | Ramp Rates)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        Channel 1: Trajectory Model         Channel 2: Risk Classifier
          (Multi-Output XGBRegressor)            (XGBClassifier, Calibrated)
                  │                                   │
                  ▼                                   ▼
       4-Step Demand Trajectory            1-Hour Breach Probability (%)
    [+15m, +30m, +45m, +60m] kVA            ROC-AUC: 0.933
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                      Operational Risk Classifier
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
      GREEN Tier (<20%)       AMBER Tier (20-60%)     RED Tier (≥60%)
      80.7% of intervals      11.5% of intervals      7.8% of intervals
      1.2% breach risk        12.2% breach risk       51.3% PRECISION
            │                       │                       │
            ▼                       ▼                       ▼
    [SAFE_TO_CHARGE]       [HOLD_RESERVE_SOC]     [LOCK_DISCHARGE_MODE]
```

---

## Benchmark Performance (Full-Year Backtest: Feb–Dec 2018)

### 1. Multi-Step Trajectory Accuracy
* **+15 min Horizon:** $\text{MAE} = 29.01\text{ kVA} \quad | \quad R^2 = \mathbf{0.856}$
* **+30 min Horizon:** $\text{MAE} = 38.86\text{ kVA} \quad | \quad R^2 = \mathbf{0.740}$
* **+45 min Horizon:** $\text{MAE} = 45.49\text{ kVA} \quad | \quad R^2 = \mathbf{0.672}$
* **+60 min Horizon:** $\text{MAE} = 53.94\text{ kVA} \quad | \quad R^2 = \mathbf{0.571}$

### 2. Operational Risk Tiers
* **Breach Classifier ROC-AUC:** **$0.933$**
* **RED Tier (High Risk):** Fires only $7.8\%$ of the time with **$51.3\%$ Precision** (eliminating operator alert fatigue).
* **AMBER + RED Combined:** Captures **$84.5\%$ of all historical demand breaches**.

---

## Python API Usage (`engine.py`)

```python
import pandas as pd
from engine import PowerPredictorEngine

# 1. Initialize engine (loads model_trajectory.json and model_classifier.json)
engine = PowerPredictorEngine(contract_limit_kva=475.0)

# 2. Extract features from a rolling buffer of smart meter readings
recent_telemetry_df = pd.read_csv('Steel_industry_data.csv').tail(150)
features = PowerPredictorEngine.extract_features_from_buffer(recent_telemetry_df)

# 3. Generate 1-hour forecast and EMS guidance
forecast = engine.predict(features)
print(forecast)
```
