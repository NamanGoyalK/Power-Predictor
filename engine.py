"""
PowerPredictorEngine: Industrial 1-Hour Forward Demand & Spike Prediction Module

Designed for Battery Energy Storage System (BESS) Energy Management Systems (EMS).
Predicts the 4-step forward demand trajectory (+15m, +30m, +45m, +60m) and provides
calibrated operational risk tiers (GREEN, AMBER, RED) for intelligent peak shaving.
"""

import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb


class PowerPredictorEngine:
    """
    Production-ready dual-channel prediction engine for industrial load forecasting.
    
    Attributes:
        contract_limit_kva (float): Sanctioned peak contract demand limit (default: 475.0 kVA).
        traj_model (xgb.XGBRegressor): Multi-output regressor for 4-step trajectory.
        clf_model (xgb.XGBClassifier): Calibrated breach probability classifier.
    """

    FEATURE_COLS = [
        'demand_kva_now', 'demand_kva_lag_15m', 'demand_kva_lag_30m', 'demand_kva_lag_1h',
        'demand_kva_lag_2h', 'demand_kva_lag_24h', 'usage_now', 'power_factor_now',
        'reactive_power_now', 'reactive_power_lag_1h', 'rolling_max_1h', 'rolling_mean_1h',
        'rolling_max_4h', 'rolling_mean_4h', 'rolling_max_24h', 'ramp_rate_15m',
        'ramp_rate_1h', 'hour', 'dayofweek', 'month', 'is_weekend', 'hour_sin',
        'hour_cos', 'load_type_code'
    ]

    HORIZONS = ['+15 min', '+30 min', '+45 min', '+60 min']

    def __init__(
        self,
        traj_model_path: str = 'model_trajectory.json',
        clf_model_path: str = 'model_classifier.json',
        contract_limit_kva: float = 475.0,
        amber_threshold: float = 0.20,
        red_threshold: float = 0.60
    ):
        self.contract_limit_kva = contract_limit_kva
        self.amber_threshold = amber_threshold
        self.red_threshold = red_threshold

        self.traj_model = xgb.XGBRegressor()
        self.traj_model.load_model(traj_model_path)

        self.clf_model = xgb.XGBClassifier()
        self.clf_model.load_model(clf_model_path)

    @staticmethod
    def extract_features_from_buffer(history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a historical buffer of 15-minute readings containing:
        ['date', 'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
         'Lagging_Current_Power_Factor', 'Load_Type']
        computes the exact feature vector for the latest timestamp.
        """
        df = history_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            df = df.sort_values('date').set_index('date')

        pf = df['Lagging_Current_Power_Factor'].clip(lower=50.0) / 100.0
        df['Demand_kVA'] = (df['Usage_kWh'] * 4.0) / pf

        # Compute features
        df['demand_kva_now']      = df['Demand_kVA']
        df['demand_kva_lag_15m']  = df['Demand_kVA'].shift(1)
        df['demand_kva_lag_30m']  = df['Demand_kVA'].shift(2)
        df['demand_kva_lag_1h']   = df['Demand_kVA'].shift(4)
        df['demand_kva_lag_2h']   = df['Demand_kVA'].shift(8)
        df['demand_kva_lag_24h']  = df['Demand_kVA'].shift(96)

        df['usage_now']             = df['Usage_kWh']
        df['power_factor_now']       = df['Lagging_Current_Power_Factor']
        df['reactive_power_now']    = df['Lagging_Current_Reactive.Power_kVarh']
        df['reactive_power_lag_1h'] = df['Lagging_Current_Reactive.Power_kVarh'].shift(4)

        df['rolling_max_1h']  = df['Demand_kVA'].rolling(4).max()
        df['rolling_mean_1h'] = df['Demand_kVA'].rolling(4).mean()
        df['rolling_max_4h']  = df['Demand_kVA'].rolling(16).max()
        df['rolling_mean_4h'] = df['Demand_kVA'].rolling(16).mean()
        df['rolling_max_24h'] = df['Demand_kVA'].rolling(96).max()

        df['ramp_rate_15m'] = df['Demand_kVA'] - df['Demand_kVA'].shift(1)
        df['ramp_rate_1h']  = df['Demand_kVA'] - df['Demand_kVA'].shift(4)

        df['hour']       = df.index.hour
        df['dayofweek']  = df.index.dayofweek
        df['month']      = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

        load_type_map = {'Light_Load': 0, 'Medium_Load': 1, 'Maximum_Load': 2}
        df['load_type_code'] = df['Load_Type'].map(load_type_map)

        return df[PowerPredictorEngine.FEATURE_COLS].iloc[[-1]]

    def predict(self, feature_row: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes forward inference on a single feature row.
        
        Returns:
            Dict containing trajectory, risk probability, operational tier, and EMS recommendation.
        """
        X = feature_row[self.FEATURE_COLS]

        # 1. Trajectory Forecast (+15m, +30m, +45m, +60m)
        traj_preds = self.traj_model.predict(X)[0]
        trajectory = {
            self.HORIZONS[i]: round(float(traj_preds[i]), 1) for i in range(4)
        }
        projected_peak_kva = round(float(np.max(traj_preds)), 1)

        # 2. Breach Risk Probability
        prob = float(self.clf_model.predict_proba(X)[0, 1])

        # 3. Operational Tier & EMS Action
        if prob >= self.red_threshold:
            tier = "RED"
            status = "HIGH_BREACH_RISK"
            action = "LOCK_DISCHARGE_MODE"
            ems_guidance = (
                f"Spike risk is {prob*100:.1f}%. High probability of breaching "
                f"{self.contract_limit_kva} kVA limit. Arm battery inverter to clamp peaks."
            )
        elif prob >= self.amber_threshold:
            tier = "AMBER"
            status = "ELEVATED_DEMAND"
            action = "HOLD_RESERVE_CAPACITY"
            ems_guidance = (
                f"Spike risk is {prob*100:.1f}%. Elevated background load. Maintain "
                f"at least 30% battery SoC reserve. Restrict bulk grid charging."
            )
        else:
            tier = "GREEN"
            status = "SAFE_OPERATION"
            action = "SAFE_TO_CHARGE"
            ems_guidance = (
                f"Spike risk is {prob*100:.1f}%. Plant is operating well below limit. "
                f"Safe to charge battery from grid or solar without penalty risk."
            )

        return {
            "timestamp": str(feature_row.index[-1]) if hasattr(feature_row.index, 'hour') else None,
            "projected_trajectory_kva": trajectory,
            "projected_peak_next_hour_kva": projected_peak_kva,
            "contract_limit_kva": self.contract_limit_kva,
            "breach_projected": projected_peak_kva >= self.contract_limit_kva,
            "spike_probability_pct": round(prob * 100, 1),
            "operational_tier": tier,
            "ems_status": status,
            "recommended_action": action,
            "guidance": ems_guidance
        }


if __name__ == '__main__':
    print("Testing PowerPredictorEngine...")
    engine = PowerPredictorEngine()
    
    # Load recent 100 rows as buffer
    raw = pd.read_csv('Steel_industry_data.csv').tail(150)
    feat_vector = PowerPredictorEngine.extract_features_from_buffer(raw)
    
    result = engine.predict(feat_vector)
    print("\nInference Output for Latest Telemetry Reading:")
    print(json.dumps(result, indent=2))
