"""XGBoost inference and predictive-maintenance readiness services."""

from .maintenance import PredictiveMaintenanceService
from .xgboost_models import XGBoostModelRegistry

__all__ = ["PredictiveMaintenanceService", "XGBoostModelRegistry"]

