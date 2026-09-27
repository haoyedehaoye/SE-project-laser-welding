"""Readiness and guarded interpretation for the predictive-maintenance roadmap."""

from __future__ import annotations

from app.data_analysis.xgboost_models import XGBoostModelRegistry
from app.data_transport import EventBus


class PredictiveMaintenanceService:
    def __init__(self, bus: EventBus, models: XGBoostModelRegistry) -> None:
        self.bus = bus
        self.models = models

    def readiness(self) -> dict:
        modalities = {
            "current": self.bus.latest("stm32") is not None,
            "weld_video": False,
            "temperature_matrix": self.bus.latest("thermal") is not None,
            "robot_speed_time": self.bus.latest("robot") is not None,
        }
        return {
            "goal": "robot_predictive_maintenance",
            "modalities": modalities,
            "available_modalities": sum(modalities.values()),
            "required_modalities": len(modalities),
            "xgboost_artifacts": self.models.describe(),
            "maintenance_model_ready": False,
            "blocking_gap": (
                "现有模型预测焊接窗口异常、裂纹和焊缝几何；它们没有机械臂故障/维修时间标签，"
                "不能直接输出剩余寿命(RUL)或故障时间。"
            ),
            "next_training_target": "以维修工单/故障码/部件更换记录为标签训练故障概率与RUL模型",
        }
