"""One production registry for every XGBoost model artifact in ``models/``."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"

PROCESS_CLASSIFIERS = (
    "xgb_lap_joint",
    "xgb_public",
    "xgb_public_group",
    "xgb_public_sensor_only",
)
GEOMETRY_MODELS = ("steel_w", "copper_w", "copper_d", "gap")


class ModelInputError(ValueError):
    pass


class XGBoostModelRegistry:
    """Lazy model loader so the camera/current paths do not require ML at startup."""

    def __init__(self, model_root: str | Path = DEFAULT_MODEL_ROOT) -> None:
        self.root = Path(model_root)
        self._models: dict[str, Any] = {}

    def describe(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        anomaly_dir = self.root / "xgb_anomaly"
        entries.append(self._entry("xgb_anomaly", "window_anomaly", anomaly_dir, 1))
        for name in PROCESS_CLASSIFIERS:
            entries.append(self._entry(name, "crack_classifier", self.root / name, 1))
        multi = self.root / "xgb_multiout"
        entries.append(self._entry("xgb_multiout", "crack_and_geometry", multi, 5))
        return entries

    def predict_process(self, parameters: dict[str, Any]) -> dict[str, Any]:
        normalized = {self._canonical(k): float(v) for k, v in parameters.items()}
        results: dict[str, Any] = {}
        probabilities: list[float] = []
        for name in PROCESS_CLASSIFIERS:
            try:
                features = self._read_json(self.root / name / "features.json")
                vector = [self._required(normalized, self._canonical(feature)) for feature in features]
                model = self._classifier(name, self.root / name / "model.json")
                probability = self._positive_probability(model, vector)
                probabilities.append(probability)
                results[name] = {"status": "ok", "crack_probability": round(probability, 6)}
            except Exception as exc:
                results[name] = {"status": "unavailable", "reason": str(exc)}

        results["xgb_multiout"] = self._predict_multiout(normalized)
        multi_probability = results["xgb_multiout"].get("crack_probability")
        if multi_probability is not None:
            probabilities.append(float(multi_probability))
        return {
            "models": results,
            "ensemble": {
                "models_used": len(probabilities),
                "mean_crack_probability": round(fmean(probabilities), 6) if probabilities else None,
                "disagreement": round(pstdev(probabilities), 6) if len(probabilities) > 1 else 0.0,
                "note": "研究模型集成结果，不等同于设备故障概率或剩余寿命。",
            },
        }

    def predict_anomaly(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        directory = self.root / "xgb_anomaly"
        config = self._read_json(directory / "config.json")
        window_size = int(config.get("window_size", 32))
        if len(samples) < window_size:
            raise ModelInputError(f"xgb_anomaly 至少需要 {window_size} 条时序样本")
        window = samples[-window_size:]
        features: list[float] = []
        for field, operations in config["feature_spec"].items():
            values = [float(self._required(row, field)) for row in window]
            for operation in operations:
                features.append(self._aggregate(values, operation))
        expected = self._read_json(directory / "features.json")
        if len(features) != len(expected):
            raise RuntimeError("xgb_anomaly 特征配置与模型不一致")
        probability = self._positive_probability(
            self._classifier("xgb_anomaly", directory / "model.json"), features
        )
        threshold = float(config.get("thresholds", {}).get("anomaly_prob", 0.6))
        return {
            "model": "xgb_anomaly",
            "window_size": window_size,
            "anomaly_probability": round(probability, 6),
            "threshold": threshold,
            "label": "anomaly" if probability >= threshold else "normal",
            "training_scope": "synthetic",
        }

    def predict_all(
        self, parameters: dict[str, Any] | None = None, samples: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        output: dict[str, Any] = {"process_models": self.predict_process(parameters or {})}
        if samples is None:
            output["window_anomaly"] = {"status": "unavailable", "reason": "未提供 samples"}
        else:
            try:
                output["window_anomaly"] = {"status": "ok", **self.predict_anomaly(samples)}
            except Exception as exc:
                output["window_anomaly"] = {"status": "unavailable", "reason": str(exc)}
        return output

    def _predict_multiout(self, parameters: dict[str, float]) -> dict[str, Any]:
        try:
            directory = self.root / "xgb_multiout"
            features = self._read_json(directory / "input_features.json")
            vector = [self._required(parameters, feature) for feature in features]
            crack = self._classifier("xgb_multiout/crack", directory / "crack" / "model.json")
            probability = self._positive_probability(crack, vector)
            geometry: dict[str, float] = {}
            for target in GEOMETRY_MODELS:
                model = self._regressor(
                    f"xgb_multiout/geo/{target}", directory / "geo" / target / "model.json"
                )
                geometry[target] = round(float(model.predict(self._array(vector))[0]), 4)
            return {
                "status": "ok",
                "crack_probability": round(probability, 6),
                "geometry": geometry,
            }
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc)}

    def _classifier(self, key: str, path: Path):
        if key not in self._models:
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(str(path))
            self._models[key] = model
        return self._models[key]

    def _regressor(self, key: str, path: Path):
        if key not in self._models:
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(str(path))
            self._models[key] = model
        return self._models[key]

    @staticmethod
    def _array(vector: list[float]):
        import numpy as np

        return np.asarray([vector], dtype=np.float32)

    def _positive_probability(self, model: Any, vector: list[float]) -> float:
        probabilities = model.predict_proba(self._array(vector))[0]
        return float(probabilities[-1])

    @staticmethod
    def _aggregate(values: list[float], operation: str) -> float:
        if operation == "mean":
            return fmean(values)
        if operation == "std":
            return pstdev(values)
        if operation == "min":
            return min(values)
        if operation == "max":
            return max(values)
        if operation == "last":
            return values[-1]
        if operation == "delta":
            return values[-1] - values[0]
        if operation == "rate":
            return sum(bool(v) for v in values) / len(values)
        if operation == "count_true":
            return float(sum(bool(v) for v in values))
        raise ModelInputError(f"不支持的窗口聚合: {operation}")

    @staticmethod
    def _required(mapping: dict[str, Any], key: str) -> float:
        if key not in mapping:
            raise ModelInputError(f"缺少模型输入字段: {key}")
        return float(mapping[key])

    @staticmethod
    def _canonical(name: str) -> str:
        low = name.strip().lower().replace("_", " ")
        aliases = (
            ("cross section", "cross_section_mm"),
            ("power", "power"),
            ("welding speed", "speed"),
            ("speed", "speed"),
            ("gas flow", "gas_flow"),
            ("gas", "gas_flow"),
            ("focal", "focal"),
            ("angular", "angular"),
            ("thickness", "thickness"),
        )
        for token, canonical in aliases:
            if token in low:
                return canonical
        return name

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _entry(name: str, task: str, directory: Path, model_count: int) -> dict[str, Any]:
        return {
            "name": name,
            "task": task,
            "model_count": model_count,
            "available": directory.exists() and any(directory.rglob("model.json")),
            "path": str(directory),
        }

