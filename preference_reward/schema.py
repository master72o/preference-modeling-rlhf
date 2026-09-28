"""
Data Schemas for Preference Modeling & Reward Model Calibration.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


class RewardModelVersion(str, Enum):
    RM_V1_BASELINE = "RM_v1_Baseline"
    RM_V2_DPO_CALIBRATED = "RM_v2_DPO_Calibrated"
    RM_V3_MARGIN_SCALED = "RM_v3_Margin_Scaled"


@dataclass
class PreferencePair:
    __test__ = False

    pair_id: str
    prompt: str
    response_chosen: str
    response_rejected: str
    domain: str = "general"
    human_margin: float = 1.0
    rewards_v1: Dict[str, float] = field(default_factory=dict)
    rewards_v2: Dict[str, float] = field(default_factory=dict)
    rewards_v3: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencePair":
        return cls(
            pair_id=data["pair_id"],
            prompt=data["prompt"],
            response_chosen=data["response_chosen"],
            response_rejected=data["response_rejected"],
            domain=data.get("domain", "general"),
            human_margin=data.get("human_margin", 1.0),
            rewards_v1=data.get("rewards_v1", {"chosen": 0.5, "rejected": -0.5}),
            rewards_v2=data.get("rewards_v2", {"chosen": 1.2, "rejected": -0.8}),
            rewards_v3=data.get("rewards_v3", {"chosen": 2.1, "rejected": -1.5}),
        )


@dataclass
class RewardPrediction:
    pair_id: str
    model_version: RewardModelVersion
    reward_chosen: float
    reward_rejected: float
    reward_margin: float
    sigmoid_prob: float
    correct_prediction: bool
    log_loss: float
    brier_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "model_version": self.model_version.value if isinstance(self.model_version, RewardModelVersion) else str(self.model_version),
            "reward_chosen": round(self.reward_chosen, 4),
            "reward_rejected": round(self.reward_rejected, 4),
            "reward_margin": round(self.reward_margin, 4),
            "sigmoid_prob": round(self.sigmoid_prob, 4),
            "correct_prediction": bool(self.correct_prediction),
            "log_loss": round(self.log_loss, 4),
            "brier_score": round(self.brier_score, 4),
        }


class PreferenceDataset:
    """Dataset container for pairwise preference pairs."""

    def __init__(self, pairs: List[PreferencePair]):
        self.pairs = pairs

    @classmethod
    def from_jsonl(cls, file_path: str) -> "PreferenceDataset":
        import json
        pairs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(PreferencePair.from_dict(json.loads(line)))
        return cls(pairs=pairs)
