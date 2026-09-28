"""
Experiment Runner executing Reward Model Evaluation across Model Checkpoints.
"""

from typing import List, Dict
from preference_reward.schema import (
    PreferencePair,
    RewardModelVersion,
    RewardPrediction,
)
from preference_reward.evaluator import RewardEvaluator


class ExperimentRunner:
    """Runs reward model evaluation across RM checkpoints."""

    @staticmethod
    def evaluate_pair_across_checkpoints(pair: PreferencePair) -> Dict[RewardModelVersion, RewardPrediction]:
        evaluations = {}

        # 1. RM v1 Baseline
        r1 = pair.rewards_v1
        evaluations[RewardModelVersion.RM_V1_BASELINE] = RewardEvaluator.evaluate_pair(
            pair, RewardModelVersion.RM_V1_BASELINE, r1.get("chosen", 0.0), r1.get("rejected", 0.0)
        )

        # 2. RM v2 DPO Calibrated
        r2 = pair.rewards_v2
        evaluations[RewardModelVersion.RM_V2_DPO_CALIBRATED] = RewardEvaluator.evaluate_pair(
            pair, RewardModelVersion.RM_V2_DPO_CALIBRATED, r2.get("chosen", 0.0), r2.get("rejected", 0.0)
        )

        # 3. RM v3 Margin Scaled
        r3 = pair.rewards_v3
        evaluations[RewardModelVersion.RM_V3_MARGIN_SCALED] = RewardEvaluator.evaluate_pair(
            pair, RewardModelVersion.RM_V3_MARGIN_SCALED, r3.get("chosen", 0.0), r3.get("rejected", 0.0)
        )

        return evaluations
