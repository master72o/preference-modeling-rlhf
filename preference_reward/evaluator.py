"""
Reward Model Evaluator & Bradley-Terry Calibration Engine.
"""

import numpy as np
from typing import Dict, Any
from preference_reward.schema import (
    PreferencePair,
    RewardModelVersion,
    RewardPrediction,
)


class RewardEvaluator:
    """Evaluates pairwise reward model predictions against human preference ground truth."""

    @staticmethod
    def evaluate_pair(
        pair: PreferencePair,
        model_version: RewardModelVersion,
        reward_chosen: float,
        reward_rejected: float
    ) -> RewardPrediction:
        reward_margin = reward_chosen - reward_rejected

        # Bradley-Terry Sigmoid Probability P(chosen > rejected) = 1 / (1 + exp(-margin))
        sigmoid_prob = float(1.0 / (1.0 + np.exp(-reward_margin)))
        sigmoid_prob = max(1e-6, min(1.0 - 1e-6, sigmoid_prob))

        correct_prediction = reward_margin > 0.0
        log_loss = float(-np.log(sigmoid_prob))
        brier_score = float((sigmoid_prob - 1.0) ** 2)

        return RewardPrediction(
            pair_id=pair.pair_id,
            model_version=model_version,
            reward_chosen=round(reward_chosen, 4),
            reward_rejected=round(reward_rejected, 4),
            reward_margin=round(reward_margin, 4),
            sigmoid_prob=round(sigmoid_prob, 4),
            correct_prediction=correct_prediction,
            log_loss=round(log_loss, 4),
            brier_score=round(brier_score, 4),
        )
