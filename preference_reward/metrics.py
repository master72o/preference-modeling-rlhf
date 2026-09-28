"""
Metrics Aggregator & Expected Calibration Error (ECE) Calculator.
"""

import numpy as np
from typing import List, Dict, Any
from preference_reward.schema import RewardPrediction, RewardModelVersion


class RewardMetricsCalculator:
    """Calculates pairwise accuracy, ECE, log loss, and Brier score."""

    @staticmethod
    def calculate_checkpoint_metrics(predictions: List[RewardPrediction]) -> Dict[str, Any]:
        if not predictions:
            return {"total_pairs": 0, "accuracy": 0.0}

        total = len(predictions)
        correct = sum(1 for p in predictions if p.correct_prediction)
        accuracy = correct / total

        margins = [p.reward_margin for p in predictions]
        probs = [p.sigmoid_prob for p in predictions]
        log_losses = [p.log_loss for p in predictions]
        brier_scores = [p.brier_score for p in predictions]

        # Calculate Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Collect items in bin
            bin_indices = [
                j for j, p in enumerate(probs)
                if (p > bin_lower and p <= bin_upper) or (i == 0 and p >= bin_lower and p <= bin_upper)
            ]

            if bin_indices:
                bin_acc = np.mean([1.0 if predictions[j].correct_prediction else 0.0 for j in bin_indices])
                bin_conf = np.mean([probs[j] for j in bin_indices])
                ece += (len(bin_indices) / total) * abs(bin_acc - bin_conf)

        return {
            "total_pairs": total,
            "accuracy": round(accuracy, 4),
            "mean_margin": round(float(np.mean(margins)), 4),
            "mean_log_loss": round(float(np.mean(log_losses)), 4),
            "mean_brier_score": round(float(np.mean(brier_scores)), 4),
            "expected_calibration_error": round(float(ece), 4),
        }

    @classmethod
    def calculate_suite_metrics(
        cls, all_predictions: Dict[RewardModelVersion, List[RewardPrediction]]
    ) -> Dict[str, Any]:
        suite_metrics = {}
        for rm_ver, preds in all_predictions.items():
            key = rm_ver.value if isinstance(rm_ver, RewardModelVersion) else str(rm_ver)
            suite_metrics[key] = cls.calculate_checkpoint_metrics(preds)
        return suite_metrics


MetricsCalculator = RewardMetricsCalculator
