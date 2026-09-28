"""
Tests for Reward Evaluator Engine.
"""

from preference_reward.schema import PreferencePair, RewardModelVersion
from preference_reward.evaluator import RewardEvaluator


def test_evaluate_pair_accurate():
    pair = PreferencePair(
        pair_id="p1",
        prompt="prompt",
        response_chosen="good",
        response_rejected="bad",
    )
    res = RewardEvaluator.evaluate_pair(pair, RewardModelVersion.RM_V3_MARGIN_SCALED, 2.5, -1.5)
    assert res.reward_margin == 4.0
    assert res.correct_prediction is True
    assert res.sigmoid_prob > 0.95
    assert res.brier_score < 0.05


def test_evaluate_pair_inaccurate():
    pair = PreferencePair(
        pair_id="p2",
        prompt="prompt",
        response_chosen="good",
        response_rejected="bad",
    )
    res = RewardEvaluator.evaluate_pair(pair, RewardModelVersion.RM_V1_BASELINE, -1.0, 1.0)
    assert res.reward_margin == -2.0
    assert res.correct_prediction is False
    assert res.sigmoid_prob < 0.5
