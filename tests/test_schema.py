"""
Tests for Schema module in Preference Modeling & Reward Calibration.
"""

from preference_reward.schema import PreferencePair, RewardModelVersion


def test_preference_pair_parsing():
    data = {
        "pair_id": "p1",
        "prompt": "prompt",
        "response_chosen": "chosen",
        "response_rejected": "rejected",
        "human_margin": 1.0,
    }
    pair = PreferencePair.from_dict(data)
    assert pair.pair_id == "p1"
    assert pair.response_chosen == "chosen"


def test_reward_model_version_enum():
    assert RewardModelVersion.RM_V1_BASELINE.value == "RM_v1_Baseline"
