"""
Tests for Experiment Runner and Metrics in Preference Modeling & Reward Calibration.
"""

from preference_reward.schema import PreferencePair, RewardModelVersion, PreferenceDataset
from preference_reward.experiment_runner import ExperimentRunner
from preference_reward.metrics import MetricsCalculator


def test_experiment_runner():
    pair = PreferencePair(
        pair_id="p1",
        prompt="prompt",
        response_chosen="chosen",
        response_rejected="rejected",
        rewards_v1={"chosen": 0.5, "rejected": 0.0},
        rewards_v2={"chosen": 1.5, "rejected": -0.5},
        rewards_v3={"chosen": 3.0, "rejected": -2.0},
    )
    res_dict = ExperimentRunner.evaluate_pair_across_checkpoints(pair)
    assert RewardModelVersion.RM_V1_BASELINE in res_dict
    assert res_dict[RewardModelVersion.RM_V3_MARGIN_SCALED].correct_prediction is True


def test_metrics_calculator(tmp_path):
    jsonl = '{"pair_id":"p1","prompt":"p","response_chosen":"c","response_rejected":"r"}\n'
    f_path = tmp_path / "dataset.jsonl"
    f_path.write_text(jsonl)

    dataset = PreferenceDataset.from_jsonl(str(f_path))
    all_preds = {rm_v: [] for rm_v in RewardModelVersion}
    for pair in dataset.pairs:
        res = ExperimentRunner.evaluate_pair_across_checkpoints(pair)
        for rm_v, p in res.items():
            all_preds[rm_v].append(p)

    metrics = MetricsCalculator.calculate_suite_metrics(all_preds)
    assert "RM_v1_Baseline" in metrics
    assert metrics["RM_v1_Baseline"]["total_pairs"] == 1
    assert "expected_calibration_error" in metrics["RM_v1_Baseline"]
