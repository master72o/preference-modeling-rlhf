# Pairwise Preference & Reward Modeling Dataset

This directory contains pairwise preference datasets used to evaluate reward model predictions ($r_{\text{chosen}}$ vs $r_{\text{rejected}}$) and Bradley-Terry sigmoid probabilities across model checkpoints.

## Dataset Schema (`preference_dataset.jsonl`)

```json
{
  "pair_id": "pref_001",
  "domain": "reasoning",
  "prompt": "How to optimize SQL query performance?",
  "response_chosen": "Use proper indexing, avoid SELECT *, and analyze execution plans.",
  "response_rejected": "Just restart the database server.",
  "human_margin": 1.0,
  "rewards_v1": {"chosen": 0.5, "rejected": -0.2},
  "rewards_v2": {"chosen": 1.5, "rejected": -0.8},
  "rewards_v3": {"chosen": 2.8, "rejected": -1.8}
}
```
