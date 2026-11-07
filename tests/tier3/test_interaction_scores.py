# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interaction arithmetic and the complete shared-case scoring contract."""

from __future__ import annotations

import copy
import json

import pytest

from skillevaluator.tier3.interaction_scores import calculate_interaction


def _reward(score: float, *, case: str = "task-1", trial: str = "trial-1", judge: str = "judge-a") -> dict:
    return {
        "entry_id": case,
        "trial_id": trial,
        "goal_accuracy": score,
        "details": {"goal_accuracy": {"judge_model_used": judge}},
    }


def _arms(*scores: float) -> dict[str, list[dict]]:
    labels = ["baseline", *[f"skill-{index}" for index in range(1, len(scores) - 1)], "combined"]
    return {name: [_reward(score)] for name, score in zip(labels, scores, strict=True)}


def _calculate(arms: dict[str, list[dict]], **kwargs) -> dict:
    individuals = [name for name in arms if name not in {"baseline", "combined"}]
    return calculate_interaction(arms, individual_arms=individuals, **kwargs)


def test_supplied_example_reports_negative_interaction_and_positive_combined_lift() -> None:
    result = _calculate(_arms(0.50, 0.75, 0.70, 0.65))

    assert result["status"] == "complete"
    assert result["baseline_score"] == 0.50
    assert result["individual_scores"] == {"skill-1": 0.75, "skill-2": 0.70}
    assert result["combined_score"] == 0.65
    assert result["combined_lift"] == pytest.approx(0.15)
    assert result["interaction_score"] == pytest.approx(-0.30)
    assert result["interaction_type"] == "negative"
    assert result["case_results"][0]["interaction_score"] == pytest.approx(-0.30)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("combined", "expected", "kind"),
    [(0.9, 0.1, "positive"), (0.8, 0.0, "zero"), (0.7, -0.1, "negative")],
)
def test_classifies_departure_from_additive_individual_lifts(combined: float, expected: float, kind: str) -> None:
    result = _calculate(_arms(0.2, 0.5, 0.5, combined))

    assert result["interaction_score"] == pytest.approx(expected)
    assert result["interaction_type"] == kind


def test_three_skills_reports_total_nonadditivity() -> None:
    result = _calculate(_arms(0.2, 0.3, 0.4, 0.5, 0.9))

    assert result["skills"] == ["skill-1", "skill-2", "skill-3"]
    assert result["individual_lifts"] == pytest.approx({"skill-1": 0.1, "skill-2": 0.2, "skill-3": 0.3})
    assert result["combined_lift"] == pytest.approx(0.7)
    assert result["interaction_score"] == pytest.approx(0.1)


def test_averages_repeated_trials_with_equal_case_weighting() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    for rows in arms.values():
        rows.append(_reward(1.0, trial="trial-2"))
        rows.extend([_reward(0.0, case="task-2", trial="trial-3"), _reward(0.0, case="task-2", trial="trial-4")])

    result = _calculate(arms, n_attempts=2, expected_case_ids=["task-1", "task-2"])

    assert result["status"] == "complete"
    assert result["baseline_score"] == pytest.approx(0.3)
    assert result["combined_score"] == pytest.approx(0.45)
    assert result["interaction_score"] == pytest.approx(0.075)
    assert result["basis"]["compared_cases"] == 2


def test_excluded_case_keeps_diagnostics_but_withholds_all_aggregate_scores() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    for name, rows in arms.items():
        if name != "skill-2":
            rows.append(_reward(1.0, case="task-2", trial="trial-2"))

    result = _calculate(arms, expected_case_ids=["task-1", "task-2"])

    assert result["status"] == "incomplete"
    assert result["interaction_score"] is None
    assert result["combined_score"] is None
    assert result["individual_scores"] == {}
    assert result["case_results"][0]["case_id"] == "task-1"
    assert result["basis"]["excluded_cases"] == [
        {"case_id": "task-2", "reason": "attempt_count_mismatch", "arms": ["skill-2"]}
    ]


@pytest.mark.parametrize("bad_score", [None, float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_metric_does_not_become_a_zero_score(bad_score: object) -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    arms["combined"][0]["goal_accuracy"] = bad_score

    result = _calculate(arms)

    assert result["interaction_score"] is None
    assert result["basis"]["excluded_cases"] == [{"case_id": "task-1", "reason": "not_scored", "arms": ["combined"]}]
    json.dumps(result, allow_nan=False)


def test_explicit_not_applicable_is_reported() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    arms["combined"][0]["goal_accuracy"] = None
    arms["combined"][0]["details"]["goal_accuracy"]["status"] = "not_applicable"

    result = _calculate(arms)

    assert result["basis"]["excluded_cases"][0]["reason"] == "not_applicable"


@pytest.mark.parametrize("status", ["error", "failed", "ERROR"])
def test_failed_trial_is_unscored_even_when_a_metric_survives(status: str) -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    arms["combined"][0]["evaluation_status"] = status

    result = _calculate(arms)

    assert result["basis"]["excluded_cases"][0]["reason"] == "not_scored"


@pytest.mark.parametrize("missing", [True, False])
def test_unknown_or_mixed_judge_makes_case_incomparable(missing: bool) -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    detail = arms["combined"][0]["details"]["goal_accuracy"]
    if missing:
        detail.clear()
    else:
        detail["judge_model_used"] = "fallback-judge"

    result = _calculate(arms)

    assert result["status"] == "incomplete"
    assert result["basis"]["excluded_cases"][0]["reason"] == ("judge_unknown" if missing else "judge_mixed")


def test_ragas_and_prompt_judge_with_same_model_are_different_scorers() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    arms["combined"][0]["details"]["goal_accuracy"]["method"] = "ragas"

    result = _calculate(arms)

    assert result["basis"]["excluded_cases"][0]["reason"] == "judge_mixed"


def test_multistep_judge_models_are_checked_across_all_arms() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    arms["combined"][0]["details"]["goal_accuracy"] = {"judge_models": ["judge-a", "judge-b"]}

    assert _calculate(arms)["basis"]["excluded_cases"][0]["reason"] == "judge_mixed"


def test_duplicate_physical_trial_is_not_counted_as_another_attempt() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    for rows in arms.values():
        rows.append(_reward(0.5, trial="trial-2"))
    arms["combined"][1] = copy.deepcopy(arms["combined"][0])

    result = _calculate(arms, n_attempts=2)

    assert result["basis"]["excluded_cases"] == [
        {"case_id": "task-1", "reason": "duplicate_trial", "arms": ["combined"]}
    ]


def test_unknown_trial_identity_is_not_silently_used() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    del arms["combined"][0]["trial_id"]

    assert _calculate(arms)["basis"]["excluded_cases"][0]["reason"] == "unidentified_trial"


def test_unexpected_or_unidentified_cases_prevent_a_complete_result() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    arms["combined"].extend(
        [_reward(1.0, case="unexpected", trial="extra-1"), _reward(1.0, case="unknown", trial="extra-2")]
    )

    result = _calculate(arms, expected_case_ids=["task-1"])

    assert result["status"] == "incomplete"
    assert result["interaction_score"] is None
    assert result["basis"]["excluded_cases"][0]["reason"] == "unexpected_case"
    assert result["basis"]["unidentified_rows"] == {"combined": 1}
    assert len(result["case_results"]) == 1


def test_no_cases_is_incomplete() -> None:
    arms = {name: [] for name in _arms(0.2, 0.3, 0.4, 0.8)}

    result = _calculate(arms)

    assert result["status"] == "incomplete"
    assert result["interaction_score"] is None


def test_custom_only_outcome_uses_existing_custom_metric_shapes_without_judge() -> None:
    arms = _arms(0.5, 0.75, 0.7, 0.65)
    for rows in arms.values():
        row = rows[0]
        row["custom_metrics"] = {"task_success": {"score": row.pop("goal_accuracy")}}
        row["metric_set"] = "custom-only"
        del row["details"]

    result = _calculate(arms, metric="custom:task_success")

    assert result["status"] == "complete"
    assert result["interaction_score"] == pytest.approx(-0.3)


@pytest.mark.parametrize("metric", ["overall", "accuracy", "skill_execution", "custom:", "custom:goal_accuracy", "bad"])
def test_rejects_unsupported_or_reserved_metric_names(metric: str) -> None:
    with pytest.raises(ValueError, match="Interaction metric"):
        _calculate(_arms(0.2, 0.3, 0.4, 0.8), metric=metric)


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
def test_rejects_invalid_attempt_budget(attempts: object) -> None:
    with pytest.raises(ValueError, match="n_attempts"):
        _calculate(_arms(0.2, 0.3, 0.4, 0.8), n_attempts=attempts)


@pytest.mark.parametrize("individuals", [["skill-1"], ["skill-1", "skill-1"], ["baseline", "skill-1"]])
def test_rejects_incomplete_or_colliding_arm_configuration(individuals: list[str]) -> None:
    with pytest.raises(ValueError, match="arm"):
        calculate_interaction(_arms(0.2, 0.3, 0.4, 0.8), individual_arms=individuals)


def test_rejects_unexpected_arm_instead_of_ignoring_it() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    arms["unused"] = []

    with pytest.raises(ValueError, match="exactly"):
        calculate_interaction(arms, individual_arms=["skill-1", "skill-2"])


def test_rejects_duplicate_expected_case_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        _calculate(_arms(0.2, 0.3, 0.4, 0.8), expected_case_ids=["task-1", "task-1"])


def test_nested_metric_scores_are_supported_without_mutating_inputs() -> None:
    arms = _arms(0.2, 0.3, 0.4, 0.8)
    for rows in arms.values():
        rows[0]["metrics"] = {"goal_accuracy": {"score": rows[0].pop("goal_accuracy")}}
    before = copy.deepcopy(arms)

    result = _calculate(arms)

    assert result["status"] == "complete"
    assert arms == before
