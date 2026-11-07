# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculate skill interactions on a complete, shared outcome-scoring basis.

For two skills this is ``P(A,B) - P(A) - P(B) + P(empty)``. For a larger
group it is the group's lift minus the sum of its singleton lifts: total
nonadditivity, rather than an isolated higher-order interaction coefficient.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from skillevaluator.tier3.harbor.metrics import (
    RESERVED_METRIC_NAMES,
    custom_metric_name_is_publishable,
    extract_custom_metrics,
    metric_is_not_applicable,
    metric_set_for_reward,
    metric_value,
)

_ZERO_TOLERANCE = 1e-12
_UNKNOWN_CASE_IDS = frozenset({"", "unknown"})


def _validate_arguments(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    individual_arms: Sequence[str],
    baseline_arm: str,
    combined_arm: str,
    metric: str,
    expected_case_ids: Sequence[str],
    n_attempts: int,
) -> tuple[list[str], str | None]:
    if not isinstance(arms, Mapping):
        raise ValueError("Interaction arms must be a mapping")
    if not isinstance(individual_arms, Sequence) or isinstance(individual_arms, str | bytes):
        raise ValueError("Individual arm names must be a sequence")
    if not isinstance(expected_case_ids, Sequence) or isinstance(expected_case_ids, str | bytes):
        raise ValueError("Expected case IDs must be a sequence")
    individuals = list(individual_arms)
    labels = [baseline_arm, *individuals, combined_arm]
    if len(individuals) < 2:
        raise ValueError("Interaction evaluation requires at least two individual skill arms")
    if any(not isinstance(label, str) or not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Interaction arm names must be non-empty and distinct")
    if set(arms) != set(labels):
        raise ValueError("Interaction arms must contain exactly the baseline, individual, and combined arms")
    if isinstance(n_attempts, bool) or not isinstance(n_attempts, int) or n_attempts < 1:
        raise ValueError("n_attempts must be a positive integer")
    if any(not isinstance(case, str) or case in _UNKNOWN_CASE_IDS for case in expected_case_ids):
        raise ValueError("Expected case IDs must be non-empty, identified strings")
    if len(set(expected_case_ids)) != len(expected_case_ids):
        raise ValueError("Expected case IDs must be unique")
    custom_name = metric.removeprefix("custom:") if isinstance(metric, str) and metric.startswith("custom:") else None
    if metric != "goal_accuracy" and (
        custom_name is None
        or not custom_name
        or custom_name in RESERVED_METRIC_NAMES
        or not custom_metric_name_is_publishable(custom_name)
    ):
        raise ValueError("Interaction metric must be goal_accuracy or custom:<metric-name>")
    return labels, custom_name


def _judge_models(reward: Mapping[str, Any], metric: str) -> set[str]:
    """Read the same judge identity shapes persisted by the Harbor collector."""
    details = reward.get("details")
    detail = details.get(metric) if isinstance(details, Mapping) else None
    if not isinstance(detail, Mapping):
        return set()
    models = detail.get("judge_models")
    if isinstance(models, list):
        return {model for model in models if isinstance(model, str) and model.strip()}
    model = detail.get("judge_model_used")
    if not isinstance(model, str) or not model.strip():
        return set()
    return {f"{model} (ragas)" if detail.get("method") == "ragas" else model}


def _metric_score(reward: Mapping[str, Any], metric: str, custom_name: str | None) -> float | None:
    if str(reward.get("evaluation_status") or "").casefold() in {"error", "failed"}:
        return None
    data = dict(reward)
    if custom_name is not None:
        return extract_custom_metrics(data).get(custom_name)
    _, metrics = metric_set_for_reward(data)
    return metric_value(data, metric) if metric in metrics else None


def _interaction_values(
    scores: Mapping[str, float], individual_arms: Sequence[str], baseline_arm: str, combined_arm: str
) -> dict[str, Any]:
    baseline = scores[baseline_arm]
    combined = scores[combined_arm]
    lifts = {arm: scores[arm] - baseline for arm in individual_arms}
    combined_lift = combined - baseline
    interaction = math.fsum([combined_lift, *(-value for value in lifts.values())])
    if math.isclose(interaction, 0.0, abs_tol=_ZERO_TOLERANCE):
        interaction = 0.0
    return {
        "baseline_score": baseline,
        "individual_scores": {arm: scores[arm] for arm in individual_arms},
        "combined_score": combined,
        "individual_lifts": lifts,
        "combined_lift": combined_lift,
        "interaction_score": interaction,
        "interaction_type": "positive" if interaction > 0 else "negative" if interaction < 0 else "zero",
    }


def _case_failure(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    custom_name: str | None,
    n_attempts: int,
    duplicate_trials: set[str],
) -> str | None:
    if len(rows) != n_attempts:
        return "attempt_count_mismatch"
    if any(not isinstance(row.get("trial_id"), str) or not row["trial_id"].strip() for row in rows):
        return "unidentified_trial"
    if any(row["trial_id"] in duplicate_trials for row in rows):
        return "duplicate_trial"
    if custom_name is None and any(metric_is_not_applicable(dict(row), metric) for row in rows):
        return "not_applicable"
    if any(_metric_score(row, metric, custom_name) is None for row in rows):
        return "not_scored"
    if custom_name is None and any(not _judge_models(row, metric) for row in rows):
        return "judge_unknown"
    return None


def calculate_interaction(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    individual_arms: Sequence[str],
    baseline_arm: str = "baseline",
    combined_arm: str = "combined",
    metric: str = "goal_accuracy",
    expected_case_ids: Sequence[str] = (),
    n_attempts: int = 1,
) -> dict[str, Any]:
    """Compare baseline, singleton, and combined rewards on the same cases.

    Rows use the collector's saved ``reward.json`` shape, including ``entry_id``
    and ``trial_id``. Every case must have exactly ``n_attempts`` valid, distinct
    trials in every arm. ``goal_accuracy`` additionally requires one known judge
    identity across every contributing trial of a case. ``custom:<name>`` uses
    the existing custom-metric contract for an operator-owned outcome grader.

    Incomplete matrices retain valid per-case diagnostics but publish no
    aggregate interaction. Cases are weighted equally after averaging their
    repeated attempts. Positive/negative/zero describes observed departure
    from additivity; it does not assert statistical significance or causality.
    """
    labels, custom_name = _validate_arguments(
        arms, individual_arms, baseline_arm, combined_arm, metric, expected_case_ids, n_attempts
    )
    by_case: dict[str, dict[str, list[Mapping[str, Any]]]] = {arm: {} for arm in labels}
    unidentified = dict.fromkeys(labels, 0)
    duplicate_trials: dict[str, set[str]] = {}
    for arm in labels:
        rows = arms[arm]
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            raise ValueError("Each interaction arm must contain a sequence of reward mappings")
        trial_ids: Counter[str] = Counter()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Each interaction reward must be a mapping")
            trial_id = row.get("trial_id")
            if isinstance(trial_id, str) and trial_id.strip():
                trial_ids[trial_id] += 1
            case_id = row.get("entry_id")
            if not isinstance(case_id, str) or case_id in _UNKNOWN_CASE_IDS:
                unidentified[arm] += 1
                continue
            by_case[arm].setdefault(case_id, []).append(row)
        duplicate_trials[arm] = {trial for trial, count in trial_ids.items() if count > 1}

    observed_cases = set().union(*(set(cases) for cases in by_case.values()))
    case_ids = list(expected_case_ids) if expected_case_ids else sorted(observed_cases)
    excluded: list[dict[str, Any]] = [
        {"case_id": case_id, "reason": "unexpected_case", "arms": [arm for arm in labels if case_id in by_case[arm]]}
        for case_id in sorted(observed_cases - set(case_ids))
    ]
    case_results: list[dict[str, Any]] = []
    case_scores: list[dict[str, float]] = []
    for case_id in case_ids:
        failures: dict[str, list[str]] = {}
        for arm in labels:
            reason = _case_failure(
                by_case[arm].get(case_id, ()),
                metric=metric,
                custom_name=custom_name,
                n_attempts=n_attempts,
                duplicate_trials=duplicate_trials[arm],
            )
            if reason is not None:
                failures.setdefault(reason, []).append(arm)
        if not failures and custom_name is None:
            models = set().union(*(_judge_models(row, metric) for arm in labels for row in by_case[arm][case_id]))
            if len(models) != 1:
                failures["judge_mixed"] = labels
        if failures:
            excluded.extend({"case_id": case_id, "reason": reason, "arms": names} for reason, names in failures.items())
            continue
        scores = {
            arm: math.fsum(_metric_score(row, metric, custom_name) for row in by_case[arm][case_id]) / n_attempts
            for arm in labels
        }
        case_scores.append(scores)
        case_results.append(
            {"case_id": case_id, **_interaction_values(scores, individual_arms, baseline_arm, combined_arm)}
        )

    complete = bool(case_results) and not excluded and not any(unidentified.values())
    values = {
        "baseline_score": None,
        "individual_scores": {},
        "combined_score": None,
        "individual_lifts": {},
        "combined_lift": None,
        "interaction_score": None,
        "interaction_type": None,
    }
    if complete:
        scores = {arm: math.fsum(case[arm] for case in case_scores) / len(case_scores) for arm in labels}
        values = _interaction_values(scores, individual_arms, baseline_arm, combined_arm)
    return {
        "schema_version": 1,
        "metric": metric,
        "skills": list(individual_arms),
        "formula": "P(S) - P(empty) - sum(P(s) - P(empty) for s in S)",
        "status": "complete" if complete else "incomplete",
        **values,
        "case_results": case_results,
        "basis": {
            "expected_cases": len(case_ids),
            "compared_cases": len(case_results),
            "attempts_per_case": n_attempts,
            "excluded_cases": excluded,
            "unidentified_rows": {arm: count for arm, count in unidentified.items() if count},
        },
    }
