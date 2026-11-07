# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate baseline, individual, and combined skill evaluations."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from skillevaluator import __version__
from skillevaluator.tier3.case_ids import validate_case_ids
from skillevaluator.tier3.evals_config import load_evals_config, parse_environment_kwarg_overrides
from skillevaluator.tier3.harbor.adapter import (
    private_evaluator_skill_snapshot,
    validate_output_provenance_key_location,
    validate_results_root_location,
)
from skillevaluator.tier3.harbor.progress import redact_progress_detail, secret_values_from_environment
from skillevaluator.tier3.harbor.report_data import load_agent_data, load_dataset
from skillevaluator.tier3.harbor.secure_copy import copytree_secure, tree_content_fingerprint_secure
from skillevaluator.tier3.workflow import _validate_skill_root

MAX_INTERACTION_SKILLS = 20


def _validate_skills(skill_paths: Sequence[Path]) -> list[Path]:
    from skillevaluator.tier3.harbor.runner import _workspace_skill_name

    if not 2 <= len(skill_paths) <= MAX_INTERACTION_SKILLS:
        raise ValueError(f"Select between 2 and {MAX_INTERACTION_SKILLS} skills for interaction evaluation")
    roots: list[Path] = []
    names: set[str] = set()
    directory_names: set[str] = set()
    for raw in skill_paths:
        path = Path(raw).expanduser().absolute()
        _validate_skill_root(path)
        path = path.resolve(strict=True)
        if any(path == previous or path in previous.parents or previous in path.parents for previous in roots):
            raise ValueError("Interaction skills must be distinct, non-overlapping directories")
        name = unicodedata.normalize("NFC", _workspace_skill_name(path)).casefold()
        directory_name = unicodedata.normalize("NFC", path.name).casefold()
        if name in names or directory_name in directory_names:
            raise ValueError("Interaction skills must have distinct skill names and directory names")
        names.add(name)
        directory_names.add(directory_name)
        roots.append(path)
    return roots


def _validate_suite(evaluator: Path, *, metric: str, grading_mode: str | None) -> tuple[list[str], dict[str, Any]]:
    config, _ = load_evals_config(evaluator)
    workspace = config.get("skill_workspace", {})
    if workspace.get("include") or "decoy_manifest" in workspace:
        raise ValueError("Interaction evaluation requires a suite without configured companion skills or decoys")
    harbor = config.get("harbor", {})
    if harbor.get("pre_agent_setup"):
        raise ValueError("Interaction evaluation cannot use skill-owned pre-agent setup in its no-skill baseline")
    if harbor.get("task_source") == "native_harbor":
        raise ValueError("Interaction evaluation currently requires an evals/evals.* dataset")
    mode = grading_mode or config.get("grading", {}).get("mode", "default")
    if mode not in {"default", "default_plus_custom", "custom_only"}:
        raise ValueError("Unsupported interaction grading mode")
    if metric == "goal_accuracy":
        if mode == "custom_only":
            raise ValueError("goal_accuracy requires default or default_plus_custom grading")
    elif metric.startswith("custom:") and metric.removeprefix("custom:").strip():
        if mode == "default":
            raise ValueError("A custom interaction metric requires default_plus_custom or custom_only grading")
    else:
        raise ValueError("Interaction metric must be goal_accuracy or custom:NAME")
    entries = load_dataset(evaluator)
    if getattr(entries, "_report_truncation", None) or not entries:
        raise ValueError("Provide a nonempty, complete evals/evals.* dataset in the first skill")
    case_ids = validate_case_ids(entry.get("id") for entry in entries)
    return case_ids, config


def _arm_rewards(result: dict[str, Any], agents: list[str], *, baseline: bool) -> tuple[dict[str, list], list[str]]:
    """Use the bounded report reader and retain its execution/completeness gates."""
    run_dir = result.get("run_dir")
    if not isinstance(run_dir, str):
        return {}, ["The configuration produced no retained run directory"]
    loaded = load_agent_data(Path(run_dir))
    condition = "without_skill" if baseline else "with_skill"
    reward_key = "rewards_baseline" if baseline else "rewards"
    rows: dict[str, list] = {}
    errors: list[str] = []
    for agent in agents:
        info = loaded.get(agent, {})
        execution = info.get("conditions", {}).get(condition, {})
        if (
            execution.get("execution_status") != "succeeded"
            or info.get(f"{reward_key}_complete") is not True
            or info.get("_report_truncation")
        ):
            errors.append(f"{agent}: incomplete execution or retained trial evidence")
            rows[agent] = []
        else:
            rows[agent] = info.get(reward_key, [])
    return rows, errors


def _write_interaction_report(result: dict[str, Any], path: Path) -> None:
    """Write a small self-contained report without embedding untrusted markup."""
    sections: list[str] = []
    for agent, scores in result["agents"].items():
        skill_names = result.get("skill_names", {})
        values = [
            ("Baseline", scores.get("baseline_score")),
            *((skill_names.get(name, name), score) for name, score in scores.get("individual_scores", {}).items()),
            ("Combined", scores.get("combined_score")),
            ("Combined lift", scores.get("combined_lift")),
            ("Interaction", scores.get("interaction_score")),
        ]
        rows = "".join(
            f"<tr><th>{escape(name)}</th><td>{'N/A' if score is None else f'{score:.6f}'}</td></tr>"
            for name, score in values
        )
        sections.append(
            f"<section><h2>{escape(agent)}</h2><p>{escape(scores['status'])}; "
            f"interaction: {escape(str(scores.get('interaction_type') or 'unavailable'))}</p>"
            f"<table>{rows}</table></section>"
        )
    errors = "".join(f"<li>{escape(error)}</li>" for error in result["errors"])
    evidence = "".join(
        f"<li>{escape(arm['label'])}: "
        + (
            f'<a href="{escape(Path(arm["run_dir"]).relative_to(path.parent).as_posix(), quote=True)}/report.html">'
            "Configuration report</a>"
            if arm.get("run_dir") and (Path(arm["run_dir"]) / "report.html").is_file()
            else escape(arm["status"])
        )
        + "</li>"
        for arm in result["configurations"]
    )
    path.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>Skill interaction evaluation</title><style>body{font:16px system-ui;max-width:900px;"
        "margin:3rem auto;padding:0 1rem;color:#172033}table{border-collapse:collapse;width:100%;max-width:600px}"
        "th,td{text-align:left;padding:.6rem;border-bottom:1px solid #ccd3df}section{margin:2rem 0}</style>"
        f"<h1>Skill interaction evaluation</h1><p>Status: {escape(result['status'])}. "
        f"Metric: {escape(result['metric'])}.</p>"
        "<p>Interaction = combined lift minus the sum of individual lifts. Positive means superadditive, "
        "negative means subadditive, and zero means additive. Negative interaction can coexist with positive lift.</p>"
        + "".join(sections)
        + f"<h2>Configurations</h2><ul>{evidence}</ul><ul>{errors}</ul>"
        '<p><a href="interaction.json">Full scores, case evidence, and source identities (JSON)</a></p></html>',
        encoding="utf-8",
    )


def run_interaction_evaluation(
    skill_paths: Sequence[Path],
    *,
    agents: str | None = None,
    env_mode: str = "docker",
    n_attempts: int = 1,
    n_concurrent: int = 4,
    model: str | None = None,
    agent_model: tuple[str, ...] = (),
    metric: str = "goal_accuracy",
    grading_mode: str | None = None,
    output_dir: Path | None = None,
    timeout_multiplier: float | None = None,
    environment_kwarg: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Evaluate one exact group against its baseline and individual members.

    The first skill supplies the reviewed dataset and grader. One evaluator
    projection and all runtime projections are frozen before any agent runs.
    Every configuration uses the ordinary Tier 3 runner with equal budgets.
    """
    from skillevaluator.provider_config import resolve_llm_provider
    from skillevaluator.tier3.commands import parse_agent_model_overrides, parse_agents, resolve_agents, validate_agents
    from skillevaluator.tier3.decoy_identity import (
        DECOY_SOURCE_DIGEST_ALGORITHM,
        decoy_source_copy_ignore,
        decoy_source_digest,
    )
    from skillevaluator.tier3.harbor.runner import _model_for_agent, _reserve_run_dir, run_harbor_eval
    from skillevaluator.tier3.interaction_scores import calculate_interaction
    from skillevaluator.tier3_environments import is_env_mode

    for label, value in (("n_attempts", n_attempts), ("n_concurrent", n_concurrent)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if timeout_multiplier is not None and (not math.isfinite(timeout_multiplier) or timeout_multiplier <= 0):
        raise ValueError("timeout_multiplier must be finite and positive")
    if not is_env_mode(env_mode):
        raise ValueError("Unsupported interaction environment mode")
    roots = _validate_skills(skill_paths)
    labels = [f"skill_{index + 1}" for index in range(len(roots))]
    # Reuse the scorer's metric/arm contract before launching a paid run.
    calculate_interaction(
        {label: [] for label in ("baseline", *labels, "combined")},
        individual_arms=labels,
        metric=metric,
        n_attempts=n_attempts,
    )
    destination = Path(output_dir).expanduser().absolute() if output_dir is not None else roots[0] / "evals" / "results"
    validate_output_provenance_key_location(
        roots[0], destination, reference_skills_dir=None, workspace_skill_paths=roots[1:]
    )
    validate_results_root_location(roots[0], destination, workspace_skill_paths=roots[1:])
    environment_kwargs = parse_environment_kwarg_overrides(environment_kwarg, env_mode=env_mode)

    with (
        private_evaluator_skill_snapshot(roots[0]) as evaluator,
        tempfile.TemporaryDirectory(prefix="skillevaluator-interaction-") as temporary,
    ):
        case_ids, config = _validate_suite(evaluator, metric=metric, grading_mode=grading_mode)
        if config.get("harbor", {}).get("max_cases") is not None:
            raise ValueError("Interaction evaluation requires the complete suite; remove harbor.max_cases")
        # The shared report reader retains at most 512 logical trials per arm.
        # Refuse an unreportable budget before any agent work is launched.
        if len(case_ids) * n_attempts > 512:
            raise ValueError("Interaction evaluation supports at most 512 case attempts per configuration")
        provider = resolve_llm_provider()
        agent_list = parse_agents(agents) if agents is not None else resolve_agents(None, provider=provider.provider)
        if not agent_list or validate_agents(agent_list):
            raise ValueError("Select at least one supported Tier 3 agent")
        raw_overrides = parse_agent_model_overrides(agent_model)
        if set(raw_overrides) - set(agent_list):
            raise ValueError("--agent-model names an agent not selected by --agents")
        if any(len(values) != 1 for values in raw_overrides.values()):
            raise ValueError("Select one model per agent for an interaction evaluation")
        overrides = {agent: values[0] for agent, values in raw_overrides.items()}
        models = {
            agent: _model_for_agent(
                agent,
                cli_model=overrides.get(agent) or model,
                config_agents=config.get("harbor", {}).get("agents", {}),
                provider=provider,
            )[0]
            for agent in agent_list
        }
        suite_digest = tree_content_fingerprint_secure(evaluator / "evals")
        frozen: list[Path] = []
        identities: list[dict[str, str]] = []
        for index, root in enumerate(roots):
            digest = decoy_source_digest(root)
            if digest is None:
                raise ValueError(f"Cannot capture a stable runtime source for {root.name}")
            target = Path(temporary) / str(index) / root.name
            target.parent.mkdir()
            copytree_secure(root, target, allowed_root=root, ignore=decoy_source_copy_ignore(root))
            if decoy_source_digest(target) != digest or decoy_source_digest(root) != digest:
                raise ValueError(f"Skill runtime changed while being captured: {root.name}")
            copytree_secure(evaluator / "evals", target / "evals", allowed_root=evaluator / "evals")
            frozen.append(target)
            identities.append({"name": root.name, "runtime_digest": digest})

        run_dir = _reserve_run_dir(destination, "interaction_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
        plan = [
            ("baseline", "No skills", frozen[0], [], True),
            *[(label, root.name, target, [], False) for label, root, target in zip(labels, roots, frozen, strict=True)],
            ("combined", "All selected skills", frozen[0], frozen[1:], False),
        ]
        configurations = [
            {
                "id": arm,
                "label": label,
                "skills": [] if baseline else labels if arm == "combined" else [arm],
                "status": "not_run",
            }
            for arm, label, _target, _companions, baseline in plan
        ]
        all_rows: dict[str, dict[str, list]] = {agent: {arm: [] for arm, *_ in plan} for agent in agent_list}
        errors: list[str] = []
        route_identity: dict[str, Any] | None = None
        for record, (arm, _label, target, companions, baseline) in zip(configurations, plan, strict=True):
            try:
                result = run_harbor_eval(
                    skill_path=target,
                    agents=agent_list,
                    skip_baseline=not baseline,
                    skip_with_skill=baseline,
                    n_attempts=n_attempts,
                    stop_on_pass=False,
                    n_concurrent=n_concurrent,
                    agent_models=models,
                    skill_workspace_mode="group" if companions else "isolated",
                    include_skills=companions,
                    grading_mode=grading_mode,
                    output_dir=run_dir / "configurations" / arm,
                    env_mode=env_mode,
                    environment_kwargs=environment_kwargs,
                    timeout_multiplier=timeout_multiplier,
                )
            except Exception as exc:
                # The runner owns detailed/redacted diagnostics. Never copy an
                # arbitrary provider exception (which may contain secrets).
                result = {"error": [type(exc).__name__]}
            if isinstance(result.get("run_dir"), str):
                record["run_dir"] = result["run_dir"]
            rows, arm_errors = _arm_rewards(result, agent_list, baseline=baseline)
            if raw_errors := result.get("error"):
                details = raw_errors if isinstance(raw_errors, list) else [raw_errors]
                secrets = secret_values_from_environment(os.environ)
                arm_errors.extend(redact_progress_detail(detail, secret_values=secrets) for detail in details[:16])
            run_config = result.get("run_config", {})
            identity = {name: run_config.get(name) for name in ("provider", "judge", "agents")}
            if route_identity is None:
                route_identity = identity
            elif identity != route_identity:
                arm_errors.append("Agent or scorer configuration changed between configurations")
            record["status"] = "incomplete" if arm_errors else "complete"
            if arm_errors:
                record["errors"] = arm_errors
                rows = {}
            for agent, rewards in rows.items():
                all_rows[agent][arm] = rewards
            errors.extend(f"{arm}: {error}" for error in arm_errors)
            if arm_errors:
                break

        scores = {
            agent: calculate_interaction(
                arms,
                individual_arms=labels,
                metric=metric,
                expected_case_ids=case_ids,
                n_attempts=n_attempts,
            )
            for agent, arms in all_rows.items()
        }
        complete = not errors and all(score["status"] == "complete" for score in scores.values())
        result_path = run_dir / "interaction.json"
        report_path = run_dir / "interaction.html"
        output = {
            "schema_version": 1,
            "evaluator_version": __version__,
            "status": "complete" if complete else "incomplete",
            "metric": metric,
            "skill_names": dict(zip(labels, (root.name for root in roots), strict=True)),
            "skill_sources": identities,
            "runtime_digest_algorithm": DECOY_SOURCE_DIGEST_ALGORITHM,
            "suite_digest": suite_digest,
            "suite_digest_algorithm": "skill-evaluator-secure-tree/1",
            "suite_source": roots[0].name,
            "case_ids": case_ids,
            "agents": scores,
            "configurations": configurations,
            "errors": errors,
            "result_path": str(result_path),
            "report_path": str(report_path),
        }
        result_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_interaction_report(output, report_path)
        return output
