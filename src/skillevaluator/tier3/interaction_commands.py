# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command boundary for live evaluation of skill interactions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import click

from skillevaluator.cli_help import RichCommand
from skillevaluator.tier3_environments import all_env_modes
from skillevaluator.utils.rich_markup import strip_terminal_controls


def _finite_timeout(_ctx: click.Context, _param: click.Parameter, value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise click.BadParameter("must be a finite number greater than zero")
    return value


def _text(value: object) -> str:
    return " ".join(strip_terminal_controls(str(value)).split())


def _score(value: object, *, signed: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "N/A"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _print_result(result: dict[str, Any], metric: str) -> None:
    click.echo(f"Skill interactions ({_text(metric)})")
    skill_names = result.get("skill_names", {})
    for agent, report in result.get("agents", {}).items():
        click.echo(f"  {_text(agent)}: {_text(report.get('status', 'incomplete'))}")
        click.echo(f"    Baseline: {_score(report.get('baseline_score'))}")
        for skill, score in report.get("individual_scores", {}).items():
            click.echo(f"    {_text(skill_names.get(skill, skill))} alone: {_score(score)}")
        click.echo(f"    Combined: {_score(report.get('combined_score'))}")
        click.echo(f"    Combined lift: {_score(report.get('combined_lift'), signed=True)}")
        click.echo(
            f"    Interaction: {_score(report.get('interaction_score'), signed=True)} "
            f"({_text(report.get('interaction_type') or 'unavailable')})"
        )
        if report.get("status") != "complete":
            basis = report.get("basis", {})
            if "compared_cases" in basis and "expected_cases" in basis:
                click.echo(f"    Comparable cases: {basis['compared_cases']}/{basis['expected_cases']}")
            excluded = basis.get("excluded_cases", [])
            for case in excluded[:5]:
                click.echo(f"    {_text(case.get('case_id', 'unknown'))}: {_text(case.get('reason', 'incomparable'))}")
            if len(excluded) > 5:
                click.echo(f"    {len(excluded) - 5} further excluded cases; see the JSON report.")
    errors = result.get("errors") or result.get("error") or ()
    if isinstance(errors, str):
        errors = (errors,)
    if isinstance(errors, dict):
        errors = errors.values()
    for error in errors:
        click.echo(f"  {_text(error)}", err=True)
    for label, key in (("Results", "result_path"), ("Report", "report_path")):
        if result.get(key):
            click.echo(f"{label}: {_text(result[key])}")


@click.command("interactions", cls=RichCommand, context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("skill_paths", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-a", "--agents", help="Comma-separated agents; defaults to the configured provider's agent.")
@click.option("--env-mode", type=click.Choice(list(all_env_modes())), default="docker", show_default=True)
@click.option("--n-attempts", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--n-concurrent", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--model", help="Agent model override, shared by every configuration.")
@click.option("--agent-model", multiple=True, help="Per-agent AGENT=MODEL override (repeatable).")
@click.option(
    "--metric", default="goal_accuracy", show_default=True, help="Outcome score: goal_accuracy or custom:NAME."
)
@click.option("--grading-mode", type=click.Choice(["default", "default_plus_custom", "custom_only"]))
@click.option(
    "--results-dir",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Results root; defaults to the first skill's evals/results directory.",
)
@click.option(
    "--timeout-multiplier",
    type=click.FloatRange(min=0, min_open=True),
    callback=_finite_timeout,
    help="Finite positive multiplier for Harbor timeouts.",
)
@click.option("--environment-kwarg", "--ek", multiple=True, help="Operator-only backend KEY=VALUE option (repeatable).")
def interactions_command(
    skill_paths: tuple[Path, ...],
    agents: str | None,
    env_mode: str,
    n_attempts: int,
    n_concurrent: int,
    model: str | None,
    agent_model: tuple[str, ...],
    metric: str,
    grading_mode: str | None,
    output_dir: Path | None,
    timeout_multiplier: float | None,
    environment_kwarg: tuple[str, ...],
) -> None:
    """Evaluate two or more skills alone and together on one fixed task suite.

    The first skill supplies the existing evaluation cases and grader. Run a
    no-skill baseline, each skill alone, and the combined group, then compare
    combined lift with the sum of individual lifts. No dataset is generated.
    """
    if len(skill_paths) < 2:
        raise click.UsageError("Select at least two skill directories.")

    from skillevaluator.tier3.interactions import run_interaction_evaluation

    click.echo(f"Evaluating {len(skill_paths)} skills across {len(skill_paths) + 2} configurations...")
    try:
        result = run_interaction_evaluation(
            skill_paths,
            agents=agents,
            env_mode=env_mode,
            n_attempts=n_attempts,
            n_concurrent=n_concurrent,
            model=model,
            agent_model=agent_model,
            metric=metric,
            grading_mode=grading_mode,
            output_dir=output_dir,
            timeout_multiplier=timeout_multiplier,
            environment_kwarg=environment_kwarg,
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(_text(exc)) from exc
    _print_result(result, metric)
    if result.get("status") != "complete":
        raise click.exceptions.Exit(1)
