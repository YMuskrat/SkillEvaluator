# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI execution and reporting contract for skill interaction evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli


def _skills(tmp_path: Path, count: int = 2) -> list[str]:
    paths = []
    for index in range(count):
        skill = tmp_path / f"skill-{index}"
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"# Skill {index}\n", encoding="utf-8")
        paths.append(str(skill))
    return paths


def _result(interaction: float | None = -0.3, *, status: str = "complete") -> dict:
    return {
        "status": status,
        "agents": {
            "codex": {
                "status": status,
                "baseline_score": 0.5 if status == "complete" else None,
                "individual_scores": {"skill-0": 0.75, "skill-1": 0.7} if status == "complete" else {},
                "combined_score": 0.65 if status == "complete" else None,
                "combined_lift": 0.15 if status == "complete" else None,
                "interaction_score": interaction,
                "interaction_type": "negative" if interaction is not None and interaction < 0 else "positive",
            },
        },
        "result_path": "/results/interactions/result.json",
        "report_path": "/results/interactions/report.html",
        "configurations": [],
    }


def test_command_forwards_shared_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import interactions

    run = Mock(return_value=_result())
    monkeypatch.setattr(interactions, "run_interaction_evaluation", run)
    paths = _skills(tmp_path, 3)
    output = tmp_path / "reports"
    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "interactions",
            *paths,
            "--agents",
            "codex,opencode",
            "--env-mode",
            "gke",
            "--n-attempts",
            "3",
            "--n-concurrent",
            "2",
            "--model",
            "shared-model",
            "--agent-model",
            "codex=codex-model",
            "--metric",
            "custom:task_success",
            "--grading-mode",
            "custom_only",
            "--results-dir",
            str(output),
            "--timeout-multiplier",
            "2.5",
            "--environment-kwarg",
            "namespace=evals",
        ],
    )

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        tuple(Path(path) for path in paths),
        agents="codex,opencode",
        env_mode="gke",
        n_attempts=3,
        n_concurrent=2,
        model="shared-model",
        agent_model=("codex=codex-model",),
        metric="custom:task_success",
        grading_mode="custom_only",
        output_dir=output,
        timeout_multiplier=2.5,
        environment_kwarg=("namespace=evals",),
    )


def test_negative_interaction_is_a_completed_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import interactions

    run = Mock(return_value=_result())
    monkeypatch.setattr(interactions, "run_interaction_evaluation", run)
    result = CliRunner().invoke(cli, ["tier3", "interactions", *_skills(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Baseline: 0.5000" in result.output
    assert "skill-0 alone: 0.7500" in result.output
    assert "skill-1 alone: 0.7000" in result.output
    assert "Combined: 0.6500" in result.output
    assert "Combined lift: +0.1500" in result.output
    assert "Interaction: -0.3000 (negative)" in result.output
    assert "Results: /results/interactions/result.json" in result.output
    assert run.call_args.kwargs["metric"] == "goal_accuracy"
    assert run.call_args.kwargs["n_attempts"] == 1


def test_incomplete_matrix_exits_one_and_never_shows_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import interactions

    payload = _result(None, status="incomplete")
    payload["agents"]["codex"]["interaction_type"] = None
    payload["agents"]["codex"]["basis"] = {
        "compared_cases": 0,
        "expected_cases": 1,
        "excluded_cases": [{"case_id": "case-1", "reason": "missing_attempts"}],
    }
    payload["errors"] = ["Missing comparable attempts"]
    monkeypatch.setattr(interactions, "run_interaction_evaluation", Mock(return_value=payload))
    result = CliRunner().invoke(cli, ["tier3", "interactions", *_skills(tmp_path)])

    assert result.exit_code == 1
    assert "Interaction: N/A (unavailable)" in result.output
    assert "Combined lift: N/A" in result.output
    assert "Missing comparable attempts" in result.output
    assert "Comparable cases: 0/1" in result.output
    assert "case-1: missing_attempts" in result.output
    assert "0.0000" not in result.output


@pytest.mark.parametrize("count", [0, 1])
def test_two_skills_required_before_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    from skillevaluator.tier3 import interactions

    run = Mock()
    monkeypatch.setattr(interactions, "run_interaction_evaluation", run)
    result = CliRunner().invoke(cli, ["tier3", "interactions", *_skills(tmp_path, count)])

    assert result.exit_code == 2
    run.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        ["--n-attempts", "0"],
        ["--n-concurrent", "0"],
        ["--timeout-multiplier", "0"],
        ["--timeout-multiplier", "nan"],
        ["--timeout-multiplier", "inf"],
        ["--timeout-multiplier", "-inf"],
    ],
)
def test_invalid_budgets_fail_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    from skillevaluator.tier3 import interactions

    run = Mock()
    monkeypatch.setattr(interactions, "run_interaction_evaluation", run)
    result = CliRunner().invoke(cli, ["tier3", "interactions", *_skills(tmp_path), *options])

    assert result.exit_code == 2
    run.assert_not_called()


def test_runtime_validation_error_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import interactions

    monkeypatch.setattr(
        interactions, "run_interaction_evaluation", Mock(side_effect=ValueError("No shared eval suite"))
    )
    result = CliRunner().invoke(cli, ["tier3", "interactions", *_skills(tmp_path)])

    assert result.exit_code == 1
    assert "Error: No shared eval suite" in result.output


def test_names_and_errors_cannot_emit_terminal_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import interactions

    payload = _result()
    report = payload["agents"].pop("codex")
    report["individual_scores"] = {"skill_1": 0.5}
    payload["skill_names"] = {"skill_1": "[bold]skill\x1b[2J\nnext"}
    payload["agents"]["codex\x1b]0;title\x07"] = report
    payload["errors"] = ["failed\x1b[2J\rrewritten"]
    monkeypatch.setattr(interactions, "run_interaction_evaluation", Mock(return_value=payload))
    result = CliRunner().invoke(cli, ["tier3", "interactions", *_skills(tmp_path)])

    assert result.exit_code == 0
    assert "\x1b" not in result.output
    assert "\x07" not in result.output
    assert "[bold]skill next alone" in result.output
    assert "failed rewritten" in result.output


def test_help_does_not_import_live_evaluator() -> None:
    script = """
import importlib.abc
import sys
class NoInteractionEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'skillevaluator.tier3.interactions':
            raise AssertionError('Live evaluator imported for help')
sys.meta_path.insert(0, NoInteractionEngine())
from click.testing import CliRunner
from skillevaluator.cli import cli
result = CliRunner().invoke(cli, ['tier3', 'interactions', '--help'])
assert result.exit_code == 0, (result.output, result.exception)
assert '--metric' in result.output
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
