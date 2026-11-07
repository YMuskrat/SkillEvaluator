# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interaction orchestration through secure snapshots and retained Tier 3 evidence."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.tier3 import interactions
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS


def _skill(parent: Path, name: str, *, skill_name: str | None = None, suite: bool = False) -> Path:
    root = parent / name
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {skill_name or name}\ndescription: Perform the {name} workflow.\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text(f"# Runtime for {name}\n", encoding="utf-8")
    if suite:
        (root / "evals" / "files").mkdir(parents=True)
        (root / "evals" / "files" / "input.txt").write_text("Shared task input\n", encoding="utf-8")
        (root / "evals" / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": name,
                    "evals": [{"id": "case-one", "prompt": "Solve the task", "expected_output": "The expected answer"}],
                }
            ),
            encoding="utf-8",
        )
    return root


def _config(root: Path, value: dict[str, Any]) -> None:
    (root / "evals" / "config.yml").write_text(yaml.safe_dump(value), encoding="utf-8")


class _RetainedRuns:
    """Use real report reading while replacing only paid agent execution."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.scores = {"baseline": 0.5, "skill_1": 0.75, "skill_2": 0.7, "combined": 0.65}
        self.failed: set[str] = set()
        self.missing: set[str] = set()
        self.routes: dict[str, str] = {}
        self.custom = False
        self.on_call: Callable[[str], None] | None = None
        self.mutate_reward: Callable[[str, dict[str, Any]], None] | None = None

    def __call__(self, **options: Any) -> dict[str, Any]:
        arm = options["output_dir"].name
        target = options["skill_path"]
        self.calls.append(
            {
                **options,
                "runtime": (target / "scripts" / "run.py").read_text(encoding="utf-8"),
                "suite": (target / "evals" / "evals.json").read_bytes(),
                "input": (target / "evals" / "files" / "input.txt").read_bytes(),
                "available": ([] if options["skip_with_skill"] else [target.name])
                + [path.name for path in options["include_skills"]],
                "companions_have_evals": [bool((path / "evals").exists()) for path in options["include_skills"]],
            }
        )
        if self.on_call:
            self.on_call(arm)
        run_dir = options["output_dir"] / "run"
        baseline = options["skip_with_skill"]
        for agent in options["agents"]:
            for variant in ("with-skill", "without-skill"):
                active = (variant == "without-skill") == baseline
                condition_dir = run_dir / agent / variant
                condition_dir.mkdir(parents=True)
                summary = {
                    "scores": {},
                    "custom_scores": {},
                    "metrics": list(DEFAULT_METRICS) if active and not self.custom else [],
                    "num_trials": options["n_attempts"] if active else 0,
                    "num_reward_rows": options["n_attempts"] if active else 0,
                    "execution_status": "failed"
                    if active and arm in self.failed
                    else "succeeded"
                    if active
                    else "skipped",
                    "expected_attempts": options["n_attempts"] if active else 0,
                    "scored_attempts": options["n_attempts"] if active else 0,
                }
                (condition_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                if not active or arm in self.missing:
                    continue
                for attempt in range(options["n_attempts"]):
                    trial = condition_dir / "trials" / f"case-one-{attempt + 1}"
                    trial.mkdir(parents=True)
                    score = self.scores[arm]
                    reward = {
                        "entry_id": "case-one",
                        "trial_id": f"trial-{attempt + 1}",
                        "goal_accuracy": score,
                        "metric_set": DEFAULT_METRIC_SET,
                        "details": {"goal_accuracy": {"judge_model_used": "test-judge"}},
                    }
                    if self.custom:
                        reward = {
                            "entry_id": "case-one",
                            "trial_id": f"trial-{attempt + 1}",
                            "metric_set": "custom-only",
                            "custom_metrics": {"task_success": score},
                            "overall": score,
                        }
                    if self.mutate_reward:
                        self.mutate_reward(arm, reward)
                    (trial / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
        (run_dir / "report.html").write_text("<!doctype html><title>Configuration</title>", encoding="utf-8")
        return {
            "run_dir": str(run_dir),
            "run_config": {
                "provider": {"name": "test", "model": "test-judge"},
                "judge": {"model": self.routes.get(arm, "test-judge")},
                "agents": {agent: {"model": "test-agent"} for agent in options["agents"]},
            },
        }


@pytest.fixture
def run_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], _RetainedRuns]:
    from skillevaluator import provider_config
    from skillevaluator.tier3.harbor import runner

    skills = [_skill(tmp_path / "skills", "alpha", suite=True), _skill(tmp_path / "skills", "beta")]
    fake = _RetainedRuns()
    monkeypatch.setattr(
        provider_config,
        "resolve_llm_provider",
        lambda: ProviderConfig(
            provider="openai", model="test-judge", api_key=None, base_url=None, litellm_model="test-judge"
        ),
    )
    monkeypatch.setattr(runner, "run_harbor_eval", fake)
    # Bound test-owned source and temporary roots separately from the checkout.
    # The existing provenance-key fixture remains outside both Git roots.
    subprocess.run(["git", "init", "--quiet", str(skills[0].parent)], check=True, capture_output=True)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    subprocess.run(["git", "init", "--quiet", str(temporary)], check=True, capture_output=True)
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    return skills, fake


def _run(skills: list[Path], tmp_path: Path, **options: Any) -> dict[str, Any]:
    return interactions.run_interaction_evaluation(skills, agents="codex", output_dir=tmp_path / "results", **options)


def test_two_skills_run_exact_four_conditions_with_one_frozen_suite(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path
) -> None:
    skills, fake = run_setup
    # Beta's unrelated evaluation sources must not replace the shared alpha suite.
    (skills[1] / "evals").mkdir()
    (skills[1] / "evals" / "grader.py").write_text("raise RuntimeError('wrong suite')", encoding="utf-8")
    result = _run(skills, tmp_path, n_attempts=2)

    assert result["status"] == "complete"
    assert [call["available"] for call in fake.calls] == [[], ["alpha"], ["beta"], ["alpha", "beta"]]
    assert len({call["suite"] for call in fake.calls}) == 1
    assert len({call["input"] for call in fake.calls}) == 1
    assert [call["runtime"] for call in fake.calls] == [
        "# Runtime for alpha\n",
        "# Runtime for alpha\n",
        "# Runtime for beta\n",
        "# Runtime for alpha\n",
    ]
    assert all(call["n_attempts"] == 2 and call["stop_on_pass"] is False for call in fake.calls)
    assert all(call["agent_models"] == {"codex": "test-judge"} for call in fake.calls)
    assert result["agents"]["codex"]["interaction_score"] == pytest.approx(-0.3)
    assert result["agents"]["codex"]["combined_lift"] == pytest.approx(0.15)
    assert result["agents"]["codex"]["interaction_type"] == "negative"
    assert json.loads(Path(result["result_path"]).read_text())["status"] == "complete"
    assert "Interaction" in Path(result["report_path"]).read_text()
    assert all(not call["skill_path"].exists() for call in fake.calls), "private source copies must be cleaned up"
    assert (skills[1] / "evals" / "grader.py").read_text() == "raise RuntimeError('wrong suite')"


def test_three_skills_run_baseline_singletons_and_combined(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path
) -> None:
    skills, fake = run_setup
    skills.append(_skill(tmp_path / "skills", "gamma"))
    fake.scores = {"baseline": 0.1, "skill_1": 0.2, "skill_2": 0.3, "skill_3": 0.2, "combined": 0.8}
    result = _run(skills, tmp_path)

    assert [call["available"] for call in fake.calls] == [
        [],
        ["alpha"],
        ["beta"],
        ["gamma"],
        ["alpha", "beta", "gamma"],
    ]
    assert result["agents"]["codex"]["interaction_score"] == pytest.approx(0.3)
    assert result["agents"]["codex"]["interaction_type"] == "positive"


@pytest.mark.parametrize("combined, expected", [(0.65, "negative"), (0.95, "zero"), (1.0, "positive")])
def test_custom_outcome_metric_uses_retained_numeric_evidence(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, combined: float, expected: str
) -> None:
    skills, fake = run_setup
    fake.custom = True
    fake.scores["combined"] = combined
    _config(skills[0], {"schema_version": 1, "grading": {"mode": "custom_only"}})
    result = _run(skills, tmp_path, metric="custom:task_success")

    assert result["status"] == "complete"
    assert result["agents"]["codex"]["interaction_type"] == expected
    assert result["agents"]["codex"]["interaction_score"] == pytest.approx(combined - 0.95)


def test_source_edits_after_snapshot_do_not_change_later_configurations(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path
) -> None:
    skills, fake = run_setup

    def mutate(arm: str) -> None:
        if arm == "baseline":
            (skills[1] / "scripts" / "run.py").write_text("# Changed after capture\n", encoding="utf-8")
            (skills[0] / "evals" / "files" / "input.txt").write_text("Changed authored input\n", encoding="utf-8")

    fake.on_call = mutate
    result = _run(skills, tmp_path)

    assert result["status"] == "complete"
    assert fake.calls[2]["runtime"] == "# Runtime for beta\n"
    assert all(call["input"] == b"Shared task input\n" for call in fake.calls)
    assert (skills[1] / "scripts" / "run.py").read_text() == "# Changed after capture\n"


def test_source_mutation_during_runtime_capture_stops_before_agent_work(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills, fake = run_setup
    original_copy = interactions.copytree_secure

    def mutate_after_copy(source: Path, destination: Path, **options: Any) -> None:
        original_copy(source, destination, **options)
        if source == skills[1]:
            (source / "scripts" / "run.py").write_text("# Changed during capture\n", encoding="utf-8")

    monkeypatch.setattr(interactions, "copytree_secure", mutate_after_copy)
    with pytest.raises(ValueError, match="changed while being captured"):
        _run(skills, tmp_path)
    assert fake.calls == []


@pytest.mark.parametrize("shape", ["same_root", "nested", "same_name", "same_directory_name"])
def test_rejects_overlapping_or_ambiguous_skills_before_agent_work(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, shape: str
) -> None:
    skills, fake = run_setup
    if shape == "same_root":
        skills[1] = skills[0]
    elif shape == "nested":
        skills[1] = _skill(skills[0], "nested")
    elif shape == "same_name":
        (skills[1] / "SKILL.md").write_text("---\nname: ALPHA\ndescription: Duplicate.\n---\n", encoding="utf-8")
    else:
        skills[1] = _skill(tmp_path / "other", "alpha", skill_name="different")

    with pytest.raises(ValueError, match=r"distinct|non-overlapping"):
        _run(skills, tmp_path)
    assert fake.calls == []


@pytest.mark.parametrize(
    "config, metric",
    [
        ({"skill_workspace": {"mode": "group", "include": ["beta"]}}, "goal_accuracy"),
        ({"skill_workspace": {"mode": "group", "decoy_manifest": "decoys.json"}}, "goal_accuracy"),
        ({"harbor": {"pre_agent_setup": ["echo setup"]}}, "goal_accuracy"),
        ({"grading": {"mode": "custom_only"}}, "goal_accuracy"),
        ({}, "overall"),
        ({}, "custom:task_success"),
    ],
)
def test_invalid_suite_policy_never_resolves_provider_or_runs_agents(
    run_setup: tuple[list[Path], _RetainedRuns],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    metric: str,
) -> None:
    from skillevaluator import provider_config

    skills, fake = run_setup
    _config(skills[0], {"schema_version": 1, **config})
    provider = Mock(side_effect=AssertionError("provider must not be resolved"))
    monkeypatch.setattr(provider_config, "resolve_llm_provider", provider)
    with pytest.raises(ValueError):
        _run(skills, tmp_path, metric=metric)
    provider.assert_not_called()
    assert fake.calls == []


@pytest.mark.parametrize("target", ["anchor", "companion", "evaluator_input", "parent_of_skills"])
def test_output_collision_is_rejected_without_touching_authored_inputs(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, target: str
) -> None:
    skills, fake = run_setup
    output = {
        "anchor": skills[0] / "scripts" / "results",
        "companion": skills[1] / "reports",
        "evaluator_input": skills[0] / "evals" / "files" / "results",
        "parent_of_skills": skills[0].parent,
    }[target]
    before = (skills[0] / "SKILL.md").read_bytes()
    with pytest.raises(ValueError):
        interactions.run_interaction_evaluation(skills, agents="codex", output_dir=output)
    assert fake.calls == []
    assert (skills[0] / "SKILL.md").read_bytes() == before


@pytest.mark.parametrize("fault", ["execution_failed", "missing_rewards", "changed_route"])
def test_invalid_configuration_evidence_stops_remaining_paid_runs(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, fault: str
) -> None:
    skills, fake = run_setup
    if fault == "execution_failed":
        fake.failed.add("skill_1")
    elif fault == "missing_rewards":
        fake.missing.add("skill_1")
    else:
        fake.routes["skill_1"] = "changed-judge"
    result = _run(skills, tmp_path)

    assert len(fake.calls) == 2
    assert result["status"] == "incomplete"
    assert [record["status"] for record in result["configurations"]] == ["complete", "incomplete", "not_run", "not_run"]
    assert result["agents"]["codex"]["interaction_score"] is None
    assert result["errors"]
    assert Path(result["result_path"]).is_file()


def test_unknown_judge_provenance_withholds_interaction(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path
) -> None:
    skills, fake = run_setup

    def remove_identity(_arm: str, reward: dict[str, Any]) -> None:
        reward.pop("details", None)

    fake.mutate_reward = remove_identity
    result = _run(skills, tmp_path)

    assert result["status"] == "incomplete"
    assert result["agents"]["codex"]["interaction_score"] is None
    assert {case["reason"] for case in result["agents"]["codex"]["basis"]["excluded_cases"]} == {"judge_unknown"}


def test_real_task_projection_and_custom_grader_preserve_evaluator_isolation(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import sys

    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

    skills, fake = run_setup
    fake.custom = True
    _config(skills[0], {"schema_version": 1, "grading": {"mode": "custom_only"}})
    (skills[0] / "evals" / "grader.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "answer = (Path(os.environ['HARBOR_LOGS_DIR']) / 'agent' / 'answer.txt').read_text()\n"
        "score = float(answer == '42')\n"
        "Path(os.environ['HARBOR_REWARD_JSON']).write_text(json.dumps({'overall': score, 'custom_metrics': {'task_success': score}}))\n",
        encoding="utf-8",
    )
    (skills[1] / "evals").mkdir()
    (skills[1] / "evals" / "grader.py").write_text("raise RuntimeError('wrong grader')", encoding="utf-8")
    observed: list[list[str]] = []

    def stage_and_grade(arm: str) -> None:
        call = fake.calls[-1]
        tasks = generate_harbor_tasks(
            call["skill_path"],
            tmp_path / "staged" / arm,
            with_skill=not call["skip_with_skill"],
            workspace_skill_paths=call["include_skills"],
            workspace_mode=call["skill_workspace_mode"],
            grading_mode="custom_only",
        )
        task = tasks[0]
        staged_skills = task / "environment" / "skills"
        available = sorted(path.name for path in staged_skills.iterdir())
        observed.append(available)
        for name in available:
            runtime = staged_skills / name
            assert (runtime / "scripts" / "run.py").is_file()
            assert not (runtime / "evals").exists()
        tests = task / "tests"
        logs = tmp_path / "grader-logs" / arm
        (logs / "agent").mkdir(parents=True)
        (logs / "verifier").mkdir()
        (logs / "agent" / "answer.txt").write_text("42" if arm in {"skill_1", "skill_2"} else "wrong")
        reward = logs / "verifier" / "reward.json"
        wrapper = next(tests.rglob("custom_grader_runner.py"))
        environment = {
            **os.environ,
            "HARBOR_LOGS_DIR": str(logs),
            "HARBOR_TESTS_DIR": str(tests),
            "HARBOR_REWARD_JSON": str(reward),
        }
        completed = subprocess.run(
            [sys.executable, str(wrapper), "--mode", "custom_only"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        fake.scores[arm] = json.loads(reward.read_text())["task_success"]

    # The current process is already under the test's isolated environment;
    # no model client or infrastructure executable is invoked by this stub.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    fake.on_call = stage_and_grade
    result = _run(skills, tmp_path, metric="custom:task_success")

    assert observed == [[], ["alpha"], ["beta"], ["alpha", "beta"]]
    assert result["status"] == "complete"
    assert result["agents"]["codex"]["interaction_score"] == -2.0


def test_explicit_agent_model_is_reused_across_all_configurations(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path
) -> None:
    skills, fake = run_setup
    result = _run(skills, tmp_path, agent_model=("codex=chosen-agent-model",))

    assert result["status"] == "complete"
    assert all(call["agent_models"] == {"codex": "chosen-agent-model"} for call in fake.calls)


def test_multiple_models_for_one_agent_are_rejected_before_execution(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path
) -> None:
    skills, fake = run_setup
    with pytest.raises(ValueError, match="only one model"):
        _run(skills, tmp_path, agent_model=("codex=first-model", "codex=second-model"))
    assert fake.calls == []


@pytest.mark.parametrize("config", [{"harbor": {"max_cases": 1}}, {}])
def test_incomplete_or_oversized_case_selection_is_rejected_before_execution(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, config: dict[str, Any]
) -> None:
    skills, fake = run_setup
    _config(skills[0], {"schema_version": 1, **config})
    with pytest.raises(ValueError, match=r"max_cases|512"):
        _run(skills, tmp_path, n_attempts=1 if config else 513)
    assert fake.calls == []


def test_missing_runtime_prerequisite_remains_actionable_and_redacted(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3.harbor import runner

    skills, _fake = run_setup
    credential = "interaction-test-private-credential-123456789"
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    run = Mock(
        return_value={"error": [f"Docker daemon is unavailable. Check Docker startup. Credential: {credential}"]}
    )
    monkeypatch.setattr(runner, "run_harbor_eval", run)
    result = _run(skills, tmp_path)

    assert result["status"] == "incomplete"
    assert run.call_count == 1
    assert any("Docker daemon is unavailable" in error for error in result["errors"])
    assert "no retained run directory" in " ".join(result["errors"])
    assert credential not in json.dumps(result)
    assert credential not in Path(result["report_path"]).read_text()
    assert all(score["interaction_score"] is None for score in result["agents"].values())


def test_unexpected_runner_exception_never_publishes_its_message(
    run_setup: tuple[list[Path], _RetainedRuns], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3.harbor import runner

    skills, _fake = run_setup
    message = "Unrecognized provider details may contain private request material"
    run = Mock(side_effect=RuntimeError(message))
    monkeypatch.setattr(runner, "run_harbor_eval", run)
    result = _run(skills, tmp_path)

    assert result["status"] == "incomplete"
    assert run.call_count == 1
    assert any("RuntimeError" in error for error in result["errors"])
    assert message not in json.dumps(result)
    assert message not in Path(result["report_path"]).read_text()
