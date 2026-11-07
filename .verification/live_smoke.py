#!/usr/bin/env python3
"""Prepare and verify a real, local-model skill interaction smoke run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROMPT = """Complete this small local file task. Use your shell tool; do not merely describe commands.
First list /workspace/skills. Read each SKILL.md file present there, if any.
Then write the file /workspace/answer.txt containing exactly the two digits 42 followed by one newline.
Read the file back to check it. Finally respond with Done.
Work only with /workspace/skills and /workspace/answer.txt. Do not use the network.
"""

GRADER = '''import json
import os
from pathlib import Path

answer_path = Path("/workspace/answer.txt")
try:
    answer = answer_path.read_bytes()
except OSError:
    answer = None
skills_root = Path("/workspace/skills")
available = sorted(path.name for path in skills_root.iterdir() if (path / "SKILL.md").is_file())
score = float(answer == b"42\\n")
result = {
    "overall": score,
    "custom_metrics": {"task_success": score},
    "details": {
        "task_success": {
            "answer_file_present": answer is not None,
            "answer_correct": answer == b"42\\n",
            "available_skills": available,
        }
    },
}
Path(os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json")).write_text(json.dumps(result))
'''


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    skills = (
        ("answer-format", "Use the shell tool to write /workspace/answer.txt with exactly 42 and a newline. Keep the file to one line."),
        ("answer-check", "Use the shell tool to read /workspace/answer.txt back after writing it. Confirm its content is exactly 42 and a newline."),
    )
    for name, instructions in skills:
        skill = root / name
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Help complete the local answer.txt file task.\n---\n\n# {name}\n\n{instructions}\n",
            encoding="utf-8",
        )
    evals = root / skills[0][0] / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps({
            "skill_name": skills[0][0],
            "evals": [{"id": "write-answer", "prompt": PROMPT, "expected_output": "The file /workspace/answer.txt contains exactly 42 followed by a newline.", "files": []}],
        }),
        encoding="utf-8",
    )
    (evals / "config.yml").write_text(
        "schema_version: 1\ngrading:\n  mode: custom_only\nharbor:\n  n_attempts: 1\n  n_concurrent: 1\n  stop_on_pass: false\n  timeout_multiplier: 1.5\n  resources:\n    cpus: 2\n    memory_mb: 2048\n",
        encoding="utf-8",
    )
    (evals / "grader.py").write_text(GRADER, encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify(results_root: Path, output: Path) -> None:
    files = list(results_root.glob("interaction_*/interaction.json"))
    _require(len(files) == 1, f"Expected one interaction result; found {len(files)}")
    result = json.loads(files[0].read_text())
    _require(result.get("status") == "complete", f"Incomplete interaction result: {result.get('errors')}")
    _require(result.get("metric") == "custom:task_success", "Unexpected interaction metric")
    _require(result.get("case_ids") == ["write-answer"], "Unexpected case set")
    expected_available = {
        "baseline": [],
        "skill_1": ["answer-format"],
        "skill_2": ["answer-check"],
        "combined": ["answer-check", "answer-format"],
    }
    records = result.get("configurations", [])
    _require({row["id"] for row in records} == set(expected_available), "Expected exactly four configurations")
    _require(all(row.get("status") == "complete" for row in records), "A configuration did not complete")
    sources = result.get("skill_sources", [])
    _require({row["name"] for row in sources} == {"answer-format", "answer-check"}, "Missing runtime skill identity")
    _require(all(len(row.get("runtime_digest", "")) == 64 for row in sources), "Missing runtime skill digest")
    score = result["agents"]["opencode"]
    for key in ("baseline_score", "combined_score", "combined_lift", "interaction_score"):
        value = score.get(key)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"Non-finite {key}")
    _require(len(score["individual_scores"]) == 2, "Expected two individual scores")
    _require(score["basis"]["attempts_per_case"] == 1 and score["basis"]["compared_cases"] == 1, "Wrong comparison basis")
    arm_evidence = []
    for record in records:
        variant = "without-skill" if record["id"] == "baseline" else "with-skill"
        trial_roots = list((Path(record["run_dir"]) / "opencode" / variant / "trials").iterdir())
        _require(len(trial_roots) == 1, f"Expected one retained trial for {record['id']}")
        trial = trial_roots[0]
        reward = json.loads((trial / "reward.json").read_text())
        details = reward.get("custom_details", {}).get("task_success", {})
        _require(details.get("available_skills") == expected_available[record["id"]], f"Wrong available skills in {record['id']}: {details}")
        trajectory = json.loads((trial / "trajectory.json").read_text())
        agent = trajectory.get("agent", {})
        _require(agent.get("name") == "opencode", "Expected actual OpenCode trajectory")
        _require(agent.get("version") == "1.19.5", f"OpenCode version drift: {agent.get('version')}")
        _require("local-qwen" in agent.get("model_name", ""), "Wrong live model identity")
        completions = trajectory.get("final_metrics", {}).get("total_completion_tokens")
        _require(isinstance(completions, (int, float)) and completions > 0, "No actual generated completion tokens")
        calls = sum(len(step.get("tool_calls") or []) for step in trajectory.get("steps", []))
        arm_evidence.append({
            "configuration": record["id"],
            "available_skills": details["available_skills"],
            "task_success": reward["custom_metrics"]["task_success"],
            "agent_version": agent["version"],
            "model_name": agent["model_name"],
            "completion_tokens": completions,
            "tool_calls": calls,
        })
    _require(sum(arm["tool_calls"] for arm in arm_evidence) > 0, "The live agent never used a tool")
    evidence = {
        "status": "passed",
        "interaction_result": str(files[0]),
        "outcomes_all_pass": all(arm["task_success"] == 1 for arm in arm_evidence),
        "interaction_score": score["interaction_score"],
        "combined_lift": score["combined_lift"],
        "arms": arm_evidence,
    }
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    else:
        _require(args.output is not None, "--output is required when verifying")
        verify(args.root, args.output)
