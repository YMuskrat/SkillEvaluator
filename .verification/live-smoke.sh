#!/usr/bin/env bash
# Test-only runner: use a real downloaded model, the real agent, and Docker.
set -euo pipefail
umask 077

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
smoke_root=${INTERACTION_SMOKE_ROOT:-${RUNNER_TEMP:?RUNNER_TEMP must be set}/skill-interaction-live}
mkdir -p "$smoke_root/downloads" "$smoke_root/server" "$smoke_root/artifacts" "$smoke_root/tmp" "$smoke_root/state"
chmod 700 "$smoke_root/state"
export TMPDIR="$smoke_root/tmp"
export SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE="$smoke_root/state/output-provenance.key"

llama_url=https://github.com/ggml-org/llama.cpp/releases/download/b12189/llama-b12189-bin-ubuntu-x64.tar.gz
model_url=https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
curl --fail --location --retry 3 --max-time 300 "$llama_url" -o "$smoke_root/downloads/llama.tar.gz"
curl --fail --location --retry 3 --max-time 600 "$model_url" -o "$smoke_root/downloads/model.gguf"
printf '%s  %s\n' c446a7ba4df3c3b1cc880a734fbff586d6d34b9cebc09e151754813b94ded8c9 "$smoke_root/downloads/llama.tar.gz" | sha256sum --check
printf '%s  %s\n' cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046 "$smoke_root/downloads/model.gguf" | sha256sum --check
# Use the original model author's pinned tool-aware chat template.
curl --fail --location --retry 3 --max-time 120 \
  https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/raw/2e1fd397ee46e1388853d2af2c993145b0f1098a/tokenizer_config.json \
  -o "$smoke_root/downloads/tokenizer_config.json"
printf '%s  %s\n' 959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4 "$smoke_root/downloads/tokenizer_config.json" | sha256sum --check
python - "$smoke_root/downloads" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
(root / "qwen-tool-aware.jinja").write_text(json.loads((root / "tokenizer_config.json").read_text())["chat_template"], encoding="utf-8")
PY
printf '%s  %s\n' cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f "$smoke_root/downloads/qwen-tool-aware.jinja" | sha256sum --check
tar -xzf "$smoke_root/downloads/llama.tar.gz" -C "$smoke_root/server"
llama_server=$(find "$smoke_root/server" -type f -name llama-server -print -quit)
test -n "$llama_server"
export LD_LIBRARY_PATH="$(dirname "$llama_server")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Harbor currently installs OpenCode @latest. Refuse drift and verify the
# actually executed version from each retained trajectory after the run.
test "$(npm view opencode-ai@latest version)" = 1.19.5
# Compose 5.7 handles image HEALTHCHECK NONE correctly during up --wait.
# The older runner image's Compose 2.38 treats disabled checks as missing.
compose_path="$HOME/.docker/cli-plugins/docker-compose"
mkdir -p "$(dirname "$compose_path")"
curl --fail --location --retry 3 --max-time 300 \
  https://github.com/docker/compose/releases/download/v5.7.0/docker-compose-linux-x86_64 \
  -o "$compose_path"
printf '%s  %s\n' 7758048ad38e580f22559603312874fec8bb638d021a9e64182385fc17cc12a2 "$compose_path" | sha256sum --check
chmod 755 "$compose_path"
docker compose version
docker info > "$smoke_root/artifacts/docker-info.txt"
gateway=$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')
test -n "$gateway"
export NO_PROXY="127.0.0.1,localhost,$gateway${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

"$llama_server" --model "$smoke_root/downloads/model.gguf" \
  --host "$gateway" --port 8080 --alias local-qwen --jinja \
  --chat-template-file "$smoke_root/downloads/qwen-tool-aware.jinja" \
  --ctx-size 32768 --parallel 1 --threads 2 --n-predict 512 --metrics \
  > "$smoke_root/artifacts/llama-server.log" 2>&1 &
server_pid=$!
cleanup() {
  curl --silent --show-error --max-time 5 "http://$gateway:8080/metrics" > "$smoke_root/artifacts/model-metrics.txt" || true
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
healthy=false
for attempt in $(seq 1 120); do
  if curl --fail --silent --max-time 2 "http://$gateway:8080/health" > "$smoke_root/artifacts/model-health.json"; then
    healthy=true
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -80 "$smoke_root/artifacts/llama-server.log"
    exit 1
  fi
  sleep 1
done
test "$healthy" = true

# This is an intentionally unauthenticated model server on an ephemeral
# runner. The required nonempty provider field is a local placeholder only.
export SKILL_EVAL_LLM_PROVIDER=openai-compatible
export SKILL_EVAL_LLM_BASE_URL="http://$gateway:8080/v1"
export SKILL_EVAL_MODEL_CATALOG_ALLOW_HTTP_HOSTS="$gateway"
export SKILL_EVAL_LLM_API_KEY=local-model-no-auth
export SKILL_EVAL_LLM_MODEL=local-qwen

PREFLIGHT_ARTIFACTS="$smoke_root/artifacts" python "$script_dir/model_tool_preflight.py"

python "$script_dir/live_smoke.py" prepare "$smoke_root/fixtures"
timeout --signal=TERM --kill-after=30s 24m \
  skillevaluator tier3 interactions \
    "$smoke_root/fixtures/answer-format" "$smoke_root/fixtures/answer-check" \
    --agents opencode --env-mode docker --agent-model opencode=openai/local-qwen \
    --grading-mode custom_only --metric custom:task_success \
    --n-attempts 1 --n-concurrent 1 --timeout-multiplier 1.5 \
    --results-dir "$smoke_root/artifacts/results" \
  2>&1 | tee "$smoke_root/artifacts/interaction-command.log"
python "$script_dir/live_smoke.py" verify "$smoke_root/artifacts/results" \
  --output "$smoke_root/artifacts/live-evidence.json"
