#!/usr/bin/env python3
"""Verify real model output follows the tool protocol before the live agent run."""
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

artifacts = Path(os.environ['PREFLIGHT_ARTIFACTS'])
base = os.environ['SKILL_EVAL_LLM_BASE_URL'].rstrip('/')
props_base = base.removesuffix('/v1')
with urlopen(props_base + '/props', timeout=30) as response:
    props = json.load(response)
(artifacts / 'model-props.json').write_text(json.dumps(props, indent=2))
# The explicit, pinned template is also retained in the downloaded config.
templates = json.dumps({key: value for key, value in props.items() if 'template' in key})
assert '<tool_call>' in templates, 'No native tools-aware template reported by llama.cpp'

for mode in ('auto', 'required'):
    payload = {
        'model': 'local-qwen',
        'messages': [
            {'role': 'system', 'content': 'Use a provided tool when requested.'},
            {'role': 'user', 'content': 'Call the smoke_echo tool now with value 42. Use the actual tool call protocol.'},
        ],
        'tools': [{
            'type': 'function',
            'function': {
                'name': 'smoke_echo',
                'description': 'Return the supplied integer for an inference protocol smoke test.',
                'parameters': {
                    'type': 'object',
                    'properties': {'value': {'type': 'integer'}},
                    'required': ['value'],
                    'additionalProperties': False,
                },
            },
        }],
        'tool_choice': mode,
        'temperature': 0,
        'max_tokens': 128,
        'stream': False,
    }
    request = Request(base + '/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    (artifacts / ('model-tool-preflight-' + mode + '.json')).write_text(json.dumps(result, indent=2))
    calls = result['choices'][0]['message'].get('tool_calls', [])
    assert len(calls) == 1, (mode, result)
    function = calls[0]['function']
    assert function['name'] == 'smoke_echo', (mode, result)
    assert json.loads(function['arguments']) == {'value': 42}, (mode, result)
    assert result['usage']['completion_tokens'] > 0, (mode, result)
    print(mode + ': real native tool call verified')
