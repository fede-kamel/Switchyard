# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from litellm.types.router import RoutingContext
from switchyard_litellm.plugins import CompositeRoutingPlugin, serving_judge_call
from switchyard_litellm.plugins.composite_routing_plugin import (
    DECISION_HEADER,
    decision_headers,
    has_tool_history,
)

CAPABLE = "provider/capable"
EFFICIENT = "provider/efficient"


class Recorder:
    """Stands in for a Switchyard plugin: records the call and narrows to one model."""

    def __init__(self, pick: str, algorithm: str) -> None:
        self.pick = pick
        self.algorithm = algorithm
        self.calls = 0

    async def run(self, context: RoutingContext) -> RoutingContext:
        self.calls += 1
        context.candidate_models = [self.pick]
        context.signals["switchyard"] = {
            "selected_model_id": self.pick,
            "fallback_models": [],
            "algorithm": self.algorithm,
            "evidence": {},
        }
        return context


def routing_context(messages: list[dict[str, Any]], candidates: list[str]) -> RoutingContext:
    return RoutingContext(
        raw_messages=messages,
        structured_messages=messages,
        candidate_models=candidates,
        metadata={},
        signals={},
    )


def composite() -> tuple[CompositeRoutingPlugin, Recorder, Recorder]:
    classifier = Recorder(EFFICIENT, "llm_classifier")
    stage = Recorder(CAPABLE, "stage_router")
    plugin = CompositeRoutingPlugin(classifier=classifier, stage=stage)  # type: ignore[arg-type]
    return plugin, classifier, stage


TOOL_HISTORY = [
    {"role": "user", "content": "Fix it."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "FAILED"},
]


def test_tool_history_detection() -> None:
    assert not has_tool_history([{"role": "user", "content": "hi"}])
    assert has_tool_history(TOOL_HISTORY)
    assert has_tool_history([{"role": "assistant", "content": None, "tool_calls": [{}]}])


async def test_fresh_task_goes_to_the_classifier() -> None:
    plugin, classifier, stage = composite()
    result = await plugin.run(
        routing_context([{"role": "user", "content": "Rewrite this."}], [CAPABLE, EFFICIENT])
    )
    assert (classifier.calls, stage.calls) == (1, 0)
    assert result.candidate_models == [EFFICIENT]
    assert result.signals["switchyard"]["algorithm"] == "llm_classifier"


async def test_tool_history_goes_to_stage() -> None:
    plugin, classifier, stage = composite()
    result = await plugin.run(routing_context(TOOL_HISTORY, [CAPABLE, EFFICIENT]))
    assert (classifier.calls, stage.calls) == (0, 1)
    assert result.candidate_models == [CAPABLE]
    assert result.signals["switchyard"]["algorithm"] == "stage_router"


async def test_other_pools_and_the_served_judge_call_pass_through() -> None:
    plugin, classifier, stage = composite()
    single = routing_context([{"role": "user", "content": "x"}], ["provider/judge"])
    three = routing_context([{"role": "user", "content": "x"}], [CAPABLE, EFFICIENT, "p/c"])
    judged = routing_context([{"role": "user", "content": "x"}], [CAPABLE, EFFICIENT])
    assert await plugin.run(single) is single
    assert await plugin.run(three) is three
    with serving_judge_call():
        assert await plugin.run(judged) is judged
    assert single.candidate_models == ["provider/judge"]
    assert judged.candidate_models == [CAPABLE, EFFICIENT]
    assert (classifier.calls, stage.calls) == (0, 0)


def test_decision_header_is_compact_json_from_the_signals() -> None:
    data = {
        "metadata": {
            "routing_plugin_signals": {
                "switchyard": {
                    "algorithm": "llm_classifier",
                    "selected_model_id": EFFICIENT,
                    "fallback_models": [CAPABLE],
                    "evidence": {"source": "llm-classifier", "score": 0.9, "threshold": 0.5},
                }
            }
        }
    }
    decoded = json.loads(decision_headers(data)[DECISION_HEADER])
    assert decoded == {
        "algorithm": "llm_classifier",
        "selected": EFFICIENT,
        "fallbacks": [CAPABLE],
        "evidence": {"source": "llm-classifier", "score": 0.9, "threshold": 0.5},
    }
    assert decision_headers({"metadata": {}}) == {}
    assert decision_headers({}) == {}
