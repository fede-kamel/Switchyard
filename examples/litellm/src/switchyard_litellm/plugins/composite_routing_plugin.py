# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose between Stage and the capability classifier by what the request carries.

A fresh task has nothing but the user's words, so the classifier's judge
estimates whether the efficient tier can do it. A continuing agent turn
carries tool calls and tool results, and those are a better signal than a
judge: failing tests and errors escalate, routine progress stays efficient.
One plugin object serves both, so a deployment configures one algorithm and
every client of the proxy gets the same decision.

The plugin acts on a request with exactly two unique candidates, the capable
model declared first and the efficient model second, which is the contract
Stage and the classifier share. Any other candidate pool (the judge's own
group, a single fixed model) passes through unchanged, as does the judge
call the classifier issues while it is served.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import RoutingContext

from .classifier_routing_plugin import ClassifierRoutingPlugin
from .judge import is_judge_call
from .lite_llm_request_rewriter import LiteLLMRequestRewriter
from .stage_routing_plugin import StageRoutingPlugin

DECISION_HEADER = "x-switchyard-decision"

log = logging.getLogger("switchyard_litellm.decisions")
# The proxy leaves foreign loggers at the root's WARNING; decisions are INFO and must be seen.
log.setLevel(logging.INFO)


def has_tool_history(messages: Sequence[Mapping[str, object]]) -> bool:
    """Whether the conversation already contains tool calls or tool results."""
    return any(message.get("role") == "tool" or message.get("tool_calls") for message in messages)


class CompositeRoutingPlugin(LiteLLMRequestRewriter):
    """Stage routing on tool history, classifier routing otherwise."""

    def __init__(
        self,
        *,
        classifier: ClassifierRoutingPlugin,
        stage: StageRoutingPlugin,
    ) -> None:
        super().__init__()
        self._classifier = classifier
        self._stage = stage

    @staticmethod
    def routes(context: RoutingContext) -> bool:
        """Whether this request is one the composite decides.

        Exactly two unique candidates, and not the judge call the classifier is serving
        (flagged in-process, not through request metadata a client could set).
        """
        return len(dict.fromkeys(context.candidate_models)) == 2 and not is_judge_call()

    async def run(self, context: RoutingContext) -> RoutingContext:
        """Pick the algorithm by the request's shape; the signal records which one decided."""
        if not self.routes(context):
            return context
        plugin: ClassifierRoutingPlugin | StageRoutingPlugin
        if has_tool_history(cast(Sequence[Mapping[str, object]], context.structured_messages)):
            plugin = self._stage
        else:
            plugin = self._classifier
        result = await plugin.run(context)
        decision = result.signals.get("switchyard")
        if isinstance(decision, Mapping):
            # One line per decision in the proxy's log, keyed by the caller's key alias: the
            # record a deployment reconstructs its routing from.
            log.info(
                "switchyard decision %s",
                json.dumps(
                    {
                        "algorithm": decision.get("algorithm"),
                        "selected": decision.get("selected_model_id"),
                        "fallbacks": decision.get("fallback_models"),
                        "evidence": decision.get("evidence"),
                        "key_alias": context.metadata.get("user_api_key_alias"),
                        "messages": len(context.structured_messages),
                    },
                    separators=(",", ":"),
                    default=str,
                ),
            )
        return result

    async def async_post_call_response_headers_hook(
        self,
        data: dict[Any, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> dict[str, str] | None:
        """Return the routing decision to the caller as one response header."""
        del user_api_key_dict, response, request_headers, litellm_call_info
        return decision_headers(data)


def decision_headers(data: Mapping[str, Any]) -> dict[str, str]:
    """``x-switchyard-decision`` from the signals LiteLLM stashed in the request metadata."""
    for key in ("metadata", "litellm_metadata"):
        metadata = data.get(key)
        if not isinstance(metadata, Mapping):
            continue
        signals = metadata.get("routing_plugin_signals")
        decision = signals.get("switchyard") if isinstance(signals, Mapping) else None
        if isinstance(decision, Mapping):
            compact = {
                "algorithm": decision.get("algorithm"),
                "selected": decision.get("selected_model_id"),
                "fallbacks": decision.get("fallback_models"),
                "evidence": decision.get("evidence"),
            }
            return {DECISION_HEADER: json.dumps(compact, separators=(",", ":"))}
    return {}


__all__ = ["DECISION_HEADER", "CompositeRoutingPlugin", "decision_headers", "has_tool_history"]
