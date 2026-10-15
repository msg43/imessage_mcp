"""Explicit search degradation, shared by the authenticated and local surfaces."""

from __future__ import annotations

from typing import Any

from imsg.retrieval.background_warm_up import WarmUpPhase, WarmUpStatus


def needs_models(name: str, arguments: dict[str, Any], status: WarmUpStatus | None) -> bool:
    del status
    return name == "search_messages" and arguments.get("mode", "hybrid") != "keyword"


def search_arguments(arguments: dict[str, Any], status: WarmUpStatus | None) -> dict[str, Any]:
    del status
    return {"mode": "hybrid", **arguments}


def annotate_search(
    payload: dict[str, Any], arguments: dict[str, Any], status: WarmUpStatus | None
) -> dict[str, Any]:
    mode = arguments.get("mode", "hybrid")
    state = status.phase.value if status is not None else "unknown"
    keyword = mode == "keyword"
    offline = keyword and status is not None and status.phase is not WarmUpPhase.READY
    notice = None
    if keyword:
        notice = (
            "MODELS OFFLINE — KEYWORD-ONLY RESULTS. "
            if offline
            else "KEYWORD-ONLY RESULTS — MODELS NOT USED. "
        ) + (
            "Semantic search, visual similarity search, and model reranking were NOT used. "
            "Relevant messages with different wording may be missing. No matches does NOT mean "
            "the information is absent. Show this notice prominently with the answer. "
            "Use mode='hybrid' to request full search and wait for models."
        )
    return {
        "search_mode": mode,
        "model_status": state,
        "degraded": keyword,
        "warning_code": "MODELS_OFFLINE" if offline else ("KEYWORD_ONLY" if keyword else None),
        "warning": notice,
        **payload,
    }
