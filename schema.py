"""Schema 默认值、校验与状态映射。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .utils import safe_dict, safe_list, is_timeout_error, classify_failure_reason

RESULT_SCHEMA_VERSION = "tool-first-v5-project-synthesis-stability"


# ---------------------------------------------------------------------------
# 默认值工厂
# ---------------------------------------------------------------------------

def default_material_quality() -> Dict[str, Any]:
    return {
        "level": "unknown",
        "overall": None,
        "coverage": None,
        "consistency": None,
        "evidence_grounding": None,
        "confidence_note": "quality dimensions unavailable",
        "caution_flags": [],
        "ready_for_final_summary": False,
    }


def default_synthesis() -> Dict[str, Any]:
    return {
        "core": {
            "material_ready": False,
            "source_type": "none",
            "participants_total": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_ratio": 0,
            "common_points": [],
            "differences": [],
            "notable_insights": [],
            "conflict_points": [],
            "conflict_level": "low",
            "draft_synthesis": "",
            "recommended_focus": [],
            "material_quality": default_material_quality(),
            "all_failed": False,
            "partial_success": False,
            "retryable": False,
            "source_refs": [],
            "evidence_refs": [],
            "consumer_checklist": [],
        },
        "diagnostics": {
            "overlap_groups": [],
            "duplicate_candidates": [],
            "source_attribution": {},
            "provenance": {},
            "raw_materials": {},
            "synthesized_material": {},
        },
    }


# ---------------------------------------------------------------------------
# 状态映射
# ---------------------------------------------------------------------------

def map_level_to_overall(level: str) -> Optional[float]:
    mapping = {"high": 0.85, "medium": 0.65, "low": 0.35, "unknown": None}
    return mapping.get(str(level or "unknown").lower(), None)


def compat_status(status: str) -> str:
    if status in {"partial_success", "fallback_used", "no_material"}:
        return "degraded"
    return status


def status_v2(status: str) -> str:
    normalized = str(status or "").lower()
    if normalized == "ok":
        return "ok"
    if normalized in {"partial", "partial_success", "fallback_used", "no_material"}:
        return "partial"
    if normalized in {"failed", "all_failed", "invalid_request"}:
        return "failed"
    return "partial" if normalized else "failed"


def derive_execution_status(
    *,
    material_ready: bool,
    all_failed: bool,
    partial_success: bool,
    fallback_applied: bool,
) -> str:
    if all_failed:
        return "all_failed"
    if not material_ready:
        return "no_material"
    if partial_success:
        return "partial_success"
    if fallback_applied:
        return "fallback_used"
    return "ok"


def fallback_applied(result: Dict[str, Any]) -> bool:
    fallback_from = str(result.get("fallback_from", "") or "").strip()
    fallback_reason = str(result.get("fallback_reason", "") or "").strip()
    fallback = safe_dict(result, "fallback")
    strategy = str(fallback.get("strategy", "") or "").strip()
    fb_reason = str(fallback.get("fallback_reason", "") or "").strip()

    no_meaningful = {"", "none", "no_fallback", "not_applied"}
    reason_effective = (fallback_reason.lower() not in no_meaningful) or (fb_reason.lower() not in no_meaningful)
    strategy_effective = strategy.lower() not in {"", "none", "not_applied", "partial_material_preferred"}
    return bool(fallback_from or reason_effective or strategy_effective)


# ---------------------------------------------------------------------------
# execution_control / consumer_hints 构建
# ---------------------------------------------------------------------------

def build_execution_control(result: Dict[str, Any]) -> Dict[str, Any]:
    timing = safe_dict(result, "timing")
    retry_summary = safe_dict(result, "retry_summary")
    fb = safe_dict(result, "fallback")
    truncation_info = safe_dict(result, "truncation_info")
    provider_health = safe_dict(result, "provider_health")

    timeout_tendencies = []
    timeout_failures = 0
    for pid, info in provider_health.items():
        if not isinstance(info, dict):
            continue
        tendency = info.get("timeout_tendency")
        if tendency:
            timeout_tendencies.append({"provider_id": pid, "tendency": tendency})
        timeout_failures += int(info.get("timeouts", 0) or 0)

    degradation_reasons = []
    for reason in safe_list(result, "degradation_reason"):
        if reason not in degradation_reasons:
            degradation_reasons.append(reason)
    fb_reason = fb.get("fallback_reason")
    if fb_reason and fb_reason not in {"none", "not_applied"} and fb_reason not in degradation_reasons:
        degradation_reasons.append(fb_reason)
    tr_reason = truncation_info.get("truncation_reason")
    if tr_reason and tr_reason not in {"none"} and tr_reason not in degradation_reasons:
        degradation_reasons.append(tr_reason)

    return {
        "timeout": {
            "configured_sec": timing.get("timeout_sec"),
            "total_elapsed_ms": timing.get("total_elapsed_ms"),
            "avg_elapsed_ms": timing.get("avg_elapsed_ms"),
            "timeout_failures": timeout_failures,
            "provider_tendencies": timeout_tendencies,
        },
        "retry": {
            "attempted": retry_summary.get("attempted", 0),
            "succeeded": retry_summary.get("succeeded", 0),
            "failed": retry_summary.get("failed", 0),
            "retryable": bool(result.get("retryable", False)),
        },
        "fallback": {
            "applied": bool(result.get("fallback_from") or (fb.get("fallback_reason") not in {None, "", "none", "not_applied"})),
            "strategy": fb.get("strategy"),
            "reason": fb.get("fallback_reason"),
            "from_backend": result.get("fallback_from"),
        },
        "truncation": {
            "applied": bool(truncation_info.get("truncated") or result.get("degradation_applied")),
            "reason": truncation_info.get("truncation_reason"),
            "compacted_sections": truncation_info.get("compacted_sections", []),
            "retained_sections": truncation_info.get("retained_sections", []),
            "dropped_sections": truncation_info.get("dropped_sections", []),
            "context_coverage": truncation_info.get("context_coverage", {}),
        },
        "degradation": {
            "applied": bool(result.get("degradation_applied", False)),
            "reasons": degradation_reasons,
        },
    }


def build_consumer_hints(result: Dict[str, Any]) -> Dict[str, Any]:
    quality = safe_dict(result, "material_quality")
    sv2 = result.get("status_v2") or status_v2(result.get("status"))
    overall = quality.get("overall")
    evidence_grounding = quality.get("evidence_grounding")

    do_not_overclaim = []
    if result.get("conflict_level") in {"medium", "high"}:
        do_not_overclaim.append("high_conflict_between_models")
    if result.get("degradation_applied"):
        do_not_overclaim.append("degraded_or_truncated_material")
    if isinstance(evidence_grounding, (int, float)) and evidence_grounding < 0.6:
        do_not_overclaim.append("low_evidence_grounding")

    ask_user = []
    if isinstance(overall, (int, float)) and overall < 0.6:
        ask_user.append("quality.overall < 0.6")
    if sv2 == "partial":
        ask_user.append("status_v2 == partial and user needs high-confidence conclusion")

    recommended_mode = "balanced"
    if sv2 == "failed":
        recommended_mode = "fast"
    elif result.get("conflict_level") == "high":
        recommended_mode = "consensus"

    return {
        "status_v2": sv2,
        "recommended_mode": recommended_mode,
        "do_not_overclaim": do_not_overclaim,
        "ask_user_clarification_if": ask_user,
        "suggested_next_action": result.get("next_action_hint") or "Review consensus and differences before finalizing.",
    }


# ---------------------------------------------------------------------------
# Schema 不变量保证
# ---------------------------------------------------------------------------

def ensure_schema_invariants(result: Dict[str, Any]) -> Dict[str, Any]:
    syn_default = default_synthesis()
    synthesis = safe_dict(result, "synthesis")
    core = safe_dict(synthesis, "core")
    diagnostics = safe_dict(synthesis, "diagnostics")

    normalized_core = {**syn_default["core"], **core}
    mq = safe_dict(normalized_core, "material_quality")
    normalized_core["material_quality"] = {**default_material_quality(), **mq}
    normalized_diag = {**syn_default["diagnostics"], **diagnostics}

    synthesis["core"] = normalized_core
    synthesis["diagnostics"] = normalized_diag
    result["synthesis"] = synthesis

    result["status"] = str(result.get("status", "ok") or "ok")
    result["status_compat"] = compat_status(result["status"])
    result["status_v2"] = status_v2(result["status"])

    mq_top = safe_dict(result, "material_quality")
    result["material_quality"] = {**default_material_quality(), **normalized_core["material_quality"], **mq_top}
    if result["material_quality"].get("overall") is None:
        result["material_quality"]["overall"] = map_level_to_overall(result["material_quality"].get("level"))
    if result["material_quality"].get("coverage") is None:
        sr = normalized_core.get("success_ratio", result.get("success_ratio"))
        result["material_quality"]["coverage"] = round(float(sr), 3) if isinstance(sr, (int, float)) else result["material_quality"].get("overall")
    if result["material_quality"].get("consistency") is None:
        cl = str(result.get("conflict_level", normalized_core.get("conflict_level", "low")) or "low").lower()
        result["material_quality"]["consistency"] = {"high": 0.35, "medium": 0.6, "low": 0.85}.get(cl, result["material_quality"].get("overall"))
    if result["material_quality"].get("evidence_grounding") is None:
        src = safe_list(result, "source_refs") or safe_list(normalized_core, "source_refs")
        evr = safe_list(result, "evidence_refs") or safe_list(normalized_core, "evidence_refs")
        result["material_quality"]["evidence_grounding"] = 0.8 if (src or evr) else 0.5
    if not result["material_quality"].get("confidence_note") or result["material_quality"].get("confidence_note") == "quality dimensions unavailable":
        result["material_quality"]["confidence_note"] = f"status_v2={result['status_v2']}, conflict_level={result.get('conflict_level', normalized_core.get('conflict_level', 'low'))}"

    result["provider_health"] = safe_dict(result, "provider_health")
    result["task_budget"] = safe_dict(result, "task_budget")
    result["truncation_info"] = safe_dict(result, "truncation_info")
    result["compacted_sections"] = safe_list(result, "compacted_sections")
    result["fallback"] = safe_dict(result, "fallback")
    result["retry_guidance"] = safe_dict(result, "retry_guidance")
    result["next_action_hint"] = str(result.get("next_action_hint", "") or "")
    result["recommended_default_model_action"] = str(result.get("recommended_default_model_action", "") or "")
    result["recommended_next_steps"] = safe_list(result, "recommended_next_steps")

    result["budget_used"] = safe_dict(result, "budget_used")
    result["budget_utilization"] = safe_dict(result, "budget_utilization")
    result["truncation_reason"] = str(result.get("truncation_reason", "none") or "none")
    result["context_coverage"] = safe_dict(result, "context_coverage")
    result["retained_sections"] = safe_list(result, "retained_sections")
    result["dropped_sections"] = safe_list(result, "dropped_sections")
    result["degradation_applied"] = bool(result.get("degradation_applied", False))
    result["degradation_reason"] = safe_list(result, "degradation_reason")
    result["execution_control"] = build_execution_control(result)
    result["consumer_hints"] = build_consumer_hints(result)

    return result
