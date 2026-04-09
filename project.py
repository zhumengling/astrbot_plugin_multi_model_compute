"""项目上下文处理：payload 标准化、任务构建、预算控制、ID 生成。"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from typing import Any, Dict, List, Tuple

from .utils import to_ratio, safe_list, safe_dict


# ---------------------------------------------------------------------------
# 预算
# ---------------------------------------------------------------------------

def task_budget_by_mode(mode: str) -> Dict[str, int]:
    mode = str(mode or "balanced").strip().lower()
    mapping = {
        "fast": {"soft_limit": 1200, "hard_limit": 1700, "provider_limit": 900},
        "balanced": {"soft_limit": 1600, "hard_limit": 2300, "provider_limit": 1400},
        "consensus": {"soft_limit": 1000, "hard_limit": 1400, "provider_limit": 800},
        "creative": {"soft_limit": 1500, "hard_limit": 2100, "provider_limit": 1300},
    }
    return mapping.get(mode, mapping["balanced"])


# ---------------------------------------------------------------------------
# payload 标准化
# ---------------------------------------------------------------------------

def normalize_project_payload(
    *,
    query: str,
    context_summary: str = "",
    project_goal: str = "",
    project_context: str = "",
    constraints: str = "",
    current_stage: str = "",
    current_request: str = "",
    expected_output: str = "",
    project_id: str = "",
    thread_id: str = "",
    topic_id: str = "",
) -> Dict[str, Any]:
    normalized = {
        "query": str(query or "").strip(),
        "context_summary": str(context_summary or "").strip(),
        "project_goal": str(project_goal or "").strip(),
        "project_context": str(project_context or "").strip(),
        "constraints": str(constraints or "").strip(),
        "current_stage": str(current_stage or "").strip(),
        "current_request": str(current_request or "").strip(),
        "expected_output": str(expected_output or "").strip(),
        "project_id": str(project_id or "").strip(),
        "thread_id": str(thread_id or "").strip(),
        "topic_id": str(topic_id or "").strip(),
    }
    if not normalized["query"] and normalized["current_request"]:
        normalized["query"] = normalized["current_request"]
    return normalized


# ---------------------------------------------------------------------------
# 任务构建
# ---------------------------------------------------------------------------

_SECTION_HEADERS = {
    "project_goal": "[Project Goal]",
    "project_context": "[Project Context]",
    "constraints": "[Constraints]",
    "current_stage": "[Current Stage]",
    "current_request": "[Current Request]",
    "expected_output": "[Expected Output]",
    "query": "[Query]",
    "context_summary": "[Context Summary]",
}

_PER_SECTION_CAP = {
    "project_goal": 420,
    "project_context": 650,
    "constraints": 420,
    "current_stage": 260,
    "current_request": 520,
    "expected_output": 360,
    "query": 520,
    "context_summary": 420,
}


def build_project_task_and_meta(payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    mode = str(mode or "balanced").strip().lower()
    budget = task_budget_by_mode(mode)

    sections: List[Tuple[str, str]] = [
        ("project_goal", str(payload.get("project_goal", "") or "").strip()),
        ("project_context", str(payload.get("project_context", "") or "").strip()),
        ("constraints", str(payload.get("constraints", "") or "").strip()),
        ("current_stage", str(payload.get("current_stage", "") or "").strip()),
        ("current_request", str(payload.get("current_request", "") or "").strip()),
        ("expected_output", str(payload.get("expected_output", "") or "").strip()),
        ("query", str(payload.get("query", "") or "").strip()),
        ("context_summary", str(payload.get("context_summary", "") or "").strip()),
    ]

    dedupe_values: set = set()
    assembled_sections: List[str] = []
    compacted_sections: List[Dict[str, Any]] = []
    total_raw_length = 0
    retained_sections: List[str] = []
    dropped_sections: List[str] = []

    for key, raw in sections:
        if not raw:
            continue
        normalized = re.sub(r"\s+", " ", raw).strip()
        if not normalized:
            continue
        lower_norm = normalized.lower()
        if lower_norm in dedupe_values:
            dropped_sections.append(key)
            compacted_sections.append({"section": key, "strategy": "deduplicated", "before": len(raw), "after": 0})
            continue
        dedupe_values.add(lower_norm)

        total_raw_length += len(normalized)
        cap = _PER_SECTION_CAP.get(key, 480)
        clipped = normalized
        if len(clipped) > cap:
            clipped = clipped[:cap] + " ...[section_clipped]"
            compacted_sections.append({"section": key, "strategy": "section_clip", "before": len(normalized), "after": len(clipped)})
        assembled_sections.append(f"{_SECTION_HEADERS.get(key, f'[{key}]')}\n{clipped}")
        retained_sections.append(key)

    task = "\n\n".join(assembled_sections).strip()
    if not task:
        task = str(payload.get("query", "") or payload.get("current_request", "") or "").strip()

    original_length = len(task)
    hard_limit = budget.get("hard_limit", 2200)
    soft_limit = budget.get("soft_limit", 1600)
    truncated = False
    truncation_reason = "none"

    if len(task) > hard_limit:
        task = task[:hard_limit] + "\n[project_payload_truncated_for_runtime_control]"
        truncated = True
        truncation_reason = "hard_limit"
        compacted_sections.append({"section": "global", "strategy": "hard_truncate", "before": original_length, "after": len(task)})
    elif len(task) > soft_limit:
        task = task[:soft_limit] + "\n[project_payload_soft_truncate_for_latency_control]"
        truncated = True
        truncation_reason = "soft_limit"
        compacted_sections.append({"section": "global", "strategy": "soft_truncate", "before": original_length, "after": len(task)})

    final_length = len(task)
    budget_used = {
        "task_chars": final_length,
        "raw_payload_chars": total_raw_length,
        "soft_limit": soft_limit,
        "hard_limit": hard_limit,
        "provider_limit": budget.get("provider_limit", 0),
    }
    budget_utilization = {
        "soft_limit_ratio": to_ratio(final_length, soft_limit),
        "hard_limit_ratio": to_ratio(final_length, hard_limit),
        "provider_limit_ratio": to_ratio(final_length, budget.get("provider_limit", 0)),
        "status": "over_hard" if final_length > hard_limit else ("over_soft" if final_length > soft_limit else "within_budget"),
    }

    unique_retained = list(dict.fromkeys(retained_sections))
    unique_dropped = list(dict.fromkeys(dropped_sections))
    section_total = len(unique_retained) + len(unique_dropped)
    context_coverage = {
        "retained_count": len(unique_retained),
        "dropped_count": len(unique_dropped),
        "total_count": section_total,
        "retained_ratio": to_ratio(len(unique_retained), max(1, section_total)),
        "truncated": truncated,
    }

    degradation_applied = truncated or bool(unique_dropped) or bool(compacted_sections)
    degradation_reason: List[str] = []
    if unique_dropped:
        degradation_reason.append("section_deduplicated")
    if any(str(c.get("strategy", "")).startswith("section_clip") for c in compacted_sections):
        degradation_reason.append("section_clipped")
    if truncation_reason != "none":
        degradation_reason.append(f"global_{truncation_reason}")

    has_project_semantics = any(
        bool(payload.get(k))
        for k in ("project_goal", "project_context", "constraints", "current_stage", "current_request", "expected_output", "project_id", "thread_id", "topic_id")
    )

    return {
        "task": task,
        "task_stats": {
            "original_length": original_length,
            "final_length": final_length,
            "truncated": truncated,
            "mode": mode,
            "soft_limit": soft_limit,
            "hard_limit": hard_limit,
            "raw_payload_length": total_raw_length,
        },
        "task_budget": budget,
        "budget_used": budget_used,
        "budget_utilization": budget_utilization,
        "truncation_info": {
            "truncated": truncated,
            "truncation_reason": truncation_reason,
            "compacted_sections": compacted_sections,
            "total_compacted_sections": len(compacted_sections),
            "retained_sections": unique_retained,
            "dropped_sections": unique_dropped,
            "context_coverage": context_coverage,
            "degradation_applied": degradation_applied,
            "degradation_reason": degradation_reason,
        },
        "has_project_semantics": has_project_semantics,
        "project_context": {
            "project_goal": payload.get("project_goal", ""),
            "project_context": payload.get("project_context", ""),
            "constraints": payload.get("constraints", ""),
            "current_stage": payload.get("current_stage", ""),
            "current_request": payload.get("current_request", ""),
            "expected_output": payload.get("expected_output", ""),
        },
        "project_identifiers": {
            "project_id": payload.get("project_id", ""),
            "thread_id": payload.get("thread_id", ""),
            "topic_id": payload.get("topic_id", ""),
        },
    }


# ---------------------------------------------------------------------------
# 任务压缩
# ---------------------------------------------------------------------------

def compress_task_for_provider(
    task: str,
    provider_id: str,
    mode: str,
    task_budget: Dict[str, int] | None = None,
    health_hint: Dict[str, Any] | None = None,
) -> Tuple[str, bool, str, int]:
    txt = str(task or "").strip()
    if not txt:
        return txt, False, "empty_task", 0

    pid = str(provider_id or "").lower()

    # 免除针对特定（如推理型 / 魔改 gpt-5.x）模型的简短作答约束，以防它们因为格式不适应而返回 Empty Outpu Error
    is_reasoning_or_strict_model = any(k in pid for k in ["gpt-5", "o1", "o3", "deepseek-r1"])
    
    if not is_reasoning_or_strict_model:
        brevity_instruction = """

[Answer Style]
请尽量简短精炼作答：优先用 3-6 条 bullet；每条尽量一句话；先给结论，再给必要补充；避免冗长解释、客套话和重复表述。"""
        if "[Answer Style]" not in txt:
            txt += brevity_instruction

    mode = str(mode or "balanced").strip().lower()
    budget = task_budget if isinstance(task_budget, dict) else task_budget_by_mode(mode)
    health_hint = health_hint if isinstance(health_hint, dict) else {}

    limit = int(budget.get("provider_limit", 1200) or 1200)
    reason_parts: List[str] = []

    if mode == "consensus":
        limit = min(limit, 820)
        reason_parts.append("consensus_compact")
    elif mode == "fast":
        limit = min(limit, 920)
        reason_parts.append("fast_latency_control")

    if "coder" in pid or "codex" in pid:
        limit = min(limit, 760)
        reason_parts.append("coder_short_prompt")

    if str(health_hint.get("timeout_tendency", "")).lower() == "high":
        limit = min(limit, 650)
        reason_parts.append("provider_timeout_tendency_high")
    elif int(health_hint.get("consecutive_failures", 0) or 0) >= 2:
        limit = min(limit, 700)
        reason_parts.append("provider_recent_failures")

    if len(txt) <= limit:
        return txt, False, "not_needed", limit

    compact = txt[:limit] + "\n[compressed_for_provider_timeout_control]"
    reason_parts.append(f"len>{limit}")
    return compact, True, ";".join(reason_parts), limit


def compact_task_for_consensus(task: str) -> str:
    txt = str(task or "").strip()
    if len(txt) <= 800:
        return txt
    return txt[:800] + "\n[condensed_for_consensus]"


def build_retry_task(task: str, mode: str = "fast") -> str:
    txt = str(task or "").strip()
    if not txt:
        return txt
    retry_limit = 500 if mode == "fast" else 700
    if len(txt) <= retry_limit:
        return txt
    return txt[:retry_limit] + "\n[retry_compacted]"


# ---------------------------------------------------------------------------
# ID 生成
# ---------------------------------------------------------------------------

def build_ids(request_task: str, project_identifiers: Dict[str, Any]) -> Dict[str, str]:
    seed = "|".join([
        str(request_task or "")[:200],
        str((project_identifiers or {}).get("project_id", "") or ""),
        str((project_identifiers or {}).get("thread_id", "") or ""),
        str((project_identifiers or {}).get("topic_id", "") or ""),
        str(time.time()),
    ])
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:12]
    run_id = f"mmc-{digest}"
    trace_id = f"trace-{uuid.uuid4().hex[:16]}"
    return {"run_id": run_id, "trace_id": trace_id}
