"""Provider 通信：调用、聚合、健康追踪、mock 计算、后端调度。"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List

from astrbot.api import logger

from .utils import normalize_error, classify_failure_reason, is_retryable_failure_type, is_timeout_error, now_iso_utc
from .project import task_budget_by_mode, compress_task_for_provider, compact_task_for_consensus, build_retry_task


_RUNTIME_LOG_PATH = Path(__file__).resolve().parent / "runtime.log"


def _append_runtime_log(line: str) -> None:
    try:
        _RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _RUNTIME_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{line}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Response 解析
# ---------------------------------------------------------------------------

def extract_text_from_response(llm_resp: Any) -> str:
    """稳健解析 provider.text_chat 返回，兼容不同 provider 的返回结构差异。"""

    def _clean(v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    def _from_dict(d: Dict[str, Any]) -> str:
        for key in ("completion_text", "text", "content", "answer", "message"):
            if key in d:
                value = d.get(key)
                if isinstance(value, dict):
                    nested = _from_dict(value)
                    if nested:
                        return nested
                elif isinstance(value, list):
                    parts = [_clean(x) for x in value if _clean(x)]
                    if parts:
                        return "\n".join(parts)
                else:
                    cleaned = _clean(value)
                    if cleaned:
                        return cleaned

        choices = d.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict):
                                t = _clean(item.get("text"))
                                if t:
                                    parts.append(t)
                            else:
                                t = _clean(item)
                                if t:
                                    parts.append(t)
                        if parts:
                            return "\n".join(parts)
                    text = _clean(content)
                    if text:
                        return text
                text = _clean(first.get("text"))
                if text:
                    return text
        return ""

    if llm_resp is None:
        return ""
    if isinstance(llm_resp, str):
        return llm_resp.strip()
    if isinstance(llm_resp, dict):
        return _from_dict(llm_resp)

    # AstrBot 标准 LLMResponse
    completion_text = getattr(llm_resp, "completion_text", None)
    cleaned_completion = _clean(completion_text)
    if cleaned_completion:
        return cleaned_completion

    result_chain = getattr(llm_resp, "result_chain", None)
    get_plain_text = getattr(result_chain, "get_plain_text", None)
    if callable(get_plain_text):
        try:
            chain_text = _clean(get_plain_text())
            if chain_text:
                return chain_text
        except Exception:
            pass

    for attr in ("text", "content", "answer", "message", "_completion_text"):
        v = getattr(llm_resp, attr, None)
        if isinstance(v, dict):
            nested = _from_dict(v)
            if nested:
                return nested
        else:
            cleaned = _clean(v)
            if cleaned:
                return cleaned

    raw_completion = getattr(llm_resp, "raw_completion", None)
    if raw_completion is not None:
        try:
            if hasattr(raw_completion, "model_dump"):
                raw_dict = raw_completion.model_dump()
            elif isinstance(raw_completion, dict):
                raw_dict = raw_completion
            else:
                raw_dict = None
            if isinstance(raw_dict, dict):
                raw_text = _from_dict(raw_dict)
                if raw_text:
                    return raw_text
        except Exception:
            pass

    if hasattr(llm_resp, "model_dump"):
        try:
            dumped = llm_resp.model_dump()
            if isinstance(dumped, dict):
                dumped_text = _from_dict(dumped)
                if dumped_text:
                    return dumped_text
        except Exception:
            pass

    return ""


# ---------------------------------------------------------------------------
# Provider Health 追踪
# ---------------------------------------------------------------------------

def estimate_timeout_tendency(elapsed_ms: int, timeout_sec: float) -> str:
    timeout_ms = max(1, int(float(timeout_sec or 0) * 1000))
    ratio = float(elapsed_ms or 0) / timeout_ms
    if ratio >= 0.9:
        return "high"
    if ratio >= 0.65:
        return "medium"
    return "low"


def update_provider_health(
    health_dict: Dict[str, Dict[str, Any]],
    provider_id: str,
    row: Dict[str, Any],
    timeout_sec: float,
):
    pid = str(provider_id or "")
    if not pid:
        return
    elapsed_ms = int((row or {}).get("elapsed_ms", 0) or 0)
    ok = bool((row or {}).get("ok"))
    error = str((row or {}).get("error", "") or "")
    failure_type = classify_failure_reason(error)

    health = health_dict.setdefault(pid, {
        "calls": 0, "success": 0, "failures": 0, "timeouts": 0,
        "consecutive_failures": 0, "avg_elapsed_ms": 0,
        "last_error": "", "last_failure_type": "",
        "timeout_tendency": "low", "last_updated": now_iso_utc(),
        # P2-1 Circuit Breaker 支持字段
        "last_failure_ts": 0.0,
        # P2-5 ReasoningBank 支持字段
        "winning_count": 0,
        # P2-4 滑动窗口：最近20次 elapsed_ms
        "recent_elapsed_window": [],
    })

    health["calls"] += 1
    prev_avg = int(health.get("avg_elapsed_ms", 0) or 0)
    health["avg_elapsed_ms"] = int((prev_avg * (health["calls"] - 1) + elapsed_ms) / max(1, health["calls"]))
    health["last_updated"] = now_iso_utc()

    # 滑动窗口更新（P2-4）
    window: list = health.setdefault("recent_elapsed_window", [])
    window.append(elapsed_ms)
    if len(window) > 20:
        window.pop(0)
    health["recent_elapsed_window"] = window

    if ok:
        health["success"] += 1
        health["consecutive_failures"] = 0
        health["last_error"] = ""
        health["last_failure_type"] = ""
    else:
        health["failures"] += 1
        health["consecutive_failures"] += 1
        health["last_error"] = error[:240]
        health["last_failure_type"] = failure_type
        health["last_failure_ts"] = time.time()  # P2-1 熔断时间戳
        if is_timeout_error(error):
            health["timeouts"] += 1

    tendency = estimate_timeout_tendency(health.get("avg_elapsed_ms", 0), timeout_sec)
    if health.get("timeouts", 0) >= 2:
        tendency = "high"
    elif health.get("timeouts", 0) >= 1 and tendency == "low":
        tendency = "medium"
    health["timeout_tendency"] = tendency


def provider_health_hint(health_dict: Dict[str, Dict[str, Any]], provider_id: str) -> Dict[str, Any]:
    raw = health_dict.get(str(provider_id or ""), {})
    if not isinstance(raw, dict) or not raw:
        return {"known": False, "recent_failure_hint": "none", "timeout_tendency": "unknown", "consecutive_failures": 0}
    return {
        "known": True,
        "calls": int(raw.get("calls", 0) or 0),
        "success": int(raw.get("success", 0) or 0),
        "failures": int(raw.get("failures", 0) or 0),
        "timeouts": int(raw.get("timeouts", 0) or 0),
        "consecutive_failures": int(raw.get("consecutive_failures", 0) or 0),
        "timeout_tendency": str(raw.get("timeout_tendency", "unknown") or "unknown"),
        "recent_failure_hint": str(raw.get("last_failure_type", "none") or "none"),
        "last_error": str(raw.get("last_error", "") or "")[:180],
        "avg_elapsed_ms": int(raw.get("avg_elapsed_ms", 0) or 0),
        "last_updated": str(raw.get("last_updated", "") or ""),
    }


# ---------------------------------------------------------------------------
# P2-1 Circuit Breaker
# ---------------------------------------------------------------------------

_CB_THRESHOLD = 3     # 连续失败 N 次触发熔断
_CB_COOLDOWN  = 300.0 # 冷却秒数（5分钟）


def is_circuit_open(health_dict: Dict[str, Dict[str, Any]], provider_id: str) -> bool:
    """检查 provider 是否处于熔断状态（P2-1）。"""
    h = health_dict.get(str(provider_id or ""), {})
    if not isinstance(h, dict):
        return False
    consecutive = int(h.get("consecutive_failures", 0) or 0)
    if consecutive < _CB_THRESHOLD:
        return False
    last_fail_ts = float(h.get("last_failure_ts", 0) or 0)
    return (time.time() - last_fail_ts) < _CB_COOLDOWN


# ---------------------------------------------------------------------------
# P2-4 动态超时
# ---------------------------------------------------------------------------

def dynamic_timeout(health_dict: Dict[str, Dict[str, Any]], provider_id: str, base: float = 200.0) -> float:
    """根据历史均值动态计算单模型超时（P2-4），最低 15s，最高 base。"""
    h = health_dict.get(str(provider_id or ""), {})
    if not isinstance(h, dict):
        return base
    # 优先用滑动窗口均值，其次用总体均值
    window = h.get("recent_elapsed_window", [])
    if window:
        avg_ms = sum(window) / len(window)
    else:
        avg_ms = float(h.get("avg_elapsed_ms", 0) or 0)
    if avg_ms <= 0:
        return base
    computed = avg_ms / 1000.0 * 3.5
    return max(15.0, min(base, computed))


def effective_max_models(requested: int, mode: str, task_len: int, default_count: int) -> Dict[str, Any]:
    try:
        req = int(requested)
    except Exception:
        req = 0
    if req <= 0:
        req = default_count

    mode = str(mode or "balanced").strip().lower()
    cap = req
    reasons: List[str] = []

    # 仅对超长任务做轻量保护，不再对 balanced/consensus 一刀切压到 2
    if task_len >= 3200 and cap > 3 and mode not in {"debate", "creative"}:
        cap = 3
        reasons.append("very_long_task_cap")
    elif task_len >= 2200 and cap > 4 and mode not in {"fast", "debate", "creative"}:
        cap = 4
        reasons.append("long_task_soft_cap")

    cap = max(1, cap)
    return {"requested": req, "effective": cap, "adjusted": cap != req, "reasons": reasons}


async def _llm_synthesis_layer(
    task: str,
    normalized_results: List[Dict[str, Any]],
    call_fn: Callable[[str, str], Coroutine[Any, Any, Dict[str, Any]]],
    provider_health_dict: Dict[str, Dict[str, Any]],
    timeout_sec: float = 60.0
) -> str:
    """Round 2 LLM Synthesis Layer for regular multi-model calls (P1-3)."""
    successful = [r for r in normalized_results if r.get("ok") and r.get("text")]
    if len(successful) < 2:
        return ""

    candidates = [r["provider_id"] for r in successful]
    best_pid = min(
        candidates, 
        key=lambda pid: provider_health_dict.get(pid, {}).get("avg_elapsed_ms", 999999)
    )

    sections = "\n\n".join(
        f"[模型 {r['provider_id']}]:\n{r['text']}" for r in successful
    )
    prompt = (
        f"以下是多个AI模型对同一任务的独立回答：\n\n{sections}\n\n"
        f"原始任务：{task}\n\n"
        f"请综合以上回答，提取彼此的共识、补充有效的细节，并输出一个最完整、高质量且直接可用的最终答案。\n"
        f"要求：\n"
        f"1. 直接给出最终答案内容，不要出现类似'综合各个观点'等汇报式套话\n"
        f"2. 保持逻辑结构清晰，如果存在冲突，取最合理的结论并说明理由\n"
    )

    try:
        resp = await asyncio.wait_for(call_fn(best_pid, prompt), timeout=timeout_sec)
        if resp.get("ok") and resp.get("text"):
            return resp["text"]
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# 单次调用
# ---------------------------------------------------------------------------

async def probe_real_call(context, provider_id: str, task: str, timeout_sec: float) -> Dict[str, Any]:
    """真实调用：通过 provider_id -> context.get_provider_by_id -> provider.text_chat。"""
    started = time.perf_counter()
    prompt = task.strip() if task and task.strip() else "请回复:mmprobe ok"
    probe_session_id = f"mmprobe-{uuid.uuid4().hex[:12]}"

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        provider = context.get_provider_by_id(provider_id)
        if not provider:
            raise RuntimeError(f"provider 不存在: {provider_id}")

        text_chat = getattr(provider, "text_chat", None)
        if not callable(text_chat):
            raise RuntimeError(f"provider 不支持 text_chat: {provider_id}, type={type(provider)}")

        async def _invoke_text_chat(**kwargs):
            call_result = text_chat(**kwargs)
            if inspect.isawaitable(call_result):
                if timeout_sec > 0:
                    return await asyncio.wait_for(call_result, timeout=timeout_sec)
                return await call_result
            return call_result

        llm_resp = None
        try:
            llm_resp = await _invoke_text_chat(prompt=prompt, session_id=probe_session_id, contexts=[])
        except TypeError:
            llm_resp = await _invoke_text_chat(prompt=prompt, session_id=probe_session_id)
        except asyncio.TimeoutError:
            raise TimeoutError(f"text_chat timeout after {timeout_sec}s")

        completion_text = extract_text_from_response(llm_resp)
        if not completion_text:
            completion_text = "[empty completion_text]"

        used_provider_id = provider_id
        try:
            meta = provider.meta()
            used_provider_id = getattr(meta, "id", provider_id) or provider_id
        except Exception:
            pass

        _ret = {
            "ok": True, "provider_id": used_provider_id, "session_id": probe_session_id,
            "text": completion_text, "error": "", "elapsed_ms": _elapsed_ms(),
        }
        try:
            _append_runtime_log(f"[multi_model_compute] probe_result={{'provider_id': {_ret['provider_id']!r}, 'ok': {_ret['ok']!r}, 'elapsed_ms': {_ret['elapsed_ms']!r}, 'text_preview': {str(_ret['text'])[:500]!r}, 'error': ''}}")
        except Exception:
            pass
        return _ret
    except Exception as e:
        _ret = {
            "ok": False, "provider_id": provider_id, "session_id": probe_session_id,
            "text": "", "error": normalize_error(e), "elapsed_ms": _elapsed_ms(),
        }
        try:
            _append_runtime_log(f"[multi_model_compute] probe_result={{'provider_id': {_ret['provider_id']!r}, 'ok': {_ret['ok']!r}, 'elapsed_ms': {_ret['elapsed_ms']!r}, 'text_preview': '', 'error': {str(_ret['error'])[:300]!r}}}")
        except Exception:
            pass
        return _ret


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_real_answers(per_model_results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    success_items = []
    for idx, item in enumerate(per_model_results):
        text = str(item.get("text", "") or "").strip()
        if item.get("ok") and text and text != "[empty completion_text]":
            success_items.append((idx, item, text))

    if not success_items:
        return {
            "selected_index": -1, "selected_provider_id": "", "selected_text": "",
            "strategy": "no_success", "reason": "no successful real response",
            "confidence": 0.0, "consensus_size": 0,
        }

    first_idx, first_item, first_text = success_items[0]
    selected_idx, selected_item, selected_text = first_idx, first_item, first_text
    strategy = "first_success"
    reason = "first successful response"
    consensus_size = 1

    if mode == "fast":
        strategy = "first_success_fast"
        reason = "fast mode prefers first successful response"
    elif mode == "consensus" and len(success_items) >= 2:
        buckets: Dict[str, List] = {}
        for tup in success_items:
            signature = re.sub(r"\s+", " ", tup[2]).strip().lower()[:80]
            buckets.setdefault(signature, []).append(tup)
        best_group = max(buckets.values(), key=lambda grp: len(grp))
        if len(best_group) >= 2:
            selected_idx, selected_item, selected_text = sorted(best_group, key=lambda x: (x[1].get("elapsed_ms", 0), x[0]))[0]
            strategy = "consensus_prefix"
            consensus_size = len(best_group)
            reason = f"{consensus_size} responses share similar prefix"
        else:
            selected_idx, selected_item, selected_text = max(success_items, key=lambda x: len(x[2]))
            strategy = "longest_success"
            reason = "no clear consensus, choose longest successful response"
    else:
        selected_idx, selected_item, selected_text = max(success_items, key=lambda x: (len(x[2]), -x[0]))
        strategy = "longest_success"
        reason = "prefer richer successful response"

    success_count = len(success_items)
    ratio = success_count / max(1, len(per_model_results))
    base_conf = 0.58 + 0.28 * ratio
    if strategy == "consensus_prefix":
        base_conf += 0.05 * min(consensus_size - 1, 2)
    elif strategy == "first_success_fast":
        base_conf -= 0.03
    confidence = round(max(0.35, min(0.95, base_conf)), 2)

    return {
        "selected_index": int(selected_idx),
        "selected_provider_id": str(selected_item.get("provider_id", "") or ""),
        "selected_text": selected_text,
        "strategy": strategy, "reason": reason,
        "confidence": confidence, "consensus_size": consensus_size,
    }


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

def mock_compute(
    *,
    task: str,
    context_summary: str,
    mode: str,
    max_models: int,
    models: List[Dict[str, Any]],
    default_mode: str,
    default_count: int,
    return_merged: bool,
    return_candidates_flag: bool,
) -> Dict[str, Any]:
    from .models import choose_models

    mode = (mode or default_mode).strip().lower()
    if max_models <= 0:
        max_models = default_count

    selected = choose_models(models, task, mode, max_models)
    candidates = []
    for index, model in enumerate(selected, start=1):
        candidates.append({
            "rank": index,
            "model_id": model["id"],
            "provider": model["provider"],
            "summary": f"[{model['id']}] 基于 mode={mode} 对任务\u201c{task}\u201d给出的 mock 候选结果。",
            "strengths": model.get("strengths", []),
            "confidence": round(max(0.5, 0.78 - (index - 1) * 0.06), 2),
        })

    mode_labels = {
        "fast": "快速模式已优先选择响应更快的模型进行 mock 计算",
        "consensus": "共识模式已汇总多个模型候选，倾向采用它们的重合结论",
        "creative": "创意模式已尽量保留多样性输出",
    }
    label = mode_labels.get(mode, "平衡模式已综合考虑任务类型、模型能力与默认参与数")
    merged_answer = f"{label}：{', '.join([m['id'] for m in selected])}。"

    result: Dict[str, Any] = {
        "used": True, "mode": mode, "query": task,
        "context_summary": context_summary,
        "participants": [m["id"] for m in selected],
        "note": "v0.11 多模型材料汇总（mock）：用于给默认模型提供可归纳的结构化材料。",
        "backend": "mock",
    }
    if return_merged:
        result["merged_answer"] = merged_answer
    if return_candidates_flag:
        result["candidates"] = candidates
    return result


# ---------------------------------------------------------------------------
# 多模型调用主流程
# ---------------------------------------------------------------------------

async def real_multi_call(
    *,
    context,
    task: str,
    max_models: int,
    mode: str,
    context_summary: str,
    timeout_sec: float,
    per_model_timeout_sec: float = 200.0,
    default_count: int,
    provider_health_dict: Dict[str, Dict[str, Any]],
    selected_provider_ids: List[str],
    enable_internal_llm_synthesis: bool = True,
) -> Dict[str, Any]:
    # enable_concurrency / max_concurrency 已移除：始终全并行，每模型独立 per_model_timeout_sec 超时
    started = time.perf_counter()
    mode = str(mode or "balanced").strip().lower()
    budget = task_budget_by_mode(mode)
    cap_info = effective_max_models(requested=max_models, mode=mode, task_len=len(task or ""), default_count=default_count)

    effective = cap_info.get("effective", 1)
    participants = selected_provider_ids[:min(effective, len(selected_provider_ids))]
    if not participants:
        return {
            "participants": [], "per_model_results": [],
            "success_count": 0, "failed_count": 0,
            "partial_success": False, "all_failed": True,
            "merged_answer": "", "execution": "serial",
            "aggregation": {"selected_index": -1, "selected_provider_id": "", "selected_text": "", "strategy": "no_participants", "reason": "no selected provider", "confidence": 0.0, "consensus_size": 0},
            "timing": {"total_elapsed_ms": 0, "timeout_sec": timeout_sec},
            "retry_summary": {"enabled": mode in {"consensus", "balanced", "fast"}, "attempted": 0, "succeeded": 0, "failed": 0},
            "fallback": {"strategy": "no_provider_available", "fallback_reason": "no selected providers in model_1~model_5"},
            "provider_health": {},
            "stability": {"max_models": cap_info, "query_length": len(task or ""), "task_budget": budget},
        }

    task_for_mode = compact_task_for_consensus(task) if mode == "consensus" else str(task or "")
    ctx_txt = str(context_summary or "").strip()
    if ctx_txt and "[Context Summary]" not in task_for_mode:
        task_for_mode = f"{task_for_mode}\n\n[Context Summary]\n{ctx_txt}"

    per_model_results: List[Dict[str, Any]] = []
    retry_attempted = 0
    retry_succeeded = 0
    stage_timings: Dict[str, Any] = {
        "overall_started_at": started,
        "participants_selected_count": len(participants),
    }

    pmt = float(per_model_timeout_sec) if per_model_timeout_sec and per_model_timeout_sec > 0 else 200.0

    # P2-1: 过滤熔断的 providers
    cb_filter_started = time.perf_counter()
    circuit_blocked = [pid for pid in participants if is_circuit_open(provider_health_dict, pid)]
    if circuit_blocked:
        logger.info(f"[multi_model_compute] Circuit breaker: skipping {circuit_blocked}")
    participants = [pid for pid in participants if not is_circuit_open(provider_health_dict, pid)]
    stage_timings["circuit_breaker_filter_elapsed_ms"] = int((time.perf_counter() - cb_filter_started) * 1000)
    stage_timings["circuit_breaker_blocked"] = list(circuit_blocked)
    stage_timings["participants_after_circuit_breaker"] = list(participants)
    if not participants:
        # 所有 provider 都被熔断，直接返回失败
        return {
            "participants": [], "per_model_results": [],
            "success_count": 0, "failed_count": 0,
            "partial_success": False, "all_failed": True,
            "merged_answer": "", "execution": "circuit_breaker_all_blocked",
            "aggregation": {"selected_index": -1, "selected_provider_id": "", "selected_text": "",
                            "strategy": "no_participants", "reason": "all providers circuit-breaker blocked",
                            "confidence": 0.0, "consensus_size": 0},
            "timing": {"total_elapsed_ms": 0, "timeout_sec": timeout_sec},
            "retry_summary": {"enabled": False, "attempted": 0, "succeeded": 0, "failed": 0},
            "fallback": {"strategy": "circuit_breaker_all_open",
                         "fallback_reason": f"all providers blocked by circuit breaker: {circuit_blocked}"},
            "provider_health": {},
            "stability": {"max_models": cap_info, "query_length": len(task or ""), "task_budget": budget},
        }

    async def _call_with_retry(pid: str) -> Dict[str, Any]:
        nonlocal retry_attempted, retry_succeeded
        provider_started = time.perf_counter()
        health_before = provider_health_hint(provider_health_dict, pid)
        provider_task, provider_compacted, compact_reason, provider_budget = compress_task_for_provider(
            task_for_mode, pid, mode, task_budget=budget, health_hint=health_before,
        )

        dyn_pmt = dynamic_timeout(provider_health_dict, pid, base=pmt)

        # 每个模型独立 dyn_pmt 秒超时，超时直接记录失败，不影响其他并行模型
        first_started = time.perf_counter()
        first = await probe_real_call(context, provider_id=pid, task=provider_task, timeout_sec=dyn_pmt)
        first["attempt_count"] = 1
        first["first_attempt_elapsed_ms"] = int((time.perf_counter() - first_started) * 1000)
        first["retried"] = False
        first["provider_task_compacted"] = provider_compacted
        first["provider_task_compact_reason"] = compact_reason
        first["provider_task_length"] = len(provider_task or "")
        first["provider_task_budget"] = provider_budget
        first["provider_health_hint_before"] = health_before

        final_row = first
        first_error = str(first.get("error", "") or "")
        first_failure_type = classify_failure_reason(first_error)
        can_retry = bool(not first.get("ok") and is_retryable_failure_type(first_failure_type))
        retry_mode_allowed = mode in {"consensus", "balanced", "fast"}

        if can_retry and retry_mode_allowed:
            retry_attempted += 1
            retry_task = build_retry_task(provider_task, mode="consensus" if mode != "fast" else "fast")
            retry_started = time.perf_counter()
            second = await probe_real_call(context, provider_id=pid, task=retry_task, timeout_sec=dyn_pmt)
            second["retry_elapsed_ms"] = int((time.perf_counter() - retry_started) * 1000)
            second["attempt_count"] = 2
            second["retried"] = True
            second["first_error"] = first_error
            second["first_failure_type"] = first_failure_type
            second["first_elapsed_ms"] = first.get("elapsed_ms", 0)
            second["retry_reason"] = f"retryable_{first_failure_type}"
            second["retry_task_compacted"] = True
            second["provider_task_compacted"] = True
            second["provider_task_compact_reason"] = "retry_compact"
            second["provider_task_length"] = len(retry_task or "")
            second["provider_task_budget"] = provider_budget
            second["provider_health_hint_before"] = health_before
            final_row = second
            if second.get("ok"):
                retry_succeeded += 1

        update_provider_health(provider_health_dict, pid, final_row, pmt)
        final_row["provider_health_hint_after"] = provider_health_hint(provider_health_dict, pid)
        final_row["provider_total_elapsed_ms"] = int((time.perf_counter() - provider_started) * 1000)
        return final_row

    # 始终全并行：所有模型同时发起，各自独立受 per_model_timeout_sec 约束
    gather_started = time.perf_counter()
    gathered = await asyncio.gather(
        *[_call_with_retry(pid) for pid in participants],
        return_exceptions=True,
    )
    stage_timings["gather_total_elapsed_ms"] = int((time.perf_counter() - gather_started) * 1000)
    for idx, item in enumerate(gathered):
        if isinstance(item, Exception):
            pid = participants[idx]
            row = {
                "ok": False, "provider_id": pid, "session_id": "", "text": "",
                "error": normalize_error(item), "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "attempt_count": 1, "retried": False,
                "provider_task_compacted": False, "provider_task_compact_reason": "worker_exception",
                "provider_task_length": 0, "isolation": "failed_worker_isolated",
                "provider_health_hint_before": provider_health_hint(provider_health_dict, pid),
            }
            update_provider_health(provider_health_dict, pid, row, pmt)
            row["provider_health_hint_after"] = provider_health_hint(provider_health_dict, pid)
            per_model_results.append(row)
        else:
            per_model_results.append(item)
    execution = f"concurrent(all={len(participants)})"

    success_count = sum(1 for x in per_model_results if x.get("ok"))
    failed_count = len(per_model_results) - success_count
    partial_success = success_count > 0 and failed_count > 0
    all_failed = len(per_model_results) > 0 and success_count == 0
    aggregation = aggregate_real_answers(per_model_results=per_model_results, mode=mode)

    selected_index = int(aggregation.get("selected_index", -1))
    normalized_results: List[Dict[str, Any]] = []
    for idx, item in enumerate(per_model_results):
        row = dict(item)
        row["selected"] = idx == selected_index and selected_index >= 0
        row["selected_index"] = idx
        row["failure_type"] = classify_failure_reason(row.get("error", "")) if not row.get("ok") else ""
        row["is_timeout"] = is_timeout_error(row.get("error", ""))
        normalized_results.append(row)

    elapsed_list = [int(x.get("elapsed_ms", 0) or 0) for x in normalized_results]
    total_elapsed_ms = int((time.perf_counter() - started) * 1000)
    failure_reasons = [str(x.get("error", "") or "") for x in normalized_results if not x.get("ok")]
    timeout_failures = sum(1 for x in normalized_results if x.get("is_timeout"))

    merged_answer = aggregation.get("selected_text", "")
    llm_synthesized = False
    if enable_internal_llm_synthesis and success_count >= 2 and mode != "fast":
        async def _call_fn(pid: str, prompt: str) -> Dict[str, Any]:
            return await probe_real_call(context, provider_id=pid, task=prompt, timeout_sec=pmt)

        synth_started = time.perf_counter()
        synth_ans = await _llm_synthesis_layer(
            task, normalized_results, _call_fn, provider_health_dict, timeout_sec=pmt
        )
        stage_timings["llm_synthesis_elapsed_ms"] = int((time.perf_counter() - synth_started) * 1000)
        if synth_ans:
            merged_answer = synth_ans
            llm_synthesized = True
            execution += "+llm_synthesized"

    if success_count > 0:
        fallback_strategy = "partial_material_preferred"
        fallback_reason = "partial_success" if partial_success else "none"
    elif timeout_failures > 0:
        fallback_strategy = "aggressive_compact_then_retry"
        fallback_reason = "all_failed_with_timeout"
    else:
        fallback_strategy = "provider_fail_isolated"
        fallback_reason = "all_failed_non_timeout"

    health_summary = {pid: provider_health_hint(provider_health_dict, pid) for pid in participants}

    stage_timings["overall_elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    logger.info(f"[multi_model_compute] stage_timings real_multi_call={stage_timings}")
    _append_runtime_log(f"[multi_model_compute] stage_timings real_multi_call={stage_timings}")

    return {
        "participants": participants,
        "per_model_results": normalized_results,
        "success_count": success_count, "failed_count": failed_count,
        "partial_success": partial_success, "all_failed": all_failed,
        "merged_answer": merged_answer,
        "llm_synthesized": llm_synthesized,
        "execution": execution, "aggregation": aggregation,
        "timing": {
            "total_elapsed_ms": total_elapsed_ms,
            "avg_elapsed_ms": int(sum(elapsed_list) / max(1, len(elapsed_list))),
            "fastest_ms": min(elapsed_list) if elapsed_list else 0,
            "slowest_ms": max(elapsed_list) if elapsed_list else 0,
            "timeout_sec": timeout_sec,
            "stages": stage_timings,
        },
        "retry_summary": {"enabled": mode in {"consensus", "balanced", "fast"}, "attempted": retry_attempted, "succeeded": retry_succeeded, "failed": max(0, retry_attempted - retry_succeeded)},
        "fallback": {"strategy": fallback_strategy, "fallback_reason": fallback_reason, "failure_hints": failure_reasons[:4]},
        "provider_health": health_summary,
        "stability": {"max_models": cap_info, "query_length": len(task or ""), "task_budget": budget, "timeout_failures": timeout_failures},
    }


# ---------------------------------------------------------------------------
# 后端调度
# ---------------------------------------------------------------------------

async def calc_with_backend(
    *,
    context,
    task: str,
    mode: str,
    max_models: int,
    backend: str,
    context_summary: str,
    timeout_sec: float,
    per_model_timeout_sec: float = 200.0,
    default_count: int,
    default_mode: str,
    models: List[Dict[str, Any]],
    provider_health_dict: Dict[str, Dict[str, Any]],
    selected_provider_ids: List[str],
    return_merged: bool,
    return_candidates_flag: bool,
    enable_internal_llm_synthesis: bool = True,
) -> Dict[str, Any]:
    backend = (backend or "auto").strip().lower()
    if backend not in {"auto", "real", "mock"}:
        backend = "auto"

    calc_started = time.perf_counter()
    decision_log: List[str] = [f"request backend={backend}, mode={mode}, max_models={max_models}", f"selected providers={selected_provider_ids}"]
    backend_stage_timings: Dict[str, Any] = {}

    if backend in {"real", "auto"}:
        if not selected_provider_ids:
            decision_log.append("no selected providers")
            if backend == "real":
                raise RuntimeError("no selected providers in model_1~model_5 for real backend")
            mock_started = time.perf_counter()
            mock_result = mock_compute(
                task=task, context_summary=context_summary, mode=mode, max_models=max_models,
                models=models, default_mode=default_mode, default_count=default_count,
                return_merged=return_merged, return_candidates_flag=return_candidates_flag,
            )
            mock_result["backend"] = "mock"
            mock_result["fallback_from"] = "auto-real"
            mock_result["fallback_reason"] = "no selected providers in model_1~model_5"
            mock_result["fallback"] = {"strategy": "no_provider_then_mock", "fallback_reason": mock_result["fallback_reason"]}
            backend_stage_timings["mock_fallback_elapsed_ms"] = int((time.perf_counter() - mock_started) * 1000)
            mock_result["timing"] = {"backend_stages": backend_stage_timings, "calc_with_backend_elapsed_ms": int((time.perf_counter() - calc_started) * 1000)}
            mock_result["backend_decision_log"] = decision_log + ["fallback to mock"]
            _append_runtime_log(f"[multi_model_compute] backend_stage_timings={backend_stage_timings} timing={mock_result.get('timing', {})}")
            return mock_result

        real_call_started = time.perf_counter()
        real_result = await real_multi_call(
            context=context, task=task, max_models=max_models, mode=mode, context_summary=context_summary,
            timeout_sec=timeout_sec, per_model_timeout_sec=per_model_timeout_sec,
            default_count=default_count, provider_health_dict=provider_health_dict,
            selected_provider_ids=selected_provider_ids,
            enable_internal_llm_synthesis=enable_internal_llm_synthesis,
        )
        backend_stage_timings["real_call_elapsed_ms"] = int((time.perf_counter() - real_call_started) * 1000)
        success_count = int(real_result.get("success_count", 0) or 0)
        failed_count = int(real_result.get("failed_count", 0) or 0)
        partial_success = bool(real_result.get("partial_success"))
        all_failed = bool(real_result.get("all_failed"))
        decision_log.append(f"real call finished: success={success_count}, failed={failed_count}, partial={partial_success}")

        payload = {
            "used": True, "mode": mode, "query": task,
            "participants": real_result.get("participants", []),
            "backend": "real",
            "per_model_results": real_result.get("per_model_results", []),
            "merged_answer": real_result.get("merged_answer", "") if success_count > 0 else "[all real calls failed]",
            "aggregation": real_result.get("aggregation", {}),
            "execution": real_result.get("execution", "serial"),
            "timing": {**real_result.get("timing", {}), "backend_stages": backend_stage_timings, "calc_with_backend_elapsed_ms": int((time.perf_counter() - calc_started) * 1000)},
            "retry_summary": real_result.get("retry_summary", {}),
            "stability": real_result.get("stability", {}),
            "fallback": real_result.get("fallback", {}),
            "provider_health": real_result.get("provider_health", {}),
            "partial_success": partial_success, "all_failed": all_failed,
            "backend_decision_log": decision_log,
            "note": "v0.11 tool-first project synthesis stability",
        }

        _append_runtime_log(f"[multi_model_compute] backend_stage_timings={backend_stage_timings} timing={payload.get('timing', {})}")
        return payload

    # explicit mock
    mock_started = time.perf_counter()
    mock_result = mock_compute(
        task=task, context_summary=context_summary, mode=mode, max_models=max_models,
        models=models, default_mode=default_mode, default_count=default_count,
        return_merged=return_merged, return_candidates_flag=return_candidates_flag,
    )
    mock_result["backend"] = "mock"
    mock_result["fallback"] = {"strategy": "explicit_mock_backend", "fallback_reason": "backend explicitly mock"}
    backend_stage_timings["mock_fallback_elapsed_ms"] = int((time.perf_counter() - mock_started) * 1000)
    mock_result["timing"] = {"backend_stages": backend_stage_timings, "calc_with_backend_elapsed_ms": int((time.perf_counter() - calc_started) * 1000)}
    mock_result["backend_decision_log"] = decision_log + ["backend explicitly mock"]
    _append_runtime_log(f"[multi_model_compute] backend_stage_timings={backend_stage_timings} timing={mock_result.get('timing', {})}")
    return mock_result
