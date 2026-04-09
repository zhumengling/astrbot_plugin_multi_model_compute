"""材料综合与分析：共识检测、synthesis 构建、摘要/置信度/推荐派生。"""

from __future__ import annotations

from typing import Any, Dict, List

from .utils import (
    split_sentences,
    normalize_sentence_key,
    ngram_similarity,
    classify_failure_reason,
    is_retryable_failure_type,
    now_iso_utc,
    safe_dict,
    safe_list,
)
from .schema import RESULT_SCHEMA_VERSION, default_material_quality, map_level_to_overall

# 语义相似度阈值：>= 此值视为"表达同一含义"
_SEMANTIC_THRESHOLD = 0.4


def _find_or_create_bucket(
    buckets: List[Dict[str, Any]],
    sentence: str,
    key: str,
) -> Dict[str, Any]:
    """在已有桶中寻找语义匹配的桶，找不到则创建新桶。"""
    for bucket in buckets:
        if ngram_similarity(key, bucket["key"]) >= _SEMANTIC_THRESHOLD:
            return bucket
    new_bucket = {"key": key, "representative": sentence, "providers": []}
    buckets.append(new_bucket)
    return new_bucket


def build_synthesis_material(result: Dict[str, Any]) -> Dict[str, Any]:
    """核心综合逻辑：使用 n-gram 语义相似度做共识检测（替代字面匹配）。"""
    per_model_results = result.get("per_model_results", [])
    if not isinstance(per_model_results, list):
        per_model_results = []

    success_items: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for item in per_model_results:
        if not isinstance(item, dict):
            continue
        provider_id = str(item.get("provider_id", "") or "")
        if item.get("ok"):
            text = str(item.get("text", "") or "").strip()
            if text and text != "[empty completion_text]":
                success_items.append({
                    "provider_id": provider_id,
                    "text": text,
                    "elapsed_ms": int(item.get("elapsed_ms", 0) or 0),
                    "source": "real_per_model_results",
                    "selected": bool(item.get("selected")),
                })
            else:
                failures.append({
                    "provider_id": provider_id,
                    "error": "empty_text",
                    "failure_type": "empty_response",
                    "retryable": is_retryable_failure_type("empty_response"),
                    "elapsed_ms": int(item.get("elapsed_ms", 0) or 0),
                })
        else:
            err_msg = str(item.get("error", "unknown error") or "unknown error")
            failure_type = classify_failure_reason(err_msg)
            failures.append({
                "provider_id": provider_id,
                "error": err_msg,
                "failure_type": failure_type,
                "retryable": is_retryable_failure_type(failure_type),
                "elapsed_ms": int(item.get("elapsed_ms", 0) or 0),
            })

    # Fallback 到 mock candidates
    candidates = result.get("candidates", [])
    if not success_items and isinstance(candidates, list):
        for c in candidates:
            if not isinstance(c, dict):
                continue
            provider_id = str(c.get("model_id", "") or "mock_candidate")
            text = str(c.get("summary", "") or "").strip()
            if text:
                success_items.append({
                    "provider_id": provider_id,
                    "text": text,
                    "elapsed_ms": 0,
                    "source": "mock_candidates",
                    "selected": False,
                })

    # ---- 语义级句子分组（核心改进：n-gram 相似度替代字面匹配）----
    sentence_buckets: List[Dict[str, Any]] = []
    for item in success_items:
        provider_id = item["provider_id"]
        seen_keys: set = set()
        for sentence in split_sentences(item["text"]):
            key = normalize_sentence_key(sentence)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            bucket = _find_or_create_bucket(sentence_buckets, sentence, key)
            bucket["providers"].append(provider_id)

    common_points: List[Dict[str, Any]] = []
    notable_insights: List[Dict[str, Any]] = []
    overlap_groups: List[Dict[str, Any]] = []
    duplicate_candidates: List[Dict[str, Any]] = []

    for bucket in sentence_buckets:
        providers = sorted(list(set(bucket["providers"])))
        row = {
            "point": bucket["representative"],
            "support_providers": providers,
            "support_count": len(providers),
            "stance": "shared",
        }
        if len(providers) >= 2:
            common_points.append(row)
            overlap_groups.append({
                "group_key": bucket["key"],
                "providers": providers,
                "support_count": len(providers),
                "representative": bucket["representative"],
            })
            duplicate_candidates.append({
                "signature": bucket["key"],
                "providers": providers,
                "text": bucket["representative"],
            })
        else:
            notable_insights.append({**row, "stance": "unique"})

    common_points = sorted(common_points, key=lambda x: (-x["support_count"], x["point"]))[:8]
    notable_insights = sorted(notable_insights, key=lambda x: x["point"])[:8]
    overlap_groups = sorted(overlap_groups, key=lambda x: (-x["support_count"], x["group_key"]))[:8]
    duplicate_candidates = duplicate_candidates[:10]

    # Differences & conflict
    differences: List[Dict[str, Any]] = []
    conflict_points: List[Dict[str, Any]] = []
    for item in success_items[:8]:
        diff = {
            "provider_id": item["provider_id"],
            "position": item["text"][:220],
            "elapsed_ms": item["elapsed_ms"],
            "stance": "support" if item.get("selected") else "alternative",
        }
        differences.append(diff)

    if len(differences) >= 2:
        unique_provider_count = len({d.get("provider_id", "") for d in differences})
        if unique_provider_count >= 2 and len(common_points) == 0:
            conflict_points.append({
                "topic": "global_solution_direction",
                "conflict_level": "high",
                "providers": [d.get("provider_id", "") for d in differences[:3]],
                "note": "multiple distinct positions with weak overlap",
            })
        elif unique_provider_count >= 2:
            conflict_points.append({
                "topic": "implementation_detail",
                "conflict_level": "medium",
                "providers": [d.get("provider_id", "") for d in differences[:3]],
                "note": "partial overlap but different implementation details",
            })

    conflict_level = "low"
    if any(x.get("conflict_level") == "high" for x in conflict_points):
        conflict_level = "high"
    elif conflict_points:
        conflict_level = "medium"

    # Source/evidence refs
    source_refs: List[Dict[str, Any]] = []
    for idx, item in enumerate(success_items, start=1):
        ref_id = f"src_{idx}"
        source_refs.append({
            "ref_id": ref_id,
            "provider_id": item.get("provider_id", ""),
            "source": item.get("source", ""),
            "elapsed_ms": item.get("elapsed_ms", 0),
        })
        item["ref_id"] = ref_id

    evidence_refs: List[Dict[str, Any]] = []
    for cp in common_points[:8]:
        evidence_refs.append({
            "point": cp.get("point", ""),
            "provider_ids": cp.get("support_providers", []),
            "support_count": cp.get("support_count", 0),
            "confidence_hint": "higher" if cp.get("support_count", 0) >= 2 else "lower",
        })

    # Draft
    draft_lines: List[str] = []
    if common_points:
        draft_lines.append("Common points: " + " | ".join([x["point"] for x in common_points[:3]]))
    if notable_insights:
        draft_lines.append("Notable insights: " + " | ".join([x["point"] for x in notable_insights[:2]]))
    if failures:
        failure_models = ", ".join([f.get("provider_id", "") for f in failures if f.get("provider_id")])
        if failure_models:
            draft_lines.append(f"Failed providers: {failure_models}")

    # Recommended focus
    recommended_focus: List[str] = []
    if common_points:
        recommended_focus.append("Use common_points as baseline and verify against project constraints.")
    if notable_insights:
        recommended_focus.append("Review notable_insights for supplementary improvements.")
    if differences:
        recommended_focus.append("Compare differences before selecting implementation path.")
    if failures:
        recommended_focus.append("Check failures and retry with shorter/structured query if needed.")
    if not recommended_focus:
        recommended_focus.append("No strong material yet; retry with narrower scope.")

    # Stats
    participants_total = len(per_model_results)
    effective_total = participants_total if participants_total > 0 else len(success_items)
    success_count = len(success_items)
    failure_count = len(failures)
    success_ratio = round(success_count / max(1, effective_total), 2)

    caution_flags: List[str] = []
    if success_count == 0:
        caution_flags.append("no_success_material")
    if failure_count > 0:
        caution_flags.append("has_failures")
    if not common_points and success_count > 1:
        caution_flags.append("weak_consensus")
    if result.get("backend") == "mock":
        caution_flags.append("mock_material")

    quality_level = "low"
    if success_count >= 2 and common_points:
        quality_level = "high"
    elif success_count >= 1:
        quality_level = "medium"

    raw_materials = {
        "per_model_results": per_model_results,
        "candidates": candidates if isinstance(candidates, list) else [],
    }
    synthesized_material = {
        "common_points": common_points,
        "differences": differences,
        "notable_insights": notable_insights,
        "overlap_groups": overlap_groups,
        "duplicate_candidates": duplicate_candidates,
        "conflict_points": conflict_points,
        "conflict_level": conflict_level,
        "draft_synthesis": "\n".join(draft_lines).strip(),
    }

    core = {
        "material_ready": bool(success_items),
        "source_type": "real_per_model_results" if participants_total > 0 else ("mock_candidates" if success_items else "none"),
        "participants_total": participants_total,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_ratio": success_ratio,
        "common_points": common_points,
        "differences": differences,
        "notable_insights": notable_insights,
        "conflict_points": conflict_points,
        "conflict_level": conflict_level,
        "draft_synthesis": synthesized_material["draft_synthesis"],
        "recommended_focus": recommended_focus,
        "material_quality": {
            "level": quality_level,
            "overall": map_level_to_overall(quality_level),
            "coverage": round(float(success_count) / float(max(1, participants_total)), 3) if participants_total > 0 else 0.0,
            "consistency": {"high": 0.35, "medium": 0.6, "low": 0.85}.get(conflict_level, 0.6),
            "evidence_grounding": 0.8 if (source_refs or evidence_refs) else 0.5,
            "confidence_note": f"quality derived from success/failure balance and conflict level={conflict_level}",
            "caution_flags": caution_flags,
            "ready_for_final_summary": bool(success_items),
        },
        "all_failed": participants_total > 0 and success_count == 0,
        "partial_success": success_count > 0 and failure_count > 0,
        "retryable": any(bool(f.get("retryable")) for f in failures),
        "source_refs": source_refs,
        "evidence_refs": evidence_refs,
        "consumer_checklist": [
            "Use common_points as baseline rather than direct final answer.",
            "Review differences/notable_insights for alternatives.",
            "If failures exist, decide retry strategy before coding next step.",
        ],
    }

    diagnostics = {
        "overlap_groups": overlap_groups,
        "duplicate_candidates": duplicate_candidates,
        "source_attribution": {
            "raw_material_source": "per_model_results_or_mock_candidates",
            "synthesizer": "astrbot_plugin_multi_model_compute",
        },
        "provenance": {
            "generated_at": now_iso_utc(),
            "version": RESULT_SCHEMA_VERSION,
        },
        "raw_materials": raw_materials,
        "synthesized_material": synthesized_material,
    }

    synthesis = {
        "core": core,
        "diagnostics": diagnostics,
        # backward-compatible mirror fields (legacy consumers)
        "material_ready": core["material_ready"],
        "source_type": core["source_type"],
        "participants_total": core["participants_total"],
        "success_count": core["success_count"],
        "failure_count": core["failure_count"],
        "success_ratio": core["success_ratio"],
        "common_points": core["common_points"],
        "differences": core["differences"],
        "failures": failures,
        "notable_insights": core["notable_insights"],
        "overlap_groups": diagnostics["overlap_groups"],
        "duplicate_candidates": diagnostics["duplicate_candidates"],
        "conflict_points": core["conflict_points"],
        "conflict_level": core["conflict_level"],
        "draft_synthesis": core["draft_synthesis"],
        "recommended_focus": core["recommended_focus"],
        "material_quality": core["material_quality"],
        "all_failed": core["all_failed"],
        "partial_success": core["partial_success"],
        "retryable": core["retryable"],
        "source_refs": core["source_refs"],
        "evidence_refs": core["evidence_refs"],
        "source_attribution": diagnostics["source_attribution"],
        "provenance": diagnostics["provenance"],
        "raw_materials": diagnostics["raw_materials"],
        "synthesized_material": diagnostics["synthesized_material"],
        "consumer_checklist": core["consumer_checklist"],
    }
    return synthesis


# ---------------------------------------------------------------------------
# 派生字段
# ---------------------------------------------------------------------------

def derive_project_forward_fields(synthesis: Dict[str, Any]) -> Dict[str, Any]:
    common_points = safe_list(synthesis, "common_points")
    differences = safe_list(synthesis, "differences")
    failures = safe_list(synthesis, "failures")
    partial_success = bool(synthesis.get("partial_success"))
    all_failed = bool(synthesis.get("all_failed"))

    risk_register: List[Dict[str, Any]] = []
    for f in failures[:5]:
        risk_register.append({
            "risk": f"model_failure:{f.get('provider_id', 'unknown')}",
            "impact": "medium" if f.get("retryable") else "high",
            "signal": f.get("error", ""),
            "mitigation_hint": "retry_with_shorter_query_or_reduce_models",
        })
    if not common_points:
        risk_register.append({
            "risk": "weak_consensus",
            "impact": "medium",
            "signal": "insufficient common_points",
            "mitigation_hint": "reduce scope and re-run with fast mode",
        })

    open_questions: List[str] = []
    if differences:
        open_questions.append("Which viewpoint in differences best fits current project constraints?")
    if failures:
        open_questions.append("Should failed providers be retried with tighter prompt / fewer models?")
    if not common_points:
        open_questions.append("Do we need to split current_request into smaller subtasks?")

    implementation_suggestions: List[str] = []
    if common_points:
        implementation_suggestions.append("Use common_points as baseline implementation direction.")
    if differences:
        implementation_suggestions.append("Evaluate differences as optional branches/alternatives.")
    if all_failed:
        implementation_suggestions.append("Pause implementation decisions; re-run to gather at least one successful model output.")

    testing_suggestions = [
        "Create minimal validation checklist from common_points.",
        "Add regression checks for conflict_points and high-risk failures.",
    ]

    if all_failed:
        action = "Do not finalize project decisions yet; first retry to recover at least partial model materials."
    elif partial_success:
        action = "Use common_points as provisional baseline, explicitly mark failure uncertainty, then plan focused retry."
    else:
        action = "Synthesize common_points + notable_insights, resolve conflict_points explicitly, then continue project implementation."

    next_steps = [
        "Review synthesis.core.common_points and pick a baseline path.",
        "Inspect synthesis.core.conflict_points / overlap_groups before coding decision.",
    ]
    if all_failed:
        next_steps.append("Retry with mode=fast, backend=auto, max_models=2, and shorter current_request.")
    elif partial_success:
        next_steps.append("Proceed with partial materials, then retry failed providers using compact prompt.")
    else:
        next_steps.append("Proceed implementation and keep evidence_refs for traceable summary.")

    return {
        "risk_register": risk_register,
        "open_questions": open_questions,
        "implementation_suggestions": implementation_suggestions,
        "testing_suggestions": testing_suggestions,
        "recommended_default_model_action": action,
        "recommended_next_steps": next_steps,
    }


def derive_summary_source(result: Dict[str, Any]) -> Dict[str, Any]:
    backend_used = result.get("backend", "")
    synthesis = safe_dict(result, "synthesis")
    aggregation = safe_dict(result, "aggregation")

    if backend_used == "real" and synthesis.get("material_ready"):
        return {"type": "real_synthesis_material", "strategy": "common_points_first", "success_count": synthesis.get("success_count", 0)}
    if backend_used == "mock":
        return {"type": "mock_synthesis_material", "strategy": "candidates_plus_mock_merge", "success_count": synthesis.get("success_count", 0)}
    if aggregation:
        return {"type": "aggregation_fallback", "strategy": aggregation.get("strategy", ""), "provider_id": aggregation.get("selected_provider_id", "")}
    return {"type": "unknown", "strategy": "none", "success_count": synthesis.get("success_count", 0)}


def derive_confidence(result: Dict[str, Any]) -> float:
    synthesis = safe_dict(result, "synthesis")
    success = int(synthesis.get("success_count", 0) or 0)
    total = int(synthesis.get("participants_total", 0) or 0)
    common = len(safe_list(synthesis, "common_points"))
    fail = int(synthesis.get("failure_count", 0) or 0)

    if total <= 0:
        return 0.45
    ratio = success / max(1, total)
    base = 0.5 + 0.28 * ratio
    base += min(0.12, 0.03 * common)
    base -= min(0.1, 0.02 * fail)
    return round(max(0.35, min(base, 0.92)), 2)


def derive_recommendation(result: Dict[str, Any]) -> str:
    synthesis = safe_dict(result, "synthesis")
    common_points = safe_list(synthesis, "common_points")
    notable_insights = safe_list(synthesis, "notable_insights")
    failures = safe_list(synthesis, "failures")

    if not synthesis.get("material_ready"):
        return "先检查 failures 与 fallback_reason，若允许可重试 backend=auto/real；当前不建议直接下最终结论。"
    if failures:
        return "请默认模型以 common_points 为主线、notable_insights 为补充，并在最终答案中标注 failures 带来的不确定性。"
    if common_points and notable_insights:
        return "请默认模型先提炼 common_points 形成主结论，再融合 notable_insights 与 differences 生成最终答案。"
    if common_points:
        return "请默认模型基于 common_points 直接归纳最终答案，并结合 differences 做必要取舍说明。"
    return "请默认模型综合 differences 与 per_model_results 自主归纳，避免单条输出直接当作最终答案。"
