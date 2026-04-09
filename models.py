"""模型池管理：加载、智能选择（基于标签匹配）、能力描述。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

MODEL_SLOT_COUNT = 5

# 任务关键词 → 标签的映射表
_TASK_TAG_MAPPING: Dict[str, List[str]] = {
    # coding / debug
    "code": ["coding", "debug", "review"],
    "coding": ["coding", "debug", "review"],
    "python": ["coding", "debug"],
    "java": ["coding", "debug"],
    "bug": ["debug", "coding", "review"],
    "debug": ["debug", "coding", "review"],
    "报错": ["debug", "coding", "review"],
    "异常": ["debug", "coding", "review"],
    "接口": ["coding", "analysis"],
    "api": ["coding", "analysis"],
    "架构": ["analysis", "reasoning", "stable"],
    "脚本": ["coding", "fast"],
    # analysis / decision
    "分析": ["analysis", "reasoning", "stable"],
    "推理": ["reasoning", "analysis", "stable"],
    "评估": ["analysis", "reasoning", "review"],
    "比较": ["analysis", "reasoning", "review"],
    "决策": ["analysis", "reasoning", "stable"],
    "方案": ["analysis", "reasoning", "creative"],
    # creative
    "创意": ["creative"],
    "写作": ["creative", "analysis"],
    "文案": ["creative", "fast"],
    # speed / summary
    "快速": ["fast"],
    "总结": ["analysis", "fast"],
    "摘要": ["analysis", "fast"],
    "归纳": ["analysis", "reasoning"],
    # multimodal-ish
    "图片": ["vision", "creative"],
    "视觉": ["vision", "creative"],
}

_TAG_ALIASES: Dict[str, str] = {
    "代码": "coding",
    "编程": "coding",
    "调试": "debug",
    "审查": "review",
    "评审": "review",
    "推理": "reasoning",
    "分析": "analysis",
    "稳定": "stable",
    "创意": "creative",
    "快速": "fast",
    "便宜": "cheap",
    "视觉": "vision",
}



def _normalize_tag(tag: str) -> str:
    raw = str(tag or "").strip().lower()
    return _TAG_ALIASES.get(raw, raw)


def _parse_tags(tag_str: str) -> List[str]:
    """将逗号分隔的标签字符串解析为标签列表。"""
    if not tag_str:
        return []
    tags: List[str] = []
    for t in str(tag_str).split(","):
        norm = _normalize_tag(t)
        if norm and norm not in tags:
            tags.append(norm)
    return tags


def _extract_task_tags(task: str) -> List[str]:
    """从任务文本中提取匹配的标签集合。"""
    text = (task or "").lower()
    matched_tags: List[str] = []
    for keyword, tags in _TASK_TAG_MAPPING.items():
        if keyword in text:
            for tag in tags:
                norm = _normalize_tag(tag)
                if norm not in matched_tags:
                    matched_tags.append(norm)
    return matched_tags


def _tag_match_score(model_tags: List[str], task_tags: List[str]) -> float:
    """计算模型标签与任务标签的匹配分数（0~1）。"""
    if not model_tags or not task_tags:
        return 0.0
    model_set = set(t.lower() for t in model_tags)
    task_set = set(t.lower() for t in task_tags)
    intersection = model_set & task_set
    # Jaccard 系数
    union = model_set | task_set
    return len(intersection) / max(1, len(union))


def load_builtin_models(config_get) -> List[Dict[str, Any]]:
    configured = config_get("builtin_models", None)
    if isinstance(configured, list) and configured:
        return configured
    return [
        {
            "id": "mock_reasoner",
            "provider": "mock",
            "strengths": ["推理", "分析", "多步骤回答"],
            "best_for": ["analysis", "planning", "comparison"],
            "tags": ["推理", "分析", "比较"],
            "speed": "medium",
            "cost": "low",
        },
        {
            "id": "mock_coder",
            "provider": "mock",
            "strengths": ["代码", "技术实现", "调试思路"],
            "best_for": ["code", "debug", "architecture"],
            "tags": ["代码", "调试", "架构"],
            "speed": "medium",
            "cost": "medium",
        },
        {
            "id": "mock_summarizer",
            "provider": "mock",
            "strengths": ["总结", "提炼", "归纳"],
            "best_for": ["summary", "rewrite", "merge"],
            "tags": ["总结", "归纳", "摘要"],
            "speed": "fast",
            "cost": "low",
        },
        {
            "id": "mock_critic",
            "provider": "mock",
            "strengths": ["反思", "风险识别", "一致性检查"],
            "best_for": ["review", "risk", "verification"],
            "tags": ["推理", "分析", "审查"],
            "speed": "slow",
            "cost": "low",
        },
    ]


def load_slot_selected_models(config_get) -> List[Dict[str, Any]]:
    selected = []
    for i in range(1, MODEL_SLOT_COUNT + 1):
        raw = config_get(f"model_{i}", "")
        if not raw:
            continue
        model_id = str(raw).strip()
        if not model_id:
            continue

        # 读取用户配置的 tags
        tag_str = str(config_get(f"model_{i}_tags", "") or "").strip()
        user_tags = _parse_tags(tag_str)

        selected.append(
            {
                "id": model_id,
                "provider": "astrbot_configured",
                "strengths": user_tags if user_tags else ["由 AstrBot 已配置模型提供能力"],
                "best_for": ["general", "configured"],
                "tags": user_tags,
                "speed": "unknown",
                "cost": "unknown",
                "selected_from_slot": f"model_{i}",
            }
        )
    return selected


def load_runtime_models(config_get) -> List[Dict[str, Any]]:
    slot_models = load_slot_selected_models(config_get)
    if slot_models:
        return slot_models
    builtin = load_builtin_models(config_get)
    if builtin:
        return builtin
    return []


def _health_score(pid: str, health_dict: Dict[str, Dict[str, Any]]) -> float:
    """基于 provider_health 计算健康分（0~1），越高越优先。"""
    h = health_dict.get(pid)
    if not isinstance(h, dict) or not h:
        return 0.5  # 未知状态给中等分
    calls = int(h.get("calls", 0) or 0)
    if calls == 0:
        return 0.5
    success = int(h.get("success", 0) or 0)
    success_rate = success / max(1, calls)
    consecutive_failures = int(h.get("consecutive_failures", 0) or 0)
    avg_elapsed = int(h.get("avg_elapsed_ms", 0) or 0)

    # 基础分 = 成功率
    score = success_rate * 0.7
    # 速度加分（越快越好，2000ms以下满分）
    speed_bonus = max(0, 1 - avg_elapsed / 5000) * 0.2
    score += speed_bonus
    # 连续失败惩罚
    if consecutive_failures >= 3:
        score *= 0.3
    elif consecutive_failures >= 2:
        score *= 0.5
    elif consecutive_failures >= 1:
        score *= 0.8
    # 经验加分（调用越多越可信）
    exp_bonus = min(calls / 20, 1.0) * 0.1
    score += exp_bonus

    return round(max(0, min(1, score)), 3)


def classify_task_profile(task: str, mode: str = "balanced") -> Dict[str, Any]:
    tags = _extract_task_tags(task)
    tag_set = set(tags)
    if {"coding", "debug", "review"} & tag_set:
        family = "coding"
    elif {"creative", "vision"} & tag_set:
        family = "creative"
    elif {"analysis", "reasoning", "stable"} & tag_set:
        family = "analysis"
    elif {"fast"} & tag_set:
        family = "fast"
    else:
        family = "general"

    if mode == "fast":
        recommended = 1 if family == "fast" else 2
    elif mode == "creative":
        recommended = 2
    elif mode == "consensus":
        recommended = 3 if family in {"analysis", "coding"} else 2
    else:  # balanced / auto fallback
        recommended = 3 if family in {"analysis", "coding"} else 2

    return {"family": family, "tags": tags, "recommended_count": recommended}


def select_provider_ids_for_task(
    models: List[Dict[str, Any]],
    task: str,
    mode: str,
    requested_max_models: int,
    health_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    profile = classify_task_profile(task, mode)
    available = len(models)
    if available <= 0:
        return [], {"family": profile["family"], "task_tags": profile["tags"], "requested": requested_max_models, "effective": 0, "reason": "no_available_models"}

    desired = int(requested_max_models) if int(requested_max_models or 0) > 0 else profile["recommended_count"]
    desired = max(1, min(desired, available))

    ranked = choose_models(models=models, task=task, mode=mode, max_models=available, health_dict=health_dict)
    provider_ids = [str(m.get("id", "") or "") for m in ranked if str(m.get("id", "") or "")]
    provider_ids = provider_ids[:desired]
    return provider_ids, {
        "family": profile["family"],
        "task_tags": profile["tags"],
        "requested": requested_max_models,
        "effective": len(provider_ids),
        "reason": "tag_health_routing",
    }


def rank_by_health(
    provider_ids: List[str],
    health_dict: Dict[str, Dict[str, Any]],
) -> List[str]:
    """基于历史表现对 provider 排序：健康分高的排前面。"""
    if not health_dict:
        return list(provider_ids)
    scored = [(pid, _health_score(pid, health_dict)) for pid in provider_ids]
    scored.sort(key=lambda x: -x[1])
    return [pid for pid, _ in scored]


def choose_models(
    models: List[Dict[str, Any]],
    task: str,
    mode: str,
    max_models: int,
    health_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """基于标签匹配 + 历史表现的智能模型选择。

    选择策略：
    1. 从任务文本中提取语义标签
    2. 计算每个模型的标签匹配分数
    3. 结合 provider_health 历史表现做综合排序
    4. mode 修正：fast 优先速度，creative 反转顺序，consensus 限制数量
    """
    models = list(models)
    if not models:
        return []

    task_tags = _extract_task_tags(task)
    health_dict = health_dict or {}

    # 综合评分排序
    scored = []
    for m in models:
        # 标签匹配分
        model_tags = m.get("tags", [])
        all_tags = list(model_tags) + m.get("strengths", []) + m.get("best_for", [])
        tag_score = _tag_match_score(all_tags, task_tags) if task_tags else 0.0
        # 健康分
        pid = m.get("id", "")
        h_score = _health_score(pid, health_dict)
        # 综合分 = tag 权重 0.6 + health 权重 0.4
        combined = tag_score * 0.6 + h_score * 0.4
        scored.append((combined, tag_score, h_score, m))

    scored.sort(key=lambda x: -x[0])
    models = [m for _, _, _, m in scored]

    # mode 修正
    if mode == "fast":
        models = sorted(models, key=lambda x: (
            -_tag_match_score(x.get("tags", []) + x.get("strengths", []) + x.get("best_for", []), task_tags) if task_tags else 0,
            {"fast": 0, "medium": 1, "slow": 2, "unknown": 1}.get(str(x.get("speed", "unknown")), 1),
        ))
    elif mode == "creative":
        models = list(reversed(models))
    elif mode == "consensus":
        return models[: max(1, min(max_models, len(models)))]

    return models[: max(1, min(max_models, len(models)))]


def find_selected_provider_ids(config_get) -> List[str]:
    ids: List[str] = []
    for i in range(1, MODEL_SLOT_COUNT + 1):
        raw = config_get(f"model_{i}", "")
        provider_id = str(raw or "").strip()
        if provider_id:
            ids.append(provider_id)
    return ids


def render_model_line(m: Dict[str, Any]) -> str:
    slot = m.get("selected_from_slot")
    slot_part = f" | 槽位: {slot}" if slot else ""
    tags = m.get("tags", [])
    tag_part = f" | 标签: {', '.join(tags)}" if tags else ""
    strengths = ", ".join(m.get("strengths", []))
    return f"- {m.get('id', 'unknown')} ({m.get('provider', 'unknown')}){slot_part}{tag_part} | 擅长: {strengths}"


def build_slot_status(config_get) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    for i in range(1, MODEL_SLOT_COUNT + 1):
        slot_name = f"model_{i}"
        provider_id = str(config_get(slot_name, "") or "").strip()
        tag_str = str(config_get(f"{slot_name}_tags", "") or "").strip()
        slots.append(
            {
                "slot": slot_name,
                "provider_id": provider_id,
                "enabled": bool(provider_id),
                "tags": _parse_tags(tag_str),
            }
        )
    return slots
