"""通用工具函数：无插件状态依赖的纯函数。"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 时间 / 错误 / 比率
# ---------------------------------------------------------------------------

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_error(e: Exception) -> str:
    name = type(e).__name__
    msg = str(e).strip()
    return f"{name}: {msg}" if msg else name


def to_ratio(used: int, limit: int) -> float:
    if not limit or int(limit) <= 0:
        return 0.0
    return round(float(max(0, int(used))) / float(max(1, int(limit))), 4)


# ---------------------------------------------------------------------------
# 安全取值（替代全局 50+ 处重复类型守卫）
# ---------------------------------------------------------------------------

def safe_dict(d: Dict[str, Any], key: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    val = d.get(key)
    return val if isinstance(val, dict) else (default if default is not None else {})


def safe_list(d: Dict[str, Any], key: str) -> List[Any]:
    val = d.get(key)
    return val if isinstance(val, list) else []


# ---------------------------------------------------------------------------
# 句子分割 / 归一化 / 语义相似度
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    parts = re.split(r"[。！？!?\n]+", cleaned)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_sentence_key(sentence: str) -> str:
    s = re.sub(r"\s+", " ", str(sentence or "").strip().lower())
    s = re.sub(r"[^\w一-鿿 ]+", "", s)
    return s[:120]


def ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """Character-level n-gram (Jaccard) 相似度。

    用于替代字面完全匹配的共识检测。
    对中文短句，character bigram 无需分词依赖，阈值 0.4 经验上效果较好。
    """
    t1 = list(normalize_sentence_key(s1))
    t2 = list(normalize_sentence_key(s2))
    if len(t1) < n or len(t2) < n:
        return 1.0 if t1 == t2 else 0.0
    ng1 = set(zip(*[t1[i:] for i in range(n)]))
    ng2 = set(zip(*[t2[i:] for i in range(n)]))
    intersection = ng1 & ng2
    union = ng1 | ng2
    return len(intersection) / max(1, len(union))


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------

def classify_failure_reason(error: str) -> str:
    msg = str(error or "").strip().lower()
    if not msg:
        return "unknown"
    if "timeout" in msg:
        return "timeout"
    if "empty_text" in msg or "empty completion" in msg or "emptymodeloutput" in msg:
        return "empty_response"
    if "provider" in msg and ("not found" in msg or "unavailable" in msg or "失败" in msg):
        return "provider_unavailable"
    if "auth" in msg or "permission" in msg or "unauthorized" in msg:
        return "auth_error"
    return "runtime_error"


def is_retryable_failure_type(failure_type: str) -> bool:
    return failure_type in {"timeout", "runtime_error", "provider_unavailable"}


def is_timeout_error(error: str) -> bool:
    return "timeout" in str(error or "").strip().lower()
