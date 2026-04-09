"""结果缓存：基于内存的短期缓存，避免重复多模型调用。"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional

# 默认 TTL 300 秒（5 分钟）
DEFAULT_CACHE_TTL = 300


class ResultCache:
    """线程安全的内存缓存，按 query+mode+backend+extra_sig 做键。

    设计决策：
    - 使用内存缓存而非 KV 存储，因为缓存需要高频读写且生命周期短。
    - KV 存储适合持久化数据（如 provider_health），缓存适合瞬态数据。
    - 缓存容量上限 64 条，LRU 淘汰最旧条目。
    """

    MAX_ENTRIES = 64

    def __init__(self, ttl_sec: int = DEFAULT_CACHE_TTL):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._ttl_sec = max(0, ttl_sec)

    @property
    def ttl_sec(self) -> int:
        return self._ttl_sec

    @ttl_sec.setter
    def ttl_sec(self, value: int):
        self._ttl_sec = max(0, value)

    @staticmethod
    def _make_key(query: str, mode: str, backend: str, extra_sig: str = "") -> str:
        # v2: 增加 extra_sig 参与键，避免不同 max_models / 模型配置命中同一缓存。
        raw = (
            f"v2|{(query or '').strip()}|"
            f"{(mode or 'balanced').strip().lower()}|"
            f"{(backend or 'auto').strip().lower()}|"
            f"{(extra_sig or '').strip()}"
        )
        return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:20]

    def get(self, query: str, mode: str, backend: str, extra_sig: str = "") -> Optional[Dict[str, Any]]:
        """命中缓存则返回结果 dict，否则返回 None。"""
        if self._ttl_sec <= 0:
            return None
        key = self._make_key(query, mode, backend, extra_sig)
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() - entry["ts"] > self._ttl_sec:
            self._store.pop(key, None)
            return None
        entry["hits"] = entry.get("hits", 0) + 1
        return entry["result"]

    def put(self, query: str, mode: str, backend: str, result: Dict[str, Any], extra_sig: str = ""):
        """写入缓存。超过容量上限时淘汰最旧条目。"""
        if self._ttl_sec <= 0:
            return
        key = self._make_key(query, mode, backend, extra_sig)
        self._store[key] = {"result": result, "ts": time.time(), "hits": 0}
        self._evict_if_needed()

    def invalidate(self, query: str = "", mode: str = "", backend: str = "", extra_sig: str = ""):
        """手动失效指定缓存条目；若全部为空则清空所有缓存。"""
        if not query and not mode and not backend:
            self._store.clear()
            return
        key = self._make_key(query, mode, backend, extra_sig)
        self._store.pop(key, None)

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息。"""
        now = time.time()
        alive = sum(1 for e in self._store.values() if now - e["ts"] <= self._ttl_sec)
        total_hits = sum(e.get("hits", 0) for e in self._store.values())
        return {
            "total_entries": len(self._store),
            "alive_entries": alive,
            "expired_entries": len(self._store) - alive,
            "total_hits": total_hits,
            "ttl_sec": self._ttl_sec,
            "max_entries": self.MAX_ENTRIES,
        }

    def _evict_if_needed(self):
        """淘汰过期条目；若仍超容量则移除最旧的。"""
        now = time.time()
        # 先清过期
        expired_keys = [k for k, v in self._store.items() if now - v["ts"] > self._ttl_sec]
        for k in expired_keys:
            del self._store[k]
        # 再按时间淘汰
        while len(self._store) > self.MAX_ENTRIES:
            oldest_key = min(self._store, key=lambda k: self._store[k]["ts"])
            del self._store[oldest_key]
