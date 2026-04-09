"""多模型材料汇总 Tool-First 插件 v0.13 — 精简主文件。

职责：Star 子类定义 + Handler/Tool 注册入口。
业务逻辑已拆分至 utils / models / provider_call / synthesis / schema / project / cache / report 模块。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .utils import now_iso_utc, safe_dict, safe_list
from .cache import ResultCache
from .debate import DebateEngine
from .models import (
    MODEL_SLOT_COUNT,
    load_runtime_models,
    find_selected_provider_ids,
    render_model_line,
    build_slot_status,
    load_slot_selected_models,
    select_provider_ids_for_task,
    classify_task_profile,
)
from .provider_call import (
    calc_with_backend,
    mock_compute,
    probe_real_call,
    provider_health_hint,
    extract_text_from_response,
)
from .synthesis import (
    build_synthesis_material,
    derive_project_forward_fields,
    derive_summary_source,
    derive_confidence,
    derive_recommendation,
)
from .schema import (
    RESULT_SCHEMA_VERSION,
    default_synthesis,
    default_material_quality,
    ensure_schema_invariants,
    fallback_applied as _fallback_applied,
    derive_execution_status,
    status_v2 as _status_v2,
)
from .project import (
    normalize_project_payload,
    build_project_task_and_meta,
    build_ids,
    task_budget_by_mode,
)
from .webui import WebUIServer

VALID_MODES = {"balanced", "fast", "consensus", "creative", "debate", "auto"}
VALID_BACKENDS = {"auto", "real", "mock"}
PLUGIN_VERSION = "v0.13.1"


def route_task_mode(task: str) -> str:
    """根据任务特征返回执行模式（P3-1 任务分级路由）"""
    task_str = str(task or "").strip()
    length = len(task_str)
    if length < 50 and not any(kw in task_str for kw in ["分析", "评估", "比较", "建议", "选择", "探讨", "规划"]):
        return "fast"       # 简单问答：速度优先
    elif any(kw in task_str for kw in ["代码", "实现", "debug", "解决", "脚本", "报错", "异常", "逻辑"]):
        return "balanced"   # 技术任务：2模型并行 + 1次综合
    else:
        return "consensus"  # 复杂分析：强制要求2个以上模型共识


@register(
    "astrbot_plugin_multi_model_compute",
    "OpenAI",
    "供主模型调用的多模型计算插件",
    PLUGIN_VERSION,
)
class MultiModelComputePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config: AstrBotConfig = config

        self.models = load_runtime_models(self._config_get)
        self._provider_health: Dict[str, Dict[str, Any]] = {}
        self._health_loaded = False
        self.webui_server = None

        # 结果缓存
        cache_ttl = self._cache_ttl_sec_from_config(config)
        self._cache = ResultCache(ttl_sec=cache_ttl)

        # 安全默认：不在启动时自动修改全局配置。可通过配置显式开启。
        if bool(self._config_get("auto_raise_tool_call_timeout_on_startup", False)):
            self._ensure_tool_call_timeout(context, min_timeout=420)
        else:
            logger.info("[multi_model_compute] skip auto tool_call_timeout update on startup (secure default)")

        logger.info(f"[multi_model_compute] plugin initialized ({PLUGIN_VERSION}), cache_ttl={cache_ttl}s")
        import asyncio
        asyncio.create_task(self._start_webui())

    def _sanitize_plain_text_answer(self, text: Any) -> str:
        """将常见 Markdown 展示标记清洗为纯文本（仅用于用户最终展示）。"""
        s = str(text or "")
        if not s:
            return ""

        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # 标题前缀: # ## ### ...
        s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
        # 粗体 / 强调
        s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
        s = re.sub(r"__(.*?)__", r"\1", s)
        # 行内强调残留星号（尽量保守，仅去掉夹在非空白字符之间的 *）
        s = re.sub(r"(?<=\S)\*(?=\S)", "", s)

        # 清理可能的标题尾部 ###
        s = re.sub(r"(?m)\s+#{1,6}\s*$", "", s)
        # 压缩过多空行
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    @staticmethod
    def _cache_ttl_sec_from_config(config) -> int:
        """从配置读取缓存 TTL（__init__ 中 _config_get 尚未可用，直接读 config）。"""
        try:
            if isinstance(config, dict):
                return max(0, int(config.get("cache_ttl_sec", 300)))
            getter = getattr(config, "get", None)
            if callable(getter):
                return max(0, int(getter("cache_ttl_sec", 300)))
        except Exception:
            pass
        return 300

    @staticmethod
    def _ensure_tool_call_timeout(context: Context, min_timeout: int = 420) -> None:
        """检查并提升所有 AstrBot 配置文件中的 provider_settings.tool_call_timeout。

        AstrBot v4.0.0 起支持多配置文件（默认 + data/config/abconf_*.json）。
        此方法遍历全部已加载的配置，对值小于 min_timeout 的逐一写入并持久化，
        确保本插件的工具调用不会被 120s 的默认超时截断。
        """
        try:
            cfg_mgr = context.astrbot_config_mgr          # AstrBotConfigManager
            all_confs: dict = cfg_mgr.confs               # {"default": cfg, uuid: cfg, ...}
        except Exception as e:
            logger.warning(f"[multi_model_compute] 获取配置管理器失败，跳过 tool_call_timeout 检查: {e}")
            return

        updated, skipped = 0, 0
        for conf_id, astrbot_cfg in all_confs.items():
            try:
                provider_settings = astrbot_cfg.get("provider_settings", {})
                if not isinstance(provider_settings, dict):
                    continue
                current = int(provider_settings.get("tool_call_timeout", 120))
                if current < min_timeout:
                    provider_settings["tool_call_timeout"] = min_timeout
                    astrbot_cfg["provider_settings"] = provider_settings
                    astrbot_cfg.save_config()
                    logger.info(
                        f"[multi_model_compute] [{conf_id}] tool_call_timeout "
                        f"{current}s → {min_timeout}s（{astrbot_cfg.config_path}）"
                    )
                    updated += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"[multi_model_compute] [{conf_id}] 调整 tool_call_timeout 失败: {e}")

        logger.info(
            f"[multi_model_compute] tool_call_timeout 检查完毕："
            f"已更新 {updated} 个配置，{skipped} 个已满足 >={min_timeout}s"
        )

    async def _ensure_health_loaded(self):
        """Lazy 加载 provider_health：首次调用时从 KV 存储恢复。"""
        if self._health_loaded:
            return
        self._health_loaded = True
        try:
            saved = await self.get_kv_data("provider_health")
            if isinstance(saved, dict) and saved:
                self._provider_health.update(saved)
                logger.info(f"[multi_model_compute] restored provider_health from KV ({len(saved)} providers)")
            
            # P2-5: 恢复 ReasoningBank
            rb_saved = await self.get_kv_data("reasoning_bank")
            if isinstance(rb_saved, dict):
                self._reasoning_bank = rb_saved
                logger.info(f"[multi_model_compute] restored reasoning_bank ({len(rb_saved)} trajectories)")
            else:
                self._reasoning_bank = {}
        except Exception as e:
            logger.debug(f"[multi_model_compute] KV load skipped: {e}")

    async def terminate(self):
        '''插件卸载/停用时持久化健康数据并清理。'''
        if self._provider_health:
            try:
                await self.put_kv_data("provider_health", dict(self._provider_health))
                logger.info(f"[multi_model_compute] saved provider_health to KV ({len(self._provider_health)} providers)")
            except Exception as e:
                logger.warning(f"[multi_model_compute] health KV save failed: {e}")
        
        if getattr(self, "_reasoning_bank", None) is not None:
            try:
                await self.put_kv_data("reasoning_bank", dict(self._reasoning_bank))
                logger.info(f"[multi_model_compute] saved reasoning_bank to KV ({len(self._reasoning_bank)})")
            except Exception as e:
                logger.warning(f"[multi_model_compute] reasoning_bank KV save failed: {e}")
                
        await self._stop_webui()
        self._provider_health.clear()
        self._cache.invalidate()
        logger.info("[multi_model_compute] plugin terminated")


    def _webui_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self._config_get("webui_enabled", True)),
            "host": str(self._config_get("webui_host", "0.0.0.0") or "0.0.0.0"),
            "port": int(self._config_get("webui_port", 8099) or 8099),
        }

    async def _start_webui(self):
        cfg = self._webui_config()
        if not cfg.get("enabled"):
            return
        if self.webui_server:
            return
        try:
            self.webui_server = WebUIServer(config=cfg)
            await self.webui_server.start()
            actual_port = getattr(self.webui_server, 'actual_port', cfg.get('port'))
            logger.info(f"[multi_model_compute] WebUI started at http://{cfg.get('host')}:{actual_port}")
        except Exception as e:
            logger.warning(f"[multi_model_compute] WebUI start failed: {e}")
            self.webui_server = None

    async def _stop_webui(self):
        if not self.webui_server:
            return
        try:
            await self.webui_server.stop()
        except Exception as e:
            logger.warning(f"[multi_model_compute] WebUI stop failed: {e}")
        finally:
            self.webui_server = None

    # -----------------------------------------------------------------------
    # Config helpers
    # -----------------------------------------------------------------------

    def _config_get(self, key: str, default=None):
        cfg = self.config
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        getter = getattr(cfg, "get", None)
        if callable(getter):
            try:
                return getter(key, default)
            except Exception:
                return default
        return default

    def _default_mode(self) -> str:
        mode = str(self._config_get("default_mode", "balanced") or "balanced").strip().lower()
        return mode if mode in VALID_MODES else "balanced"

    def _default_participant_count(self) -> int:
        """内部默认值：仅在未显式指定且无法从任务特征推断时兜底使用。"""
        return 3

    def _return_merged_answer(self) -> bool:
        return bool(self._config_get("return_merged_answer", True))

    def _return_candidates(self) -> bool:
        return bool(self._config_get("return_candidates", True))

    def _real_call_timeout_sec(self) -> float:
        """每个模型单次请求的独立超时（60s），超时则断开该模型，不影响其他并行模型。"""
        raw = self._config_get("real_call_timeout_sec", 60)
        try:
            value = float(raw)
            return value if value > 0 else 60.0
        except Exception:
            return 60.0

    @staticmethod
    def _event_umo(event: Any) -> str:
        for attr in ("unified_msg_origin", "unified_message_origin", "session_id"):
            try:
                value = getattr(event, attr, None)
            except Exception:
                value = None
            if value:
                return str(value)
        return ""

    async def _call_default_chat_model(self, event: AstrMessageEvent, prompt: str, timeout_sec: float = 0.0) -> str:
        prompt = str(prompt or "").strip()
        if not prompt:
            return ""

        umo = self._event_umo(event)
        provider = None
        try:
            if umo:
                provider = self.context.get_using_provider(umo=umo)
            else:
                provider = self.context.get_using_provider()
        except TypeError:
            provider = self.context.get_using_provider()
        except Exception as e:
            logger.warning(f"[multi_model_compute] get default provider failed: {e}")
            return ""

        if not provider:
            logger.warning("[multi_model_compute] default chat provider unavailable")
            return ""

        text_chat = getattr(provider, "text_chat", None)
        if not callable(text_chat):
            logger.warning(f"[multi_model_compute] default provider has no text_chat: {type(provider)}")
            return ""

        async def _invoke() -> Any:
            sid = f"mm-default-summary-{uuid.uuid4().hex[:10]}"
            try:
                ret = text_chat(prompt=prompt, session_id=sid, contexts=[])
            except TypeError:
                ret = text_chat(prompt=prompt, session_id=sid)
            if hasattr(ret, "__await__"):
                return await ret
            return ret

        effective_timeout = float(timeout_sec) if timeout_sec and timeout_sec > 0 else max(30.0, self._real_call_timeout_sec())
        try:
            llm_resp = await asyncio.wait_for(_invoke(), timeout=effective_timeout)
        except Exception as e:
            logger.warning(f"[multi_model_compute] default model summarize failed: {e}")
            return ""

        return str(extract_text_from_response(llm_resp) or "").strip()

    def _cache_ttl_sec(self) -> int:
        raw = self._config_get("cache_ttl_sec", 300)
        try:
            return max(0, int(raw))
        except Exception:
            return 300

    @staticmethod
    def _parse_bool_like(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on", "admin", "owner", "superadmin", "super_admin", "superuser", "master"}:
                return True
            if v in {"0", "false", "no", "n", "off", "member", "user", "guest"}:
                return False
        return None

    @staticmethod
    def _deep_get_attr(obj: Any, path: str, default: Any = None) -> Any:
        cur = obj
        for part in path.split('.'):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(part, default)
            else:
                cur = getattr(cur, part, default)
        return cur

    def _configured_timeout_admin_ids(self) -> Set[str]:
        raw = self._config_get("timeout_manage_admin_ids", [])
        if isinstance(raw, str):
            parts = [x.strip() for x in raw.replace('|', ',').split(',')]
            return {x for x in parts if x}
        if isinstance(raw, (list, tuple, set)):
            return {str(x).strip() for x in raw if str(x).strip()}
        return set()

    def _event_identity_candidates(self, event: Any) -> Set[str]:
        keys = [
            "user_id", "sender_id", "uid", "from_id", "qq", "uin",
            "sender.user_id", "sender.sender_id", "sender.uid", "sender.id",
            "sender.qq", "sender.uin", "message_obj.sender.user_id", "message_obj.sender.id",
        ]
        ids: Set[str] = set()
        for key in keys:
            value = self._deep_get_attr(event, key, None)
            if value is None:
                continue
            s = str(value).strip()
            if s:
                ids.add(s)
        return ids

    def _is_explicit_admin_context(self, event: Any) -> bool:
        bool_paths = [
            "is_admin", "is_owner", "is_master", "is_super_admin", "is_superuser",
            "sender.is_admin", "sender.is_owner", "sender.is_master", "sender.is_super_admin",
            "message_obj.sender.is_admin", "message_obj.sender.is_owner", "message_obj.sender.is_super_admin",
        ]
        for path in bool_paths:
            parsed = self._parse_bool_like(self._deep_get_attr(event, path, None))
            if parsed is True:
                return True

        role_paths = ["role", "sender.role", "sender.group_role", "message_obj.sender.role", "message_obj.sender.group_role"]
        for path in role_paths:
            role = self._deep_get_attr(event, path, None)
            if role is None:
                continue
            role_s = str(role).strip().lower()
            if role_s in {"admin", "owner", "superadmin", "super_admin", "superuser", "master"}:
                return True
        return False

    def _has_timeout_manage_permission(self, event: Any) -> Tuple[bool, str]:
        if self._is_explicit_admin_context(event):
            return True, "admin_context"

        configured_admin_ids = self._configured_timeout_admin_ids()
        if configured_admin_ids:
            identities = self._event_identity_candidates(event)
            if identities & configured_admin_ids:
                return True, "configured_admin_id"
            return False, (
                "拒绝修改：当前上下文未识别为管理员，且调用者不在 timeout_manage_admin_ids 白名单中。"
            )

        return False, (
            "拒绝修改：当前上下文未识别到明确管理员权限。"
            "请在管理员会话中执行，或在插件配置 timeout_manage_admin_ids 中显式配置允许的用户ID。"
        )

    def _build_cache_extra_sig(self, resolved_max_models: int) -> str:
        selected_models = load_slot_selected_models(self._config_get)
        selected_fingerprint = [
            {
                "slot": str(m.get("selected_from_slot", "") or ""),
                "id": str(m.get("id", "") or ""),
                "tags": sorted(str(t) for t in (m.get("tags", []) or [])),
            }
            for m in selected_models
        ]
        runtime_ids = sorted(str(m.get("id", "") or "") for m in (self.models or []))
        payload = {
            "v": 2,
            "max_models": int(resolved_max_models),
            "default_participant_count": int(self._default_participant_count()),
            "selected_slots": selected_fingerprint,
            "runtime_model_ids": runtime_ids,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8", errors="ignore")).hexdigest()[:24]

    def _enable_concurrency(self) -> bool:
        """向后兼容：并发开关已废弃，当前实现始终全并发。"""
        raw = self._config_get("enable_concurrency", True)
        try:
            return bool(raw)
        except Exception:
            return True

    def _max_concurrency(self) -> int:
        """向后兼容：并发上限已废弃，仅回显配置值用于状态展示。"""
        raw = self._config_get("max_concurrency", MODEL_SLOT_COUNT)
        try:
            return max(1, int(raw))
        except Exception:
            return MODEL_SLOT_COUNT

    # -----------------------------------------------------------------------
    # Delegating helpers
    # -----------------------------------------------------------------------

    def _find_selected_provider_ids(self) -> List[str]:
        return find_selected_provider_ids(self._config_get)

    async def _calc_with_backend(self, task: str, mode: str, max_models: int, backend: str, context_summary: str = "") -> Dict[str, Any]:
        await self._ensure_health_loaded()

        selected_models = load_slot_selected_models(self._config_get)
        routed_ids, routing_meta = select_provider_ids_for_task(
            models=selected_models,
            task=task,
            mode=mode,
            requested_max_models=max_models,
            health_dict=self._provider_health,
        )

        result = await calc_with_backend(
            context=self.context, task=task, mode=mode, max_models=max_models, backend=backend,
            context_summary=context_summary, timeout_sec=self._real_call_timeout_sec(),
            per_model_timeout_sec=self._real_call_timeout_sec(),
            default_count=self._default_participant_count(), default_mode=self._default_mode(),
            models=self.models, provider_health_dict=self._provider_health,
            selected_provider_ids=routed_ids,
            return_merged=self._return_merged_answer(), return_candidates_flag=self._return_candidates(),
            enable_internal_llm_synthesis=False,
        )
        result["routing"] = routing_meta
        return result

    def _mock_compute(self, task: str, context_summary: str = "", mode: str = "balanced", max_models: int = 0) -> Dict[str, Any]:
        return mock_compute(
            task=task, context_summary=context_summary, mode=mode, max_models=max_models,
            models=self.models, default_mode=self._default_mode(), default_count=self._default_participant_count(),
            return_merged=self._return_merged_answer(), return_candidates_flag=self._return_candidates(),
        )

    async def _probe_real_call(self, provider_id: str, task: str) -> Dict[str, Any]:
        return await probe_real_call(self.context, provider_id, task, self._real_call_timeout_sec())

    # -----------------------------------------------------------------------
    # Status / parse helpers
    # -----------------------------------------------------------------------

    def _build_status_payload(self, include_models: bool = True) -> Dict[str, Any]:
        selected_provider_ids = self._find_selected_provider_ids()
        slots = build_slot_status(self._config_get)
        status: Dict[str, Any] = {
            "plugin": "astrbot_plugin_multi_model_compute",
            "version": PLUGIN_VERSION,
            "positioning": {
                "primary_entry": "llm_tool",
                "secondary_entry": "slash_commands_for_debug",
                "web_chat_note": "Web 聊天场景建议由默认模型调用 llm_tool，不建议依赖 /mmcalc 等命令。",
            },
            "supported": {"modes": sorted(VALID_MODES), "backends": sorted(VALID_BACKENDS)},
            "runtime": {
                "default_mode": self._default_mode(),
                "default_participant_count": "auto_by_task_and_tags",
                "return_merged_answer": self._return_merged_answer(),
                "return_candidates": self._return_candidates(),
                "real_call_timeout_sec": self._real_call_timeout_sec(),
                "cache_ttl_sec": self._cache_ttl_sec(),
                "consensus_timeout_retry": True,
            },
            "cache": self._cache.stats(),
            "provider_health_count": len(self._provider_health),
            "slots": slots,
            "selected_provider_ids": selected_provider_ids,
            "backend_capability": {
                "can_real_call": len(selected_provider_ids) > 0,
                "real_ready_provider_count": len(selected_provider_ids),
                "mock_available": True,
                "auto_strategy": "优先 real，若未配置槽位或 real 全失败则回退 mock。",
            },
            "tool_guide": {
                "recommended_call_order": ["multi_model_status", "model_capabilities", "multi_model_compute"],
                "when_to_call_multi_model_compute": [
                    "需要多模型交叉验证、对比、共识汇总",
                    "需要在 real 不稳定时自动回退 mock 保持可用性",
                    "需要结构化 per_model_results 便于主模型二次决策",
                ],
                "routing_policy": "按任务特征 + 模型标签 + provider_health 自动决定调用哪些模型以及调用数量",
            },
        }
        if include_models:
            status["runtime_models"] = self.models
        return status

    def _parse_mmcalc_args(self, body: str) -> Tuple[str, int, str, str]:
        mode = self._default_mode()
        max_models = self._default_participant_count()
        backend = "auto"
        task = body or ""

        for candidate_mode in VALID_MODES:
            for token in (f"--mode {candidate_mode}", f"--mode={candidate_mode}"):
                if token in task:
                    mode = candidate_mode
                    task = task.replace(token, "").strip()

        for candidate_backend in VALID_BACKENDS:
            for token in (f"--backend {candidate_backend}", f"--backend={candidate_backend}"):
                if token in task:
                    backend = candidate_backend
                    task = task.replace(token, "").strip()

        match = re.search(r"--max(?:=|\s+)(\d+)", task)
        if match:
            try:
                max_models = int(match.group(1))
            except Exception:
                pass
            task = re.sub(r"--max(?:=|\s+)\d+", "", task).strip()

        if backend not in VALID_BACKENDS:
            backend = "auto"
        if mode not in VALID_MODES:
            mode = self._default_mode()
        return task, max_models, mode, backend

    # =======================================================================
    # Command Handlers (调试入口)
    # =======================================================================

    @filter.command("mmhelp")
    async def mm_help(self, event: AstrMessageEvent):
        '''查看多模型计算插件的使用帮助'''
        lines = [
            f"多模型计算插件 {PLUGIN_VERSION}：",
            "/mmhelp - 查看帮助",
            "/mmstatus - 查看插件状态与当前可用模型",
            "/mmodels - 查看模型能力清单",
            "/mmprobe <任务> - 对第一个已选模型做一次真实调用 probe",
            "/mmcalc <任务> [--mode ...] [--max N] [--backend ...]",
            "/mmtest [任务] [--backend ...] - 快速检查 backend",
            "/mmtimeout - 查看/修改各配置文件的工具调用超时（管理员）",
            "/深度思考 <问题> [--mode ...] [--max N] - 🧠 多模型深度分析",
            "/辩论 <问题> [--max N] - ⚔️ 多轮模型辩论提炼共识",
            "说明：/深度思考 与 /辩论 是面向用户的功能入口。",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("mmtimeout")
    async def mm_timeout(self, event: AstrMessageEvent):
        '''查看并选择性修改各 AstrBot 配置文件的工具调用超时（需管理员权限）'''
        # ---------- 收集所有配置 ----------
        try:
            cfg_mgr = self.context.astrbot_config_mgr
            all_confs: dict = cfg_mgr.confs  # {"default": cfg, uuid: cfg, ...}
        except Exception as e:
            yield event.plain_result(f"❌ 无法读取配置管理器：{e}")
            return

        # 将 confs 转为有序列表，方便用编号引用
        conf_entries: List[Tuple[str, Any]] = list(all_confs.items())  # [(id, cfg), ...]

        # ---------- 解析指令参数 ----------
        message = event.message_str.strip()
        arg = ""
        for prefix in ("/mmtimeout", "mmtimeout"):
            if message.startswith(prefix):
                arg = message[len(prefix):].strip()
                break

        # ---------- 无参数：列出所有配置 ----------
        if not arg:
            lines = [
                "⏱ 工具调用超时设置（tool_call_timeout）",
                "─" * 35,
            ]
            for idx, (conf_id, cfg) in enumerate(conf_entries, start=1):
                ps = cfg.get("provider_settings", {}) or {}
                cur = ps.get("tool_call_timeout", 120)
                # 尝试从 ConfigManager 拿可读名称
                try:
                    conf_list = cfg_mgr.get_conf_list()
                    name = next(
                        (c["name"] for c in conf_list if c["id"] == conf_id),
                        conf_id,
                    )
                except Exception:
                    name = conf_id
                status_icon = "✅" if int(cur) >= 300 else "⚠️"
                lines.append(f"{idx}. [{name}] {status_icon} {cur}s")
            lines += [
                "─" * 35,
                "用法：",
                "  /mmtimeout <编号>   - 将指定配置提升到 300s",
                "  /mmtimeout all      - 更新全部配置到 300s",
                "  /mmtimeout <编号> <秒数> - 设置为自定义秒数",
                "  /mmtimeout all <秒数>    - 全部设置为自定义秒数",
            ]
            yield event.plain_result("\n".join(lines))
            return

        # ---------- 解析 target 与可选自定义秒数 ----------
        parts = arg.split()
        target_raw = parts[0].lower()
        custom_timeout: int | None = None
        if len(parts) >= 2:
            try:
                custom_timeout = max(1, int(parts[1]))
            except ValueError:
                yield event.plain_result(f"❌ 无效的秒数 '{parts[1]}'，请输入正整数。")
                return
        new_timeout = custom_timeout if custom_timeout is not None else 300

        permitted, reason = self._has_timeout_manage_permission(event)
        if not permitted:
            yield event.plain_result(f"❌ {reason}")
            return

        # ---------- 确定要修改的配置列表 ----------
        targets: List[Tuple[str, Any]] = []
        if target_raw == "all":
            targets = conf_entries
        else:
            try:
                idx = int(target_raw)
                if idx < 1 or idx > len(conf_entries):
                    raise ValueError()
                targets = [conf_entries[idx - 1]]
            except ValueError:
                yield event.plain_result(
                    f"❌ 无效参数 '{target_raw}'。\n"
                    f"请输入 1~{len(conf_entries)} 之间的编号，或 'all'。"
                )
                return

        # ---------- 执行修改 ----------
        result_lines = [f"⏱ 设置 tool_call_timeout → {new_timeout}s"]
        for conf_id, cfg in targets:
            try:
                ps = cfg.get("provider_settings", {}) or {}
                old = int(ps.get("tool_call_timeout", 120))
                ps["tool_call_timeout"] = new_timeout
                cfg["provider_settings"] = ps
                cfg.save_config()
                try:
                    conf_list = cfg_mgr.get_conf_list()
                    name = next(
                        (c["name"] for c in conf_list if c["id"] == conf_id),
                        conf_id,
                    )
                except Exception:
                    name = conf_id
                result_lines.append(f"  ✅ [{name}]  {old}s → {new_timeout}s")
                logger.info(
                    f"[multi_model_compute] [{conf_id}] tool_call_timeout "
                    f"{old}s → {new_timeout}s（由用户指令触发）"
                )
            except Exception as e:
                result_lines.append(f"  ❌ [{conf_id}] 修改失败：{e}")
        result_lines.append("\n修改已写入配置文件，立即生效。")
        yield event.plain_result("\n".join(result_lines))

    @filter.command("mmstatus")
    async def mm_status(self, event: AstrMessageEvent):
        '''查看插件状态和模型池'''
        slot_count = len(load_slot_selected_models(self._config_get))
        selected_ids = self._find_selected_provider_ids()
        cache_stats = self._cache.stats()
        lines = [
            f"多模型计算插件状态 ({PLUGIN_VERSION})：",
            f"- 默认模式: {self._default_mode()}",
            f"- 模型选择策略: 按任务特征 + 标签自动决定",
            f"- 已选择槽位模型数: {slot_count}",
            f"- 当前运行模型池数量: {len(self.models)}",
            f"- 已选 provider_ids: {', '.join(selected_ids) if selected_ids else '无'}",
            f"- real 调用超时(秒): {self._real_call_timeout_sec()}",
            f"- 缓存: TTL={cache_stats['ttl_sec']}s, 活跃={cache_stats['alive_entries']}, 命中={cache_stats['total_hits']}",
            f"- Provider Health: {len(self._provider_health)} 条记录",
            "当前模型池：",
        ]
        for m in self.models:
            lines.append(render_model_line(m))
        yield event.plain_result("\n".join(lines))

    @filter.command("mmodels")
    async def list_models(self, event: AstrMessageEvent):
        '''查看当前可用模型能力清单'''
        lines = ["当前可用模型能力清单："]
        for m in self.models:
            lines.append(render_model_line(m))
        yield event.plain_result("\n".join(lines))

    @filter.command("mmprobe")
    async def mm_probe(self, event: AstrMessageEvent):
        '''对第一个已选模型做一次真实调用探测'''
        message = event.message_str.strip()
        task = message[len("/mmprobe"):].strip() if message.startswith("/mmprobe") else ""
        if not task:
            yield event.plain_result("用法：/mmprobe <任务内容>")
            return

        selected_ids = self._find_selected_provider_ids()
        if not selected_ids:
            mock = self._mock_compute(task=task, mode=self._default_mode(), max_models=1)
            yield event.plain_result(f"mmprobe: 未配置模型槽位，已 fallback 到 mock。\nmock 结果: {mock.get('merged_answer', '')}")
            return

        real = await self._probe_real_call(selected_ids[0], task)
        if real.get("ok"):
            text = real.get("text", "")
            if len(text) > 400:
                text = text[:400] + "..."
            yield event.plain_result(
                f"mmprobe: real 调用成功\nprovider: {real.get('provider_id')}\n"
                f"elapsed: {real.get('elapsed_ms', 0)}ms\n摘要: {text}"
            )
        else:
            err = real.get("error", "unknown error")
            mock = self._mock_compute(task=task, mode=self._default_mode(), max_models=1)
            yield event.plain_result(
                f"mmprobe: real 调用失败，已 fallback 到 mock\nprovider: {selected_ids[0]}\n"
                f"错误: {err}\nmock 结果: {mock.get('merged_answer', '')}"
            )

    @filter.command("mmcalc")
    async def multi_model_calc_cmd(self, event: AstrMessageEvent):
        '''执行多模型计算命令（调试入口）'''
        message = event.message_str.strip()
        body = message[len("/mmcalc"):].strip() if message.startswith("/mmcalc") else ""
        if not body:
            yield event.plain_result("用法：/mmcalc <任务> [--mode ...] [--max N] [--backend auto|real|mock]")
            return

        task, max_models, mode, backend = self._parse_mmcalc_args(body)
        if not task:
            yield event.plain_result("请提供任务内容。")
            return

        yield event.plain_result(f"🔄 正在调用多模型计算（mode={mode}, backend={backend}）...")

        try:
            result = await self._calc_with_backend(task=task, mode=mode, max_models=max_models, backend=backend)
        except Exception as e:
            yield event.plain_result(f"❌ 多模型计算失败（backend={backend}）：{e}")
            return

        lines = [
            "✅ 多模型计算完成：",
            f"模式: {result.get('mode', mode)} | 后端: {result.get('backend', backend)}",
            f"参与模型: {', '.join(result.get('participants', [])) or '无'}",
        ]
        if result.get("fallback_reason"):
            lines.append(f"⚠️ fallback: {result['fallback_reason']}")
        for item in result.get("per_model_results", []):
            if item.get("ok"):
                text_preview = str(item.get("text", "")).replace("\n", " ")[:140]
                lines.append(f"✓ {item.get('provider_id')}: {item.get('elapsed_ms', 0)}ms | {text_preview}")
            else:
                lines.append(f"✗ {item.get('provider_id')}: {item.get('error', '')}")
        if "merged_answer" in result:
            lines.append(f"📝 聚合: {result['merged_answer']}")
        yield event.plain_result("\n".join(lines))

    @filter.command("mmtest")
    async def mm_test(self, event: AstrMessageEvent):
        '''快速检查 backend 状态与 fallback'''
        message = event.message_str.strip()
        body = message[len("/mmtest"):].strip() if message.startswith("/mmtest") else ""

        backend = "auto"
        for cb in VALID_BACKENDS:
            for token in (f"--backend {cb}", f"--backend={cb}"):
                if token in body:
                    backend = cb
                    body = body.replace(token, "").strip()

        task = body.strip() or "请回复:mmtest ok"
        try:
            result = await self._calc_with_backend(task=task, mode=self._default_mode(), max_models=1, backend=backend)
        except Exception as e:
            yield event.plain_result(f"mmtest 失败\nbackend: {backend}\nerror: {e}")
            return

        per_results = result.get("per_model_results", [])
        lines = [
            "mmtest 结果：",
            f"- backend: {backend} -> {result.get('backend', backend)}",
            f"- 已选模型数: {len(self._find_selected_provider_ids())}",
            f"- real probe 可用: {any(item.get('ok') for item in per_results)}",
            f"- fallback: {bool(result.get('fallback_from'))}",
        ]
        if result.get("fallback_reason"):
            lines.append(f"- fallback 原因: {result.get('fallback_reason')}")
        yield event.plain_result("\n".join(lines))

    # =======================================================================
    # 用户功能入口：深度思考
    # =======================================================================

    @filter.command("深度思考")
    async def deep_think(self, event: AstrMessageEvent):
        """🧠 多模型深度分析：收集多个模型观点，并调用 AstrBot 默认对话模型归纳输出。"""
        message = event.message_str.strip()
        # 兼容多种触发格式
        for prefix in ("/深度思考", "深度思考"):
            if message.startswith(prefix):
                message = message[len(prefix):].strip()
                break

        if not message:
            yield event.plain_result(
                "🧠 深度思考 — 多模型深度分析\n\n"
                "用法：/深度思考 <你的问题>\n"
                "可选参数：\n"
                "  --mode balanced|fast|consensus|creative\n"
                "  --max N（参与模型数）\n\n"
                "示例：/深度思考 如何设计一个高并发系统 --mode consensus"
            )
            return

        task, max_models, mode, _ = self._parse_mmcalc_args(message)
        if not task:
            yield event.plain_result("请提供要深度思考的问题。")
            return

        # ---- Step 1: 告知用户开始 ----
        selected_ids = self._find_selected_provider_ids()
        model_count = min(max_models, len(selected_ids)) if selected_ids else max_models
        yield event.plain_result(
            f"🧠 开始深度思考...\n"
            f"📋 问题: {task[:80]}{'...' if len(task) > 80 else ''}\n"
            f"⚙️ 模式: {mode} | 计划调用 {model_count} 个模型"
        )

        # ---- Step 2: 执行多模型调用 ----
        try:
            result = await self._calc_with_backend(
                task=task, mode=mode, max_models=max_models, backend="auto",
            )
        except Exception as e:
            yield event.plain_result(f"❌ 深度思考执行失败: {e}")
            return

        # ---- Step 3: 构建 synthesis ----
        from .synthesis import build_synthesis_material, derive_confidence, derive_recommendation
        synthesis = build_synthesis_material(result)
        result["synthesis"] = synthesis
        result["confidence"] = derive_confidence(result)
        result["recommendation"] = derive_recommendation(result)
        result["mode"] = mode
        result["run_id"] = f"deep-{now_iso_utc()[:19].replace(':', '').replace('-', '')}"

        core = safe_dict(synthesis, "core")
        success_count = int(core.get("success_count", 0))
        failure_count = int(core.get("failure_count", 0))
        common_count = len(safe_list(core, "common_points"))

        yield event.plain_result(
            f"🔍 分析完成！成功 {success_count} 个模型，失败 {failure_count} 个\n"
            f"✅ 发现 {common_count} 个共识要点\n"
            f"🧠 正在调用默认对话模型归纳最终答案..."
        )

        # ---- Step 4: 调用宿主默认对话模型做最终归纳（不再走可视化报告） ----
        per_model_lines = []
        for item in safe_list(result, "per_model_results")[:8]:
            pid = str(item.get("provider_id", "") or "unknown")
            if item.get("ok"):
                per_model_lines.append(f"- [{pid}] {str(item.get('text', '') or '')[:1200]}")
            else:
                per_model_lines.append(f"- [{pid}] (failed) {str(item.get('error', '') or '')[:200]}")

        common_points = [cp.get("point", "") for cp in safe_list(core, "common_points")[:8] if cp.get("point")]
        diff_points = [dp.get("position", "") for dp in safe_list(core, "differences")[:6] if dp.get("position")]

        summarize_prompt = (
            "你是 AstrBot 的默认对话模型。请基于以下多模型材料，直接输出给用户的最终回答。\n"
            "要求：\n"
            "1) 直接给出结论与可执行建议；\n"
            "2) 不要输出可视化报告、HTML、图片描述或模板；\n"
            "3) 若材料冲突，请说明你采纳哪种观点及理由；\n"
            "4) 保持自然语言、结构清晰。\n\n"
            f"原始问题：{task}\n"
            f"执行模式：{mode}\n\n"
            f"共识要点：\n{chr(10).join(f'- {x}' for x in common_points) if common_points else '- (无明确共识)'}\n\n"
            f"分歧要点：\n{chr(10).join(f'- {x}' for x in diff_points) if diff_points else '- (无明显分歧)'}\n\n"
            f"各模型原始输出摘录：\n{chr(10).join(per_model_lines) if per_model_lines else '- (无可用材料)'}"
        )

        summarize_prompt += (
            "\n\n输出要求：请仅输出纯文本，不要使用 Markdown 标题（#）、加粗（**/__）、"
            "列表强调符号等格式标记。"
        )
        final_answer = await self._call_default_chat_model(event, summarize_prompt)

        if final_answer:
            final_answer = self._sanitize_plain_text_answer(final_answer)
            result["final_answer"] = final_answer
            result["answer"] = final_answer
            yield event.plain_result(final_answer)
            return

        # ---- Step 5: 默认模型不可用时的稳妥文本回退（不生成可视化报告） ----
        logger.warning("[deep_think] default chat model summarize unavailable, fallback to text synthesis")
        fallback_answer = str(synthesis.get("draft_synthesis", "") or "").strip()
        if not fallback_answer:
            fallback_answer = str(result.get("merged_answer", "") or "").strip()
        if not fallback_answer:
            fallback_answer = str(result.get("recommendation", "") or "").strip()
        if not fallback_answer:
            fallback_answer = self._build_text_report(result)

        fallback_answer = self._sanitize_plain_text_answer(fallback_answer)
        yield event.plain_result(fallback_answer)

    def _build_text_report(self, result: Dict[str, Any]) -> str:
        """当 html_render 不可用时的纯文本降级报告。"""
        synthesis = safe_dict(result, "synthesis")
        core = safe_dict(synthesis, "core")
        lines = [
            "═" * 30,
            "🤖 多模型汇总报告（文本版）",
            "═" * 30,
            f"状态: {result.get('status', 'unknown')} | 置信度: {result.get('confidence', 0):.0%}",
            f"模式: {result.get('mode', 'balanced')} | 后端: {result.get('backend', 'unknown')}",
            "",
        ]

        common_points = safe_list(core, "common_points")
        if common_points:
            lines.append("✅ 共识要点：")
            for cp in common_points[:6]:
                lines.append(f"  • {cp.get('point', '')} ({cp.get('support_count', 0)} 模型共识)")
            lines.append("")

        notable = safe_list(core, "notable_insights")
        if notable:
            lines.append("💡 独到见解：")
            for ni in notable[:4]:
                providers = ", ".join(ni.get("support_providers", []))
                lines.append(f"  • {ni.get('point', '')} [{providers}]")
            lines.append("")

        conflicts = safe_list(core, "conflict_points")
        if conflicts:
            lines.append("⚡ 冲突点：")
            for cp in conflicts[:3]:
                lines.append(f"  • {cp.get('topic', '')} ({cp.get('conflict_level', 'low')})")
            lines.append("")

        failures = safe_list(synthesis, "failures")
        if failures:
            lines.append("❌ 失败：")
            for f in failures[:3]:
                lines.append(f"  • {f.get('provider_id', '')}: {str(f.get('error', ''))[:60]}")
            lines.append("")

        lines.append("═" * 30)
        return "\n".join(lines)

    # =======================================================================
    # 用户功能入口：辩论模式 (Debate Protocol)
    # =======================================================================

    @filter.command("辩论")
    async def debate_cmd(self, event: AstrMessageEvent):
        """⚔️ 多轮模型辩论：组织多个模型进行多轮交叉辩论，并由 AstrBot 默认对话模型做最终归纳。"""
        message = event.message_str.strip()
        for prefix in ("/辩论", "辩论"):
            if message.startswith(prefix):
                message = message[len(prefix):].strip()
                break

        if not message:
            yield event.plain_result(
                "⚔️ 辩论模式 — Debate Protocol\n\n"
                "用法：/辩论 <问题>\n"
                "可选：--max N\n\n"
                "组织多个模型进行交叉评论，最终由默认对话模型收敛输出。"
            )
            return

        task, max_models, _, _ = self._parse_mmcalc_args(message)
        if not task:
            yield event.plain_result("请提供辩论主题。")
            return

        selected_ids = self._find_selected_provider_ids()
        # 辩论必须要有至少两个模型
        participants = selected_ids[:max_models]
        if len(participants) < 2:
            yield event.plain_result("⚠️ 辩论模式需要配置并选择至少 2 个模型。")
            return

        yield event.plain_result(f"⚔️ 开启辩论模式\n主题: {task[:60]}\n参与者: {', '.join(participants)}")

        async def _call_fn(pid: str, prompt: str) -> Dict[str, Any]:
            return await self._probe_real_call(pid, prompt)

        async def _progress(msg: str):
            await event.send(event.plain_result(msg))

        engine = DebateEngine(
            question=task,
            provider_ids=participants,
            call_fn=_call_fn,
            max_rounds=2,
            on_progress=_progress
        )

        try:
            result = await engine.run()
        except Exception as e:
            yield event.plain_result(f"❌ 辩论执行失败: {e}")
            return

        yield event.plain_result("✅ 辩论结束，正在调用默认对话模型归纳最终结论...")

        rounds = safe_list(result, "rounds")
        round_lines = []
        for r in rounds[-3:]:
            rid = r.get("round", "")
            views = safe_dict(r, "views")
            for pid, v in views.items():
                role = str(v.get("label") or v.get("role") or "")
                txt = str(v.get("text", "") or "")
                round_lines.append(f"- R{rid} [{pid}/{role}] {txt[:800]}")

        synth = safe_dict(result, "synthesis")
        synth_core = safe_dict(synth, "core")
        common_points = [cp.get("point", "") for cp in safe_list(synth_core, "common_points")[:8] if cp.get("point")]
        conflict_points = [cp.get("point", "") for cp in safe_list(synth_core, "conflict_points")[:6] if cp.get("point")]

        summarize_prompt = (
            "你是 AstrBot 的默认对话模型。请基于以下辩论材料给出最终结论。\n"
            "要求：\n"
            "1) 输出面向用户的自然语言结论；\n"
            "2) 先给核心结论，再给关键理由和可执行建议；\n"
            "3) 不要输出可视化报告、HTML、图片模板内容；\n"
            "4) 对分歧点给出你的取舍与依据。\n\n"
            f"辩论题目：{task}\n"
            f"参与模型：{', '.join(safe_list(result, 'participants'))}\n"
            f"辩论轮次：{result.get('total_rounds', 0)}\n\n"
            f"辩论内部综合答案（供参考）：{str(result.get('final_answer', '') or '')[:2000]}\n\n"
            f"共识点：\n{chr(10).join(f'- {x}' for x in common_points) if common_points else '- (无明确共识)'}\n\n"
            f"冲突点：\n{chr(10).join(f'- {x}' for x in conflict_points) if conflict_points else '- (无明显冲突点)'}\n\n"
            f"各轮观点摘录：\n{chr(10).join(round_lines) if round_lines else '- (无可用辩论摘录)'}"
        )

        summarize_prompt += (
            "\n\n输出要求：请仅输出纯文本，不要使用 Markdown 标题（#）、加粗（**/__）、"
            "列表强调符号等格式标记。"
        )
        final_answer = await self._call_default_chat_model(event, summarize_prompt)
        if final_answer:
            final_answer = self._sanitize_plain_text_answer(final_answer)
            result["final_answer"] = final_answer
            yield event.plain_result(final_answer)
            return

        logger.warning("[debate] default chat model summarize unavailable, fallback to textual synthesis")
        fallback_answer = str(result.get("final_answer", "") or "").strip()
        if fallback_answer:
            fallback_answer = self._sanitize_plain_text_answer(fallback_answer)
            yield event.plain_result(fallback_answer)
            return

        summary = []
        for cp in common_points:
            summary.append(f"- {cp}")
        if summary:
            yield event.plain_result("📌 最终共识:\n" + "\n".join(summary))
        else:
            yield event.plain_result("暂无达成强共识的观点。")


    # =======================================================================
    # LLM Tool Handlers
    # =======================================================================

    @filter.llm_tool(name="manage_timeout")
    async def manage_timeout(
        self,
        event: AstrMessageEvent,
        action: str = "list",
        target: str = "all",
        seconds: int = 300,
    ):
        """查询或修改 AstrBot 各配置文件的工具调用超时（tool_call_timeout）。

        当用户在网页对话中询问超时设置、或要求修改超时时调用此工具。

        Args:
            action(string): 操作类型。"list" = 仅查询当前值；"set" = 修改超时值。
            target(string): 目标配置。"all" = 全部配置；配置名称或编号（如 "1"、"default"、"工作群配置"）= 指定配置。
            seconds(number): 目标秒数，仅 action="set" 时有效，默认 300。
        """
        try:
            cfg_mgr = self.context.astrbot_config_mgr
            all_confs: dict = cfg_mgr.confs
        except Exception as e:
            return {"status": "error", "message": f"无法读取配置管理器: {e}"}

        # 构建带编号的配置信息列表
        try:
            conf_list_meta = cfg_mgr.get_conf_list()
        except Exception:
            conf_list_meta = []

        def _get_name(conf_id: str) -> str:
            for c in conf_list_meta:
                if c["id"] == conf_id:
                    return c["name"]
            return conf_id

        conf_entries: List[Tuple[str, Any]] = list(all_confs.items())
        action_norm = str(action or "").strip().lower()

        # ---- 构建各配置快照 ----
        snapshots = []
        for idx, (conf_id, cfg) in enumerate(conf_entries, start=1):
            ps = cfg.get("provider_settings", {}) or {}
            cur = int(ps.get("tool_call_timeout", 120))
            snapshots.append({
                "index": idx,
                "id": conf_id,
                "name": _get_name(conf_id),
                "current_timeout_sec": cur,
                "needs_update": cur < 300,
            })

        if action_norm == "list":
            return {
                "status": "ok",
                "action": "list",
                "configs": snapshots,
                "hint": (
                    "⚠️ 标记 needs_update=true 的配置超时不足 300s，可能导致多模型工具调用被截断。"
                    "如需修改，请告知目标配置编号（或 all）及期望秒数。"
                ),
            }

        if action_norm != "set":
            return {
                "status": "error",
                "message": f"unsupported action: {action}. expected 'list' or 'set'",
                "configs": snapshots,
            }

        permitted, reason = self._has_timeout_manage_permission(event)
        if not permitted:
            return {
                "status": "forbidden",
                "action": "set",
                "message": reason,
            }

        # ---- action == "set" ----
        new_timeout = max(1, int(seconds))

        # 确定目标配置
        target_str = str(target).strip().lower()
        if target_str == "all":
            targets = conf_entries
        else:
            matched = None
            # 先尝试按编号
            try:
                idx = int(target_str)
                if 1 <= idx <= len(conf_entries):
                    matched = [conf_entries[idx - 1]]
            except ValueError:
                pass
            # 再尝试按 id 或名称匹配（模糊）
            if matched is None:
                for conf_id, cfg in conf_entries:
                    name = _get_name(conf_id).lower()
                    if target_str in conf_id.lower() or target_str in name:
                        matched = [(conf_id, cfg)]
                        break
            if matched is None:
                return {
                    "status": "error",
                    "message": (
                        f"未找到匹配的配置 '{target}'。"
                        f"可用编号：1~{len(conf_entries)}，或使用 'all'。"
                    ),
                    "configs": snapshots,
                }
            targets = matched

        results = []
        for conf_id, cfg in targets:
            try:
                ps = cfg.get("provider_settings", {}) or {}
                old = int(ps.get("tool_call_timeout", 120))
                ps["tool_call_timeout"] = new_timeout
                cfg["provider_settings"] = ps
                cfg.save_config()
                results.append({
                    "id": conf_id,
                    "name": _get_name(conf_id),
                    "old_sec": old,
                    "new_sec": new_timeout,
                    "success": True,
                })
                logger.info(
                    f"[multi_model_compute] [{conf_id}] tool_call_timeout "
                    f"{old}s → {new_timeout}s（由 LLM Tool 触发）"
                )
            except Exception as e:
                results.append({"id": conf_id, "name": _get_name(conf_id), "success": False, "error": str(e)})

        return {
            "status": "ok",
            "action": "set",
            "new_timeout_sec": new_timeout,
            "results": results,
            "message": f"已处理 {len(results)} 个配置，修改立即生效。",
        }

    @filter.llm_tool(name="multi_model_status")
    async def multi_model_status(self, event: AstrMessageEvent):
        """工具级自检入口，快速获取插件可用性、槽位配置与 backend 能力。"""
        payload = self._build_status_payload(include_models=False)
        payload["usage"] = "建议默认模型先调用 multi_model_status，再按需调用 model_capabilities / multi_model_compute。"
        payload["status"] = "ok"
        return payload

    @filter.llm_tool(name="model_capabilities")
    async def model_capabilities(self, event: AstrMessageEvent):
        """返回插件能力清单与 tool 调用规约。"""
        status = self._build_status_payload(include_models=True)
        return {
            "status": "ok",
            "plugin": "astrbot_plugin_multi_model_compute",
            "version": PLUGIN_VERSION,
            "positioning": "tool_first",
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "models": self.models,
            "selected_provider_ids": status.get("selected_provider_ids", []),
            "slots": status.get("slots", []),
            "supported_modes": sorted(VALID_MODES),
            "supported_backends": sorted(VALID_BACKENDS),
            "backend_capability": status.get("backend_capability", {}),
            "runtime": status.get("runtime", {}),
            "tool_contracts": {
                "multi_model_compute": {
                    "purpose": "执行多模型计算并返回结构化材料",
                    "params": {
                        "query": "string, 必填任务文本",
                        "mode": "balanced|fast|consensus|creative",
                        "max_models": "int, <=0 时用默认配置",
                        "backend": "auto|real|mock",
                        "detail_level": "brief|standard|full (default: brief)",
                    },
                },
            },
            "usage": "推荐顺序：multi_model_status -> model_capabilities -> multi_model_compute。",
        }

    @filter.llm_tool(name="multi_model_compute")
    async def multi_model_compute(
        self,
        event: AstrMessageEvent,
        query: str = "",
        context_summary: str = "",
        mode: str = "balanced",
        max_models: int = 0,
        backend: str = "auto",
        detail_level: str = "brief",
        project_goal: str = "",
        project_context: str = "",
        constraints: str = "",
        current_stage: str = "",
        current_request: str = "",
        expected_output: str = "",
        project_id: str = "",
        thread_id: str = "",
        topic_id: str = "",
    ):
        """Tool-first 多模型材料汇总入口。

        Args:
            query(string): 任务文本，为空会返回 invalid_request。
            context_summary(string): 可选上下文摘要。
            mode(string): 推理模式：balanced, fast, consensus, creative。
            max_models(number): 最大模型数，<=0 时用配置默认值。
            backend(string): 后端策略：auto, real, mock。
            detail_level(string): 返回粒度：brief(精简,默认), standard(标准), full(完整)。
            project_goal(string): 可选项目目标。
            project_context(string): 可选项目背景。
            constraints(string): 可选约束条件。
            current_stage(string): 可选当前阶段。
            current_request(string): 可选当前子任务。
            expected_output(string): 可选期望输出。
            project_id(string): 可选项目标识。
            thread_id(string): 可选会话标识。
            topic_id(string): 可选话题标识。
        """
        # ---- resolve params ----
        resolved_mode = (mode or "auto").strip().lower()
        if resolved_mode not in VALID_MODES:
            resolved_mode = self._default_mode()
        resolved_backend = (backend or "auto").strip().lower()
        if resolved_backend not in VALID_BACKENDS:
            resolved_backend = "auto"
        resolved_detail = (detail_level or "brief").strip().lower()
        if resolved_detail not in {"brief", "standard", "full"}:
            resolved_detail = "brief"
        try:
            resolved_max_models = int(max_models)
        except Exception:
            resolved_max_models = 0

        # ---- build task ----
        normalized_payload = normalize_project_payload(
            query=query, context_summary=context_summary,
            project_goal=project_goal, project_context=project_context,
            constraints=constraints, current_stage=current_stage,
            current_request=current_request, expected_output=expected_output,
            project_id=project_id, thread_id=thread_id, topic_id=topic_id,
        )
        task_meta = build_project_task_and_meta(normalized_payload, resolved_mode)
        task = str(task_meta.get("task", "") or "").strip()
        
        # P3-1: 动态路由
        if resolved_mode == "auto":
            resolved_mode = route_task_mode(task)

        if not task:
            invalid_result = {
                "status": "invalid_request",
                "error": {"code": "invalid_request", "message": "query/current_request is required"},
                "tool": "multi_model_compute",
                "tool_version": PLUGIN_VERSION,
                "next_action_hint": "Provide at least query or current_request.",
                "synthesis": default_synthesis(),
                "material_quality": default_material_quality(),
            }
            return ensure_schema_invariants(invalid_result)

        ids = build_ids(task, task_meta.get("project_identifiers", {}))

        # ---- cache check ----
        self._cache.ttl_sec = self._cache_ttl_sec()
        cache_extra_sig = self._build_cache_extra_sig(resolved_max_models)
        cached = self._cache.get(task, resolved_mode, resolved_backend, extra_sig=cache_extra_sig)
        if cached is not None:
            cached["cache_hit"] = True
            cached["run_id"] = ids["run_id"]
            cached["trace_id"] = ids["trace_id"]
            logger.info(f"[multi_model_compute] cache hit for run_id={ids['run_id']}")
            # 按 detail_level 裁剪缓存结果
            if resolved_detail == "brief":
                return self._filter_brief(cached)
            elif resolved_detail == "standard":
                return self._filter_standard(cached)
            return cached

        # ---- run backend ----
        try:
            result = await self._calc_with_backend(
                task=task, mode=resolved_mode, max_models=resolved_max_models,
                backend=resolved_backend, context_summary=normalized_payload.get("context_summary", ""),
            )
        except Exception as e:
            result = self._mock_compute(
                task=task, context_summary=normalized_payload.get("context_summary", ""),
                mode=resolved_mode, max_models=resolved_max_models,
            )
            result["backend"] = "mock"
            result["fallback_from"] = resolved_backend
            result["fallback_reason"] = str(e)
            result["status"] = "fallback_used"

        # ---- synthesis ----
        synthesis = build_synthesis_material(result)
        result["synthesis"] = synthesis
        synthesis_core = safe_dict(synthesis, "core")

        # ---- enrich result ----
        result["run_id"] = ids["run_id"]
        result["trace_id"] = ids["trace_id"]
        result["version"] = RESULT_SCHEMA_VERSION
        result["generated_at"] = now_iso_utc()
        result["tool"] = "multi_model_compute"
        result["tool_version"] = PLUGIN_VERSION

        # draft answer
        draft_answer = str(synthesis.get("draft_synthesis", "") or "").strip()
        if not draft_answer:
            draft_answer = str(result.get("merged_answer", "") or "").strip()
        if not draft_answer:
            candidates = result.get("candidates", [])
            if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                draft_answer = str(candidates[0].get("summary", "") or "").strip()
        result["draft_synthesis"] = draft_answer
        result["final_answer"] = draft_answer
        result["answer"] = draft_answer

        result["summary_source"] = derive_summary_source(result)
        result["confidence"] = derive_confidence(result)
        result["recommendation"] = derive_recommendation(result)

        # selection debug
        aggregation = safe_dict(result, "aggregation")
        result["selection_debug"] = {
            "selected_provider_id": aggregation.get("selected_provider_id", ""),
            "strategy": aggregation.get("strategy", ""),
            "note": "selection_debug is reference material, not final decision.",
        }

        # status derivation
        material_ready = bool(synthesis_core.get("material_ready", synthesis.get("material_ready")))
        all_failed = bool(synthesis_core.get("all_failed", synthesis.get("all_failed")))
        partial_success = bool(synthesis_core.get("partial_success", synthesis.get("partial_success")))
        fb_applied = _fallback_applied(result)

        execution_status = derive_execution_status(
            material_ready=material_ready, all_failed=all_failed,
            partial_success=partial_success, fallback_applied=fb_applied,
        )
        result["status"] = execution_status

        # next action hint
        hints = {
            "all_failed": "All providers failed; retry with mode=fast, max_models=2.",
            "no_material": "No usable synthesis material; narrow scope and retry.",
            "partial_success": "Use successful materials first, retry failed providers with compact prompt.",
            "fallback_used": "Fallback material in use; verify key assumptions.",
        }
        result["next_action_hint"] = hints.get(execution_status, "Proceed with synthesized material.")
        result["recommendation"] = result.get("recommendation", "")

        if all_failed:
            result["recommendation"] = "先恢复可用材料：缩短 current_request，使用 fast+2 models 重试。"
        elif partial_success:
            result["recommendation"] = "先基于已有 common_points 推进，并明确失败模型带来的不确定性。"
        elif execution_status == "fallback_used":
            result["recommendation"] = "当前材料含 fallback 成分，先推进低风险步骤，尽快触发 real backend 复算。"

        # project forward fields
        project_forward = derive_project_forward_fields(synthesis)
        result.update(project_forward)
        if "recommended_next_steps" not in result:
            result["recommended_next_steps"] = project_forward.get("recommended_next_steps", [])

        # mirror common fields to top level
        result["common_points"] = synthesis_core.get("common_points", [])
        result["differences"] = synthesis_core.get("differences", [])
        result["failures"] = synthesis.get("failures", [])
        try:
            _pm = result.get("per_model_results", []) or []
            _preview = []
            for _item in _pm[:8]:
                _preview.append({
                    "provider_id": _item.get("provider_id", ""),
                    "ok": bool(_item.get("ok", False)),
                    "elapsed_ms": int(_item.get("elapsed_ms", 0) or 0),
                    "text_preview": str(_item.get("text", "") or "")[:500],
                    "error": str(_item.get("error", "") or "")[:300],
                })
            logger.info(f"[multi_model_compute] per_model_results_preview={_preview}")
        except Exception:
            pass
        result["notable_insights"] = synthesis_core.get("notable_insights", [])
        result["conflict_level"] = synthesis_core.get("conflict_level", "low")
        result["material_quality"] = synthesis_core.get("material_quality", {})
        result["source_refs"] = synthesis_core.get("source_refs", [])
        result["evidence_refs"] = synthesis_core.get("evidence_refs", [])

        # synthesis brief
        result["synthesis_brief"] = {
            "material_ready": material_ready,
            "quality_level": result["material_quality"].get("level", "unknown"),
            "success_ratio": synthesis_core.get("success_ratio", 0),
            "common_points_count": len(result["common_points"]),
            "differences_count": len(result["differences"]),
            "failure_count": synthesis_core.get("failure_count", 0),
            "conflict_level": result["conflict_level"],
        }

        # consume guidance
        result["consume_guidance"] = {
            "role_boundary": "plugin_provides_material_only",
            "default_model_must_summarize": True,
            "preferred_core_fields": [
                "synthesis.core.common_points",
                "synthesis.core.differences",
                "synthesis.core.notable_insights",
                "synthesis.core.conflict_points",
            ],
            "do_not_directly_adopt": ["final_answer", "answer", "merged_answer"],
        }

        # request echo
        result["request"] = {
            "query": task, "mode": resolved_mode, "max_models": resolved_max_models,
            "backend": resolved_backend, "detail_level": resolved_detail,
        }

        # retry guidance
        ph = safe_dict(result, "provider_health")
        timeout_tendency_high = [pid for pid, h in ph.items() if isinstance(h, dict) and h.get("timeout_tendency") == "high"]
        result["retry_guidance"] = {
            "recommended_mode_on_timeout": "fast",
            "recommended_max_models_on_timeout": 2,
            "provider_timeout_tendency_high": timeout_tendency_high,
        }

        full_result = ensure_schema_invariants(result)
        full_result["cache_hit"] = False

        # P2-5: 记录 ReasoningBank 轨迹
        if execution_status in {"success", "partial_success"} and result.get("merged_answer"):
            rb = getattr(self, "_reasoning_bank", None)
            if rb is not None:
                agg = safe_dict(result, "aggregation")
                best_pid = agg.get("selected_provider_id", "")
                if best_pid:
                    key = hashlib.md5(task[:80].encode('utf-8')).hexdigest()
                    rb[key] = {
                        "task_prefix": task[:80],
                        "winning_provider": best_pid,
                        "mode": resolved_mode,
                        "elapsed_ms": result.get("timing", {}).get("total_elapsed_ms", 0),
                        "ts": time.time()
                    }
                    if len(rb) > 500: # 限制最多500条历史轨迹
                        oldest = min(rb, key=lambda k: rb[k]["ts"])
                        del rb[oldest]

        # ---- write cache ----
        self._cache.put(task, resolved_mode, resolved_backend, full_result, extra_sig=cache_extra_sig)

        # ---- detail_level filtering ----
        if resolved_detail == "brief":
            return self._filter_brief(full_result)
        elif resolved_detail == "standard":
            return self._filter_standard(full_result)
        return full_result

    # -----------------------------------------------------------------------
    # detail_level filters
    # -----------------------------------------------------------------------

    @staticmethod
    def _filter_brief(result: Dict[str, Any]) -> Dict[str, Any]:
        """~16 个键的精简返回，节省 LLM context window。"""
        return {
            "status": result.get("status"),
            "status_v2": result.get("status_v2"),
            "backend": result.get("backend"),
            "mode": result.get("mode"),
            "cache_hit": result.get("cache_hit", False),
            "synthesis_brief": result.get("synthesis_brief"),
            "recommendation": result.get("recommendation"),
            "confidence": result.get("confidence"),
            "common_points": result.get("common_points"),
            "differences": result.get("differences"),
            "notable_insights": result.get("notable_insights"),
            "conflict_level": result.get("conflict_level"),
            "consumer_hints": result.get("consumer_hints"),
            "next_action_hint": result.get("next_action_hint"),
            "run_id": result.get("run_id"),
            "tool": result.get("tool"),
            "tool_version": result.get("tool_version"),
            "routing": result.get("routing"),
        }

    @staticmethod
    def _filter_standard(result: Dict[str, Any]) -> Dict[str, Any]:
        """中等粒度返回：包含 synthesis.core 但不含诊断数据和原始材料。"""
        brief = MultiModelComputePlugin._filter_brief(result)
        synthesis = safe_dict(result, "synthesis")
        brief["synthesis_core"] = safe_dict(synthesis, "core")
        brief["failures"] = result.get("failures")
        brief["material_quality"] = result.get("material_quality")
        brief["fallback"] = result.get("fallback")
        brief["timing"] = result.get("timing")
        brief["retry_guidance"] = result.get("retry_guidance")
        brief["recommended_next_steps"] = result.get("recommended_next_steps")

        # 精简 per_model_results：仅保留摘要
        per_model = safe_list(result, "per_model_results")
        brief["per_model_summary"] = [
            {
                "provider_id": item.get("provider_id"),
                "ok": item.get("ok"),
                "elapsed_ms": item.get("elapsed_ms"),
                "text_length": len(str(item.get("text", "") or "")),
            }
            for item in per_model
            if isinstance(item, dict)
        ]
        return brief
