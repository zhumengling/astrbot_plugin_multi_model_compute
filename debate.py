"""多轮迭代共识引擎（Debate Protocol v2）。

架构重构要点（vs v1）：
  - [P1-1] 串行 for 循环 → asyncio.gather 全并行，每轮耗时从 N×60s → 60s
  - [P1-2] 角色分工 Prompt：advocate / critic / balancer，强制多样性输出
  - [P1-3] LLM 综合层：Round 3 不再靠 n-gram，而是单次 LLM 调用直接产出最终答案
  - [P2-2] 两级超时：Phase 级（每轮 70s）+ 整体预算（210s）
  - [P2-3] 渐进降级：成功 ≥ 2 → LLM 综合；= 1 → 直接返回；= 0 → fallback

适用于需要高置信度的场景（医疗、法律、关键决策）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from .utils import now_iso_utc, safe_dict, safe_list
from .synthesis import build_synthesis_material


# ---------------------------------------------------------------------------
# 角色分工 Prompt 模板（P1-2：借鉴 TradingAgents 的角色固定策略）
# ---------------------------------------------------------------------------

ROLE_DEFINITIONS = {
    "advocate": {
        "label": "支持视角",
        "system": (
            "你是一个支持性分析师。你的任务是从最有利、最正面的角度分析问题，"
            "找出最有力的支持论据，挖掘潜在价值和机会。"
            "不要刻意平衡，专注于支持性观点。"
        ),
    },
    "critic": {
        "label": "质疑视角",
        "system": (
            "你是一个批判性分析师。你的任务是从质疑、风险和挑战的角度分析问题，"
            "找出潜在问题、漏洞和反驳论据。"
            "不要刻意平衡，专注于批判性观点。"
        ),
    },
    "balancer": {
        "label": "平衡视角",
        "system": (
            "你是一个平衡性分析师。你的任务是从中立、实用的角度分析问题，"
            "评估各方观点，给出最具可操作性的建议和折中路径。"
            "专注于实际可行性和综合评估。"
        ),
    },
}

# 根据模型数量分配角色（循环分配）
_ROLE_ORDER = ["advocate", "critic", "balancer"]


def assign_roles(provider_ids: List[str]) -> Dict[str, str]:
    """为每个 provider 分配辩论角色（循环分配）。"""
    roles = {}
    for idx, pid in enumerate(provider_ids):
        roles[pid] = _ROLE_ORDER[idx % len(_ROLE_ORDER)]
    return roles


def build_role_prompt(question: str, role: str) -> str:
    """构建角色专用的 Round 1 Prompt。"""
    role_def = ROLE_DEFINITIONS.get(role, ROLE_DEFINITIONS["balancer"])
    return (
        f"{role_def['system']}\n\n"
        f"请用你的角色视角回答以下问题，给出详细分析和具体结论：\n\n"
        f"{question}"
    )


def build_cross_critique_prompt(
    question: str,
    own_role: str,
    own_answer: str,
    other_answers: Dict[str, Tuple[str, str]],  # {pid: (role, answer)}
) -> str:
    """Round 2：交叉评论 Prompt，传入完整上轮答案而非截断版。"""
    role_def = ROLE_DEFINITIONS.get(own_role, ROLE_DEFINITIONS["balancer"])
    other_sections = "\n\n".join(
        f"--- [{ROLE_DEFINITIONS.get(r, {}).get('label', r)}视角] ---\n{ans}"
        for r, ans in [(rd["role"], rd["answer"]) for rd in [
            {"role": other_role, "answer": other_ans}
            for other_pid, (other_role, other_ans) in other_answers.items()
        ]]
    )
    if not other_sections:
        other_sections = "[暂无其他模型的观点]"

    return (
        f"{role_def['system']}\n\n"
        f"原始问题：{question}\n\n"
        f"以下是其他视角的分析：\n{other_sections}\n\n"
        f"你的初始分析（{role_def['label']}）：\n{own_answer}\n\n"
        f"请基于你的角色立场，对其他观点进行评论，并给出修正后的最终视角分析："
    )


def build_llm_synthesis_prompt(
    question: str,
    role_answers: Dict[str, Dict[str, Any]],  # {pid: {role, label, answer, weight}}
) -> str:
    """Round 3 LLM 综合 Prompt（P1-3：替代 n-gram 的 LLM 语义综合）。"""
    sections = "\n\n".join(
        f"[{info['label']} - 可信度{info['weight']:.1f}]:\n{info['answer']}"
        for pid, info in role_answers.items()
        if info.get("answer")
    )
    no_repeat_hint = "综合以上"
    return (
        "以下是针对同一问题的多角色AI深度分析：\n\n"
        + sections
        + f"\n\n原始问题：{question}\n\n"
        "请综合以上多角色分析，输出一个完整、平衡、直接可用的最终答案。\n"
        "要求：\n"
        "1. 整合不同视角的有效信息，形成完整结论\n"
        "2. 当支持与质疑观点冲突时，明确说明判断依据\n"
        "3. 给出具体可操作的建议\n"
        f"4. 直接给出答案，不要重复\"{no_repeat_hint}\"等套话\n"
        "5. 答案应完整、独立，无需读者参考上面的原始分析"
    )


# ---------------------------------------------------------------------------
# 辩论引擎 v2
# ---------------------------------------------------------------------------

class DebateEngine:
    """多轮辩论编排器 v2。

    v2 架构特性：
    - 每轮内部全并行（asyncio.gather），消除串行堆叠超时
    - 角色固定分工（advocate/critic/balancer），强制多样性
    - LLM 直接综合最终答案（替代 n-gram 语义匹配）
    - 两级超时预算：每轮 PHASE_TIMEOUT + 整体 TOTAL_BUDGET
    - 渐进降级：成功≥2→综合，=1→直接返回，=0→立即fallback
    """

    # 两级超时配置（P2-2）
    PHASE_TIMEOUT = 70     # 每轮并行收割超时（秒），含余量
    SYNTHESIS_TIMEOUT = 70 # LLM 综合调用超时（秒）
    TOTAL_BUDGET = 210     # 整体硬上限（秒），3.5分钟

    def __init__(
        self,
        question: str,
        provider_ids: List[str],
        call_fn: Callable[[str, str], Coroutine[Any, Any, Dict[str, Any]]],
        max_rounds: int = 3,
        on_progress: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
        provider_health: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.question = question
        self.provider_ids = provider_ids
        self.call_fn = call_fn
        self.max_rounds = min(max(1, max_rounds), 2)
        self.on_progress = on_progress
        self.provider_health = provider_health or {}

        # 分配角色（P1-2）
        self.roles: Dict[str, str] = assign_roles(provider_ids)

        self.rounds: List[Dict[str, Any]] = []
        self.started_at = 0.0
        self.finished_at = 0.0

        # 最终综合答案（P1-3）
        self.llm_final_answer: str = ""

    async def _notify(self, msg: str):
        if self.on_progress:
            try:
                await self.on_progress(msg)
            except Exception:
                pass

    def _get_weight(self, pid: str) -> float:
        """根据 provider_health 计算可信度权重（P1-3 加权综合）。"""
        h = self.provider_health.get(pid, {})
        consecutive_failures = int(h.get("consecutive_failures", 0))
        # 连续失败越多权重越低，但最低保留 0.3
        return max(0.3, 1.0 - consecutive_failures * 0.2)

    async def run(self) -> Dict[str, Any]:
        """执行辩论流程，返回结构化结果（包含 LLM 最终答案）。"""
        self.started_at = time.perf_counter()

        try:
            # 整体预算包裹（P2-2）
            return await asyncio.wait_for(
                self._run_pipeline(),
                timeout=self.TOTAL_BUDGET,
            )
        except asyncio.TimeoutError:
            await self._notify(f"⚠️ 辩论超过总时限 {self.TOTAL_BUDGET}s，强制收割已有结果...")
            return self._finalize(forced=True)

    async def _run_pipeline(self) -> Dict[str, Any]:
        """辩论主流程管线。"""
        participator_count = len(self.provider_ids)

        # ---- Round 1: 角色分工并行回答（P1-1 + P1-2）----
        role_summary_parts = []
        for pid in self.provider_ids:
            role_label = ROLE_DEFINITIONS[self.roles[pid]]['label']
            role_summary_parts.append(f"{pid}→{role_label}")
        role_summary_str = " | ".join(role_summary_parts)

        await self._notify(
            f"🔵 Round 1/{self.max_rounds}: {participator_count} 个模型角色分工并行分析...\n"
            f"  • {role_summary_str}"
        )
        round1 = await self._run_round_parallel(
            round_num=1,
            build_prompt_fn=lambda pid: build_role_prompt(self.question, self.roles[pid]),
        )
        self.rounds.append(round1)

        r1_success = round1.get("success_count", 0)
        if r1_success == 0:
            await self._notify("⚠️ Round 1 全部失败，直接进入 fallback...")
            return self._finalize()

        if self.max_rounds < 2:
            return self._finalize()

        # ---- Round 2: 交叉评论并行（P1-1 + 上下文完整传递 P3-3）----
        await self._notify(f"🟡 Round 2/{self.max_rounds}: 模型间交叉评论...")
        round1_views = self._extract_views_with_roles(round1)
        round2 = await self._run_round_parallel(
            round_num=2,
            build_prompt_fn=lambda pid: build_cross_critique_prompt(
                question=self.question,
                own_role=self.roles[pid],
                own_answer=round1_views.get(pid, {}).get("answer", "[未获取到 Round 1 观点]"),
                other_answers={
                    other_pid: (info["role"], info["answer"])
                    for other_pid, info in round1_views.items()
                    if other_pid != pid
                },
            ),
        )
        self.rounds.append(round2)

        return self._finalize()

    async def _run_round_parallel(
        self,
        round_num: int,
        build_prompt_fn: Callable[[str], str],
    ) -> Dict[str, Any]:
        """执行一轮调用（P1-1：asyncio.gather 全并行 + Phase 超时）。"""
        round_started = time.perf_counter()

        # 构建所有任务（显式 task，便于超时后取消与回收）
        tasks: List[asyncio.Task] = []
        for pid in self.provider_ids:
            prompt = build_prompt_fn(pid)
            tasks.append(asyncio.create_task(self._call_single(pid, prompt)))

        phase_timed_out = False
        timed_out_cancelled = 0
        pending_after_reap = 0

        try:
            raw_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.PHASE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            phase_timed_out = True

            # 硬超时：显式取消未完成任务
            for t in tasks:
                if not t.done():
                    t.cancel()
                    timed_out_cancelled += 1

            # 有界回收，避免再次长时间挂起
            try:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=min(2.0, max(0.5, self.PHASE_TIMEOUT * 0.05)),
                )
                pending_after_reap = len(pending)
                for p in pending:
                    p.cancel()
            except Exception:
                done = set()
                pending_after_reap = sum(1 for t in tasks if not t.done())

            raw_results = []
            for t in tasks:
                if t.cancelled():
                    raw_results.append(asyncio.TimeoutError(f"phase timeout({self.PHASE_TIMEOUT}s), task cancelled"))
                elif t.done():
                    try:
                        raw_results.append(t.result())
                    except Exception as ex:
                        raw_results.append(ex)
                else:
                    raw_results.append(asyncio.TimeoutError(f"phase timeout({self.PHASE_TIMEOUT}s), task unreclaimed"))

            await self._notify(
                f"⚠️ Round {round_num} phase 超时：已取消 {timed_out_cancelled} 个任务，"
                f"回收后仍挂起 {pending_after_reap} 个。"
            )

        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_results):
            pid = self.provider_ids[idx]
            if isinstance(item, Exception):
                err_text = str(item)
                if phase_timed_out and not err_text:
                    err_text = f"phase timeout({self.PHASE_TIMEOUT}s)"
                results.append({
                    "provider_id": pid,
                    "role": self.roles[pid],
                    "ok": False,
                    "text": "",
                    "error": err_text,
                    "elapsed_ms": int((time.perf_counter() - round_started) * 1000),
                })
            else:
                row = dict(item) if isinstance(item, dict) else {
                    "provider_id": pid, "ok": False, "text": "", "error": "unexpected result type"
                }
                row["role"] = self.roles.get(pid, "balancer")
                results.append(row)

        elapsed_ms = int((time.perf_counter() - round_started) * 1000)
        success_count = sum(1 for r in results if r.get("ok"))

        # 进度通知（P3-2）
        result_summary = " | ".join(
            f"{'✅' if r.get('ok') else '❌'}{r['provider_id']}({r.get('elapsed_ms', 0)}ms)"
            for r in results
        )
        await self._notify(f"  Round {round_num} 结果: {result_summary}")

        return {
            "round": round_num,
            "results": results,
            "success_count": success_count,
            "failure_count": len(results) - success_count,
            "elapsed_ms": elapsed_ms,
            "phase_timed_out": phase_timed_out,
            "phase_cancelled_tasks": timed_out_cancelled,
            "phase_pending_after_reap": pending_after_reap,
        }

    async def _call_single(self, pid: str, prompt: str) -> Dict[str, Any]:
        """单次模型调用封装，异常转为失败结果。"""
        try:
            resp = await self.call_fn(pid, prompt)
            return {
                "provider_id": pid,
                "ok": resp.get("ok", False),
                "text": resp.get("text", ""),
                "error": resp.get("error", ""),
                "elapsed_ms": resp.get("elapsed_ms", 0),
            }
        except Exception as e:
            return {
                "provider_id": pid,
                "ok": False,
                "text": "",
                "error": str(e),
                "elapsed_ms": 0,
            }

    def _extract_views_with_roles(self, round_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """从一轮结果中提取每个模型的观点（含角色信息和完整文本）。"""
        views = {}
        for r in round_data.get("results", []):
            if r.get("ok") and r.get("text"):
                pid = r["provider_id"]
                views[pid] = {
                    "role": r.get("role", self.roles.get(pid, "balancer")),
                    "label": ROLE_DEFINITIONS.get(
                        r.get("role", self.roles.get(pid, "balancer")), {}
                    ).get("label", "分析"),
                    "answer": r["text"],  # 完整文本，不截断（P3-3）
                }
        return views

    async def _llm_synthesize(
        self,
        last_round: Optional[Dict[str, Any]] = None,
        round2_views: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """LLM 综合层（P1-3）：调用性能最好的 provider 做一次语义综合。

        渐进降级策略（P2-3）：
        - 成功 ≥ 2 个模型 → LLM 语义综合（调用最快的 provider）
        - 成功 = 1 个模型 → 直接使用该模型的答案，跳过综合
        - 成功 = 0 个模型 → 设置空答案，等 _finalize fallback
        """
        # 收集最新一轮的成功结果
        if round2_views is not None:
            views = round2_views
        elif last_round is not None:
            views = self._extract_views_with_roles(last_round)
        elif self.rounds:
            views = self._extract_views_with_roles(self.rounds[-1])
        else:
            views = {}

        success_count = len(views)

        # 渐进降级（P2-3）
        if success_count == 0:
            self.llm_final_answer = ""
            return

        if success_count == 1:
            # 单模型直接返回，省去综合开销
            only_answer = next(iter(views.values()))
            self.llm_final_answer = only_answer["answer"]
            await self._notify("ℹ️ 仅1个模型成功，直接使用其答案（跳过综合）")
            return

        # 构建带权重的角色答案映射
        role_answers = {
            pid: {
                "role": info["role"],
                "label": info["label"],
                "answer": info["answer"],
                "weight": self._get_weight(pid),
            }
            for pid, info in views.items()
        }

        synthesis_prompt = build_llm_synthesis_prompt(self.question, role_answers)

        # 选择调用最快的 provider（根据 health 统计）
        best_pid = self._pick_fastest_provider(list(views.keys()))
        await self._notify(f"🔀 LLM 综合中（使用 {best_pid}）...")

        try:
            synthesis_resp = await asyncio.wait_for(
                self.call_fn(best_pid, synthesis_prompt),
                timeout=self.SYNTHESIS_TIMEOUT,
            )
            if synthesis_resp.get("ok") and synthesis_resp.get("text"):
                self.llm_final_answer = synthesis_resp["text"]
                await self._notify(f"✅ LLM 综合完成（{synthesis_resp.get('elapsed_ms', 0)}ms）")
            else:
                # 综合失败 → 降级为最长成功答案
                self.llm_final_answer = max(
                    (info["answer"] for info in views.values()),
                    key=len, default=""
                )
                await self._notify(f"⚠️ LLM 综合失败，降级为最长答案: {synthesis_resp.get('error', '')}")
        except asyncio.TimeoutError:
            self.llm_final_answer = max(
                (info["answer"] for info in views.values()),
                key=len, default=""
            )
            await self._notify(f"⚠️ LLM 综合超时（{self.SYNTHESIS_TIMEOUT}s），降级为最长答案")

    def _pick_fastest_provider(self, provider_ids: List[str]) -> str:
        """选择历史平均响应最快的 provider（P2-4 动态超时思路）。"""
        if not provider_ids:
            return self.provider_ids[0] if self.provider_ids else ""
        best = min(
            provider_ids,
            key=lambda pid: self.provider_health.get(pid, {}).get("avg_elapsed_ms", 999999)
        )
        return best

    def _finalize(self, forced: bool = False) -> Dict[str, Any]:
        """构建最终辩论结果。"""
        self.finished_at = time.perf_counter()
        total_ms = int((self.finished_at - self.started_at) * 1000)

        last_round = self.rounds[-1] if self.rounds else {}
        last_results = last_round.get("results", [])

        # 构造 synthesis（保留兼容性，供 _analyze_convergence 和 HTML 报告使用）
        synth_input = {
            "per_model_results": last_results,
            "backend": "debate",
        }
        synthesis = build_synthesis_material(synth_input)

        total_calls = sum(len(r.get("results", [])) for r in self.rounds)
        total_successes = sum(r.get("success_count", 0) for r in self.rounds)
        convergence = self._analyze_convergence()

        # 最终答案优先级：LLM综合 > n-gram最长 > empty
        final_answer = self.llm_final_answer
        if not final_answer:
            # 降级：从最后一轮取最长成功答案
            last_views = self._extract_views_with_roles(last_round)
            final_answer = max(
                (info["answer"] for info in last_views.values()),
                key=len, default=""
            )

        return {
            "debate_id": f"debate-{now_iso_utc()[:19].replace(':', '').replace('-', '')}",
            "question": self.question,
            "participants": self.provider_ids,
            "roles": self.roles,
            "total_rounds": len(self.rounds),
            "max_rounds": self.max_rounds,
            "total_calls": total_calls,
            "total_successes": total_successes,
            "total_elapsed_ms": total_ms,
            "forced_finalize": forced,

            # ★ 核心输出：LLM 语义综合后的最终答案（P1-3）
            "final_answer": final_answer,
            "llm_synthesized": bool(self.llm_final_answer),

            "rounds": [
                {
                    "round": r.get("round"),
                    "success_count": r.get("success_count"),
                    "failure_count": r.get("failure_count"),
                    "elapsed_ms": r.get("elapsed_ms"),
                    "views": {
                        item["provider_id"]: {
                            "role": item.get("role", ""),
                            "label": ROLE_DEFINITIONS.get(item.get("role", ""), {}).get("label", ""),
                            "text": item["text"][:300],  # 报告摘要截断
                        }
                        for item in r.get("results", [])
                        if item.get("ok")
                    },
                }
                for r in self.rounds
            ],
            "synthesis": synthesis,
            "convergence": convergence,
            "per_model_results": last_results,
        }

    def _analyze_convergence(self) -> Dict[str, Any]:
        """分析辩论过程中的收敛趋势（保留，供 HTML 报告使用）。"""
        if len(self.rounds) < 2:
            return {"analyzed": False, "trend": "insufficient_rounds"}

        scores = []
        for r in self.rounds:
            synth = build_synthesis_material({
                "per_model_results": r.get("results", []),
                "backend": "debate",
            })
            core = safe_dict(synth, "core")
            cp_count = len(safe_list(core, "common_points"))
            diff_count = len(safe_list(core, "differences"))
            scores.append({
                "round": r.get("round"),
                "common_points": cp_count,
                "differences": diff_count,
                "consensus_ratio": cp_count / max(1, cp_count + diff_count),
            })

        first = scores[0]
        last = scores[-1]
        if last["consensus_ratio"] > first["consensus_ratio"]:
            trend = "converging"
        elif last["consensus_ratio"] < first["consensus_ratio"]:
            trend = "diverging"
        else:
            trend = "stable"

        return {
            "analyzed": True,
            "trend": trend,
            "per_round": scores,
            "convergence_delta": round(last["consensus_ratio"] - first["consensus_ratio"], 3),
        }


# ---------------------------------------------------------------------------
# 辩论报告 HTML 模板（更新以显示角色、LLM 综合结果）
# ---------------------------------------------------------------------------

DEBATE_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    color: #e0e0e0;
    padding: 32px;
    min-width: 560px;
    max-width: 680px;
  }
  .card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .header {
    display: flex; align-items: center; gap: 12px; margin-bottom: 20px;
  }
  .header .icon { font-size: 28px; }
  .header .title {
    font-size: 22px; font-weight: 700;
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .final-answer {
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    font-size: 14px;
    line-height: 1.7;
    color: #e8f5e9;
  }
  .final-answer-title {
    font-size: 13px; font-weight: 600; color: #4ade80;
    margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
  }
  .round-card {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 4px solid;
  }
  .round-1 { border-left-color: #60a5fa; }
  .round-2 { border-left-color: #fbbf24; }
  .round-3 { border-left-color: #4ade80; }
  .round-header {
    font-size: 14px; font-weight: 600;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }
  .round-badge {
    padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600;
  }
  .badge-r1 { background: rgba(96,165,250,0.2); color: #60a5fa; }
  .badge-r2 { background: rgba(251,191,36,0.2); color: #fbbf24; }
  .badge-r3 { background: rgba(74,222,128,0.2); color: #4ade80; }
  .view-item {
    padding: 8px 12px; margin-bottom: 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 8px; font-size: 12px; line-height: 1.5;
  }
  .view-provider { font-weight: 600; color: #a78bfa; margin-bottom: 2px; }
  .view-role { font-size: 10px; color: #6b7280; margin-bottom: 4px; }
  .role-advocate { color: #34d399; }
  .role-critic { color: #f87171; }
  .role-balancer { color: #60a5fa; }
  .convergence-bar {
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; background: rgba(255,255,255,0.04);
    border-radius: 12px; margin-top: 8px;
  }
  .conv-label { font-size: 12px; color: #9ca3af; }
  .conv-value { font-size: 18px; font-weight: 700; }
  .conv-converging { color: #4ade80; }
  .conv-diverging { color: #f87171; }
  .conv-stable { color: #fbbf24; }
  .stats-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .stat-item {
    flex: 1; min-width: 80px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px; padding: 12px; text-align: center;
  }
  .stat-value {
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .stat-label { font-size: 11px; color: #9ca3af; margin-top: 4px; }
  .section-title {
    font-size: 14px; font-weight: 600; color: #f59e0b;
    margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
  }
  .point-list { list-style: none; padding: 0; }
  .point-list li {
    padding: 8px 12px; margin-bottom: 6px;
    background: rgba(255,255,255,0.04);
    border-radius: 8px; font-size: 13px; line-height: 1.5;
    border-left: 3px solid #4ade80;
  }
  .support-badge {
    display: inline-block; font-size: 10px; padding: 2px 6px;
    border-radius: 10px; background: rgba(74,222,128,0.15);
    color: #4ade80; margin-left: 6px;
  }
  .llm-badge {
    display: inline-block; font-size: 10px; padding: 2px 8px;
    border-radius: 10px; background: rgba(96,165,250,0.15);
    color: #60a5fa; margin-left: 8px;
  }
  .footer { text-align: center; font-size: 10px; color: #6b7280; margin-top: 8px; }
</style>
</head>
<body>

<div class="card">
  <div class="header">
    <span class="icon">⚔️</span>
    <span class="title">多模型辩论报告 v2</span>
    {% if llm_synthesized %}
    <span class="llm-badge">LLM 综合</span>
    {% endif %}
  </div>
  <div class="stats-row">
    <div class="stat-item">
      <div class="stat-value">{{ total_rounds }}</div>
      <div class="stat-label">总轮次</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">{{ participants|length }}</div>
      <div class="stat-label">参与模型</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">{{ total_calls }}</div>
      <div class="stat-label">总调用</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">{{ "%.1f"|format(total_elapsed_ms / 1000) }}s</div>
      <div class="stat-label">总耗时</div>
    </div>
  </div>
</div>

{% if final_answer %}
<div class="card">
  <div class="final-answer-title">
    ✅ 最终综合答案
    {% if llm_synthesized %}<span class="llm-badge">AI语义综合</span>{% endif %}
  </div>
  <div class="final-answer">{{ final_answer }}</div>
</div>
{% endif %}

{% for r in rounds %}
<div class="card">
  <div class="round-header">
    <span class="round-badge badge-r{{ r.round }}">Round {{ r.round }}</span>
    <span>成功 {{ r.success_count }} / {{ r.success_count + r.failure_count }}</span>
    <span style="margin-left:auto;font-size:11px;color:#9ca3af;">{{ r.elapsed_ms }}ms</span>
  </div>
  {% for pid, view in r.views.items() %}
  <div class="view-item">
    <div class="view-provider">{{ pid }}</div>
    <div class="view-role role-{{ view.role }}">{{ view.label }}</div>
    {{ view.text }}
  </div>
  {% endfor %}
</div>
{% endfor %}

{% if convergence.analyzed %}
<div class="card">
  <div class="section-title">📈 收敛趋势</div>
  <div class="convergence-bar">
    <div class="conv-label">趋势</div>
    <div class="conv-value {% if convergence.trend == 'converging' %}conv-converging{% elif convergence.trend == 'diverging' %}conv-diverging{% else %}conv-stable{% endif %}">
      {% if convergence.trend == 'converging' %}✅ 趋于收敛{% elif convergence.trend == 'diverging' %}⚠️ 趋于发散{% else %}— 保持稳定{% endif %}
    </div>
    <div class="conv-label" style="margin-left:auto;">
      Δ = {{ "%.1f"|format(convergence.convergence_delta * 100) }}%
    </div>
  </div>
</div>
{% endif %}

{% if final_common_points %}
<div class="card">
  <div class="section-title">🤝 共识要点</div>
  <ul class="point-list">
    {% for cp in final_common_points %}
    <li>
      {{ cp.point }}
      <span class="support-badge">{{ cp.count }} 模型</span>
    </li>
    {% endfor %}
  </ul>
</div>
{% endif %}

<div class="footer">
  {{ debate_id }} · {{ participants|join(', ') }} · Debate Protocol v2.0
</div>

</body>
</html>
"""


def extract_debate_report_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """从辩论结果中提取报告模板数据（v2 增加 final_answer 和 llm_synthesized）。"""
    synthesis = safe_dict(result, "synthesis")
    core = safe_dict(synthesis, "core")
    common_points = safe_list(core, "common_points")

    return {
        "debate_id": result.get("debate_id", ""),
        "question": result.get("question", ""),
        "participants": result.get("participants", []),
        "roles": result.get("roles", {}),
        "total_rounds": result.get("total_rounds", 0),
        "total_calls": result.get("total_calls", 0),
        "total_elapsed_ms": result.get("total_elapsed_ms", 0),
        "rounds": result.get("rounds", []),
        "convergence": result.get("convergence", {}),
        # v2 新增
        "final_answer": result.get("final_answer", ""),
        "llm_synthesized": result.get("llm_synthesized", False),
        "final_common_points": [
            {"point": cp.get("point", ""), "count": cp.get("support_count", 0)}
            for cp in common_points[:6]
        ],
    }
