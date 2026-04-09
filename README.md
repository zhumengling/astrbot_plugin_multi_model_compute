# astrbot_plugin_multi_model_compute

AstrBot 多模型**计算与协调机制**（Tool-First）

> **核心定位**：
> - 负责复杂多模型调用、辩论演化、结构化汇总和动态调度。
> - 主打面向用户的直接体验：可通过 `/深度思考` 得到高质量的图文分析报告。
> - 在 Tool 模式下作为默认模型的专属外脑。

---

## 🚀 用户功能入口

### 1. 🧠 /深度思考

> 用于普通疑难问题的平行分析。

**用法**: `/深度思考 如何设计一个高并发系统`  
**可选**: `--mode balanced|fast|consensus|creative` `--max 4`

平台会挑选多个最能满足此场景且运行健壮的模型，经过语义共识检测，为您生成展示有：共识点、冲突点、和不同模型独到见解的可视化报告卡片。

### 2. ⚔️ /辩论 (Debate)

> 用于关键医疗/法律争端的互相交叉自证与纠偏。

**用法**: `/辩论 判定这起交通事故的法律定责`  
**特征**: 
- **多轮机制**（独立分析 -> 阅读对方观点反驳 -> 形成结论）。
- 对最终的协同共识有极高的确定性。
- 提供收敛性趋势（模型分歧是在增加还是缩小）及专门的可视化争论焦点报告图。

---

## 🛠 开发与系统流

### 1. 自动模型路由决策链
```text
Task Query -> 提取 Tag 匹配模型集 (Jaccard Score) -> 并入 Health Score 计算 -> 选择执行集
```
*每一次成功或超时的请求，都将通过 KV Storage 持久化影响下一次调用的模型选用顺位。*

### 2. JSON 返回模式
作为 LLM Tool 供默认模型调取时，支持精简上下文的 `brief`、`standard` 和用于全面审计的 `full` detail level。包含诸如 `cache_hit`、`synthesis_brief.conflict_level` 等用于其二次归纳的信息。

---

## ⚙️ 配置说明

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `model_1` ~ `model_5` | | 五个并列的模型配置槽位 |
| `model_X_tags` | | 该槽位适任的主功能点（用逗号分隔，如：代码,逻辑）|
| `default_participant_count` | 2 | 默认组团发包模型数 |
| `default_mode` | balanced | fast/balanced/creative/consensus |
| `cache_ttl_sec` | 300 | 高效的结果重用时间 |
| `monthly_budget_usd` | 10.0 | 多模型调用额度 |

### 🛠 调试用途命令

- `/mmstatus` — 监控状态、缓存命中率、Health 条数
- `/mmcalc <任务> [--mode] [--backend]` — 原始工具调试结果流发送
- `/mmtest` — API Health 的最快探针
- `/mmodels` — 系统映射模型能力一览

---

## 🌲 项目结构体系 (11 files, 3600+ lines)

```
astrbot_plugin_multi_model_compute/
├── main.py          — 节点与功能挂载
├── utils.py         — 通用算子
├── models.py        — 选型、特征库、Health 评分计算决策
├── provider_call.py — 后端请求分发与聚合、容错捕捉
├── synthesis.py     — 共识合并推理
├── schema.py        — Json 合规和 Fallback 推导
├── project.py       — Payload Budget、Token 切流
├── cache.py         — TTL KV 缓存处理
├── report.py        — 深度思考可视化 HTML/Jinja2 Render
├── debate.py        — Debate Protocol 轮次推演机制与报告 Render
└── __init__.py      — 初始化注册
```
