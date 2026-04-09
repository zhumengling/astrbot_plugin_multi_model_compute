"""可视化汇总报告：HTML 模板 + 渲染辅助函数。

使用 AstrBot 的 html_render (Jinja2 + Playwright) 将 synthesis 结果渲染为精美图片。
"""

from __future__ import annotations

from typing import Any, Dict, List

from .utils import safe_dict, safe_list

# ---------------------------------------------------------------------------
# 报告数据提取
# ---------------------------------------------------------------------------

def extract_report_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """从 multi_model_compute 完整结果中提取报告所需数据。"""
    synthesis = safe_dict(result, "synthesis")
    core = safe_dict(synthesis, "core")
    timing = safe_dict(result, "timing")
    quality = safe_dict(core, "material_quality")

    common_points = safe_list(core, "common_points")
    differences = safe_list(core, "differences")
    notable_insights = safe_list(core, "notable_insights")
    conflict_points = safe_list(core, "conflict_points")
    failures = safe_list(synthesis, "failures")

    # per_model 简要摘要
    per_model = safe_list(result, "per_model_results")
    model_summary = []
    for item in per_model:
        if not isinstance(item, dict):
            continue
        model_summary.append({
            "provider_id": str(item.get("provider_id", "unknown")),
            "ok": bool(item.get("ok")),
            "elapsed_ms": int(item.get("elapsed_ms", 0) or 0),
            "text_preview": str(item.get("text", "") or "")[:120],
        })

    return {
        "status": str(result.get("status", "unknown")),
        "status_v2": str(result.get("status_v2", "unknown")),
        "backend": str(result.get("backend", "unknown")),
        "mode": str(result.get("mode", "balanced")),
        "run_id": str(result.get("run_id", "")),
        "confidence": result.get("confidence", 0),
        "recommendation": str(result.get("recommendation", "")),
        "query_preview": str(result.get("query", result.get("request", {}).get("query", "")))[:100],
        "quality_level": str(quality.get("level", "unknown")),
        "quality_overall": quality.get("overall"),
        "conflict_level": str(core.get("conflict_level", "low")),
        "success_count": int(core.get("success_count", 0)),
        "failure_count": int(core.get("failure_count", 0)),
        "participants_total": int(core.get("participants_total", 0)),
        "total_elapsed_ms": int(timing.get("total_elapsed_ms", 0)),
        "common_points": [
            {"point": cp.get("point", ""), "count": cp.get("support_count", 0)}
            for cp in common_points[:6]
        ],
        "differences": [
            {"provider_id": d.get("provider_id", ""), "position": d.get("position", "")[:100]}
            for d in differences[:4]
        ],
        "notable_insights": [
            {"point": ni.get("point", ""), "providers": ni.get("support_providers", [])}
            for ni in notable_insights[:4]
        ],
        "conflict_points": [
            {"topic": cp.get("topic", ""), "level": cp.get("conflict_level", "low")}
            for cp in conflict_points[:3]
        ],
        "failures": [
            {"provider_id": f.get("provider_id", ""), "error": str(f.get("error", ""))[:60]}
            for f in failures[:3]
        ],
        "model_summary": model_summary[:5],
        "cache_hit": bool(result.get("cache_hit", False)),
    }


# ---------------------------------------------------------------------------
# HTML 模板
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
    padding: 32px;
    min-width: 560px;
    max-width: 640px;
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
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
  }
  .header .icon { font-size: 28px; }
  .header .title {
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header .badge {
    margin-left: auto;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
  }
  .badge-ok { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
  .badge-partial { background: rgba(250,204,21,0.2); color: #fbbf24; border: 1px solid rgba(250,204,21,0.3); }
  .badge-failed { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
  .badge-cache { background: rgba(96,165,250,0.2); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }

  .stats-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .stat-item {
    flex: 1;
    min-width: 100px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
  }
  .stat-value {
    font-size: 24px;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .stat-label { font-size: 11px; color: #9ca3af; margin-top: 4px; }

  .section-title {
    font-size: 14px;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .point-list { list-style: none; padding: 0; }
  .point-list li {
    padding: 8px 12px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.5;
    border-left: 3px solid transparent;
  }
  .point-list li.consensus { border-left-color: #4ade80; }
  .point-list li.insight { border-left-color: #60a5fa; }
  .point-list li.conflict { border-left-color: #f87171; }
  .point-list li.diff { border-left-color: #fbbf24; }

  .support-badge {
    display: inline-block;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
    background: rgba(74,222,128,0.15);
    color: #4ade80;
    margin-left: 6px;
  }

  .model-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .model-row:last-child { border-bottom: none; }
  .model-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot-ok { background: #4ade80; }
  .dot-fail { background: #f87171; }
  .model-id { flex: 1; color: #d1d5db; }
  .model-time { color: #9ca3af; font-size: 11px; }

  .recommendation {
    background: linear-gradient(135deg, rgba(167,139,250,0.1), rgba(96,165,250,0.1));
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.6;
    color: #c4b5fd;
  }

  .quality-bar {
    height: 6px;
    border-radius: 3px;
    background: rgba(255,255,255,0.1);
    margin-top: 6px;
    overflow: hidden;
  }
  .quality-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
  }
  .quality-high { background: linear-gradient(90deg, #4ade80, #22c55e); }
  .quality-medium { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
  .quality-low { background: linear-gradient(90deg, #f87171, #ef4444); }

  .footer {
    text-align: center;
    font-size: 10px;
    color: #6b7280;
    margin-top: 8px;
  }

  .empty-hint {
    color: #6b7280;
    font-size: 12px;
    font-style: italic;
    padding: 6px 0;
  }
</style>
</head>
<body>

<div class="card">
  <div class="header">
    <span class="icon">🤖</span>
    <span class="title">多模型汇总报告</span>
    {% if cache_hit %}
    <span class="badge badge-cache">📦 缓存</span>
    {% endif %}
    <span class="badge {% if status_v2 == 'ok' %}badge-ok{% elif status_v2 == 'partial' %}badge-partial{% else %}badge-failed{% endif %}">
      {{ status_v2|upper }}
    </span>
  </div>

  <div class="stats-row">
    <div class="stat-item">
      <div class="stat-value">{{ success_count }}/{{ participants_total }}</div>
      <div class="stat-label">成功/总数</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">{{ "%.0f"|format(confidence * 100) }}%</div>
      <div class="stat-label">置信度</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">{{ "%.1f"|format(total_elapsed_ms / 1000) }}s</div>
      <div class="stat-label">耗时</div>
    </div>
    <div class="stat-item">
      <div class="stat-value">{{ quality_level }}</div>
      <div class="stat-label">材料质量</div>
    </div>
  </div>

  {% if quality_overall is not none %}
  <div class="quality-bar">
    <div class="quality-fill {% if quality_level == 'high' %}quality-high{% elif quality_level == 'medium' %}quality-medium{% else %}quality-low{% endif %}"
         style="width: {{ (quality_overall * 100)|int }}%"></div>
  </div>
  {% endif %}
</div>

{% if common_points %}
<div class="card">
  <div class="section-title">✅ 共识要点</div>
  <ul class="point-list">
    {% for cp in common_points %}
    <li class="consensus">
      {{ cp.point }}
      <span class="support-badge">{{ cp.count }} 模型共识</span>
    </li>
    {% endfor %}
  </ul>
</div>
{% endif %}

{% if notable_insights %}
<div class="card">
  <div class="section-title">💡 独到见解</div>
  <ul class="point-list">
    {% for ni in notable_insights %}
    <li class="insight">
      {{ ni.point }}
      <span class="support-badge" style="background:rgba(96,165,250,0.15);color:#60a5fa;">
        {{ ni.providers|join(', ') }}
      </span>
    </li>
    {% endfor %}
  </ul>
</div>
{% endif %}

{% if conflict_points %}
<div class="card">
  <div class="section-title">⚡ 冲突点</div>
  <ul class="point-list">
    {% for cp in conflict_points %}
    <li class="conflict">{{ cp.topic }} ({{ cp.level }})</li>
    {% endfor %}
  </ul>
</div>
{% endif %}

{% if differences %}
<div class="card">
  <div class="section-title">🔀 分歧观点</div>
  <ul class="point-list">
    {% for d in differences %}
    <li class="diff">
      <strong>{{ d.provider_id }}</strong>: {{ d.position }}
    </li>
    {% endfor %}
  </ul>
</div>
{% endif %}

{% if model_summary %}
<div class="card">
  <div class="section-title">📊 模型执行</div>
  {% for m in model_summary %}
  <div class="model-row">
    <div class="model-dot {% if m.ok %}dot-ok{% else %}dot-fail{% endif %}"></div>
    <div class="model-id">{{ m.provider_id }}</div>
    <div class="model-time">{{ m.elapsed_ms }}ms</div>
  </div>
  {% endfor %}
</div>
{% endif %}

{% if failures %}
<div class="card">
  <div class="section-title">❌ 失败记录</div>
  <ul class="point-list">
    {% for f in failures %}
    <li class="conflict">{{ f.provider_id }}: {{ f.error }}</li>
    {% endfor %}
  </ul>
</div>
{% endif %}

{% if recommendation %}
<div class="card">
  <div class="section-title">📋 归纳建议</div>
  <div class="recommendation">{{ recommendation }}</div>
</div>
{% endif %}

<div class="footer">
  {{ run_id }} · {{ mode }} · {{ backend }} · Multi-Model Compute v0.13
</div>

</body>
</html>
"""
