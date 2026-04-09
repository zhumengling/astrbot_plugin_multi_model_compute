from __future__ import annotations

import ast
import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PLUGIN_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PLUGIN_DIR.parent.parent
CONFIG_PATH = DATA_DIR / "config" / "astrbot_plugin_multi_model_compute_config.json"
RUNTIME_LOG = PLUGIN_DIR / "runtime.log"
METADATA_PATH = PLUGIN_DIR / "metadata.yaml"
STATIC_DIR = PLUGIN_DIR / "static"

_STAGE_RE = re.compile(r"stage_timings real_multi_call=(\{.*\})")
_BACKEND_RE = re.compile(r"backend_stage_timings=(\{.*?\}) timing=(\{.*\})")
_PER_MODEL_RE = re.compile(r"per_model_results_preview=(\[.*\])")
_PROBE_RE = re.compile(r"probe_result=(\{.*\})")


class WebUIServer:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.host = str(config.get('host', '0.0.0.0'))
        self.port = int(config.get('port', 8099))
        self._app = FastAPI(title='MultiModel Monitor', version='0.1.0')
        self._server: uvicorn.Server | None = None
        self._server_task = None
        self._setup_routes()

    def _setup_routes(self):
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_methods=['*'],
            allow_headers=['*'],
            allow_credentials=False,
        )
        if STATIC_DIR.exists():
            self._app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

        @self._app.get('/', response_class=HTMLResponse)
        async def index():
            path = STATIC_DIR / 'index.html'
            if not path.exists():
                return HTMLResponse('<h1>monitor page missing</h1>', status_code=404)
            return HTMLResponse(path.read_text(encoding='utf-8'))

        @self._app.get('/api/health')
        async def health():
            return {'status': 'ok', 'runtime_log_exists': RUNTIME_LOG.exists(), 'config_exists': CONFIG_PATH.exists()}

        @self._app.get('/api/summary')
        async def summary(provider: str = Query('all'), minutes: int = Query(0)):
            return JSONResponse(summarize(parse_runs(), provider, minutes))

        @self._app.get('/api/runs')
        async def runs(provider: str = Query('all'), minutes: int = Query(0)):
            data = summarize(parse_runs(), provider, minutes)
            return JSONResponse({'items': data['recent_runs'], 'count': len(data['recent_runs'])})

        @self._app.get('/api/events')
        async def events(request: Request, provider: str = Query('all'), minutes: int = Query(0)):
            async def event_stream():
                last = None
                while True:
                    if await request.is_disconnected():
                        break
                    payload = json.dumps(summarize(parse_runs(), provider, minutes), ensure_ascii=False)
                    if payload != last:
                        yield f'data: {payload}\n\n'
                        last = payload
                    else:
                        yield ':\n\n'
                    await asyncio.sleep(2)
            return StreamingResponse(
                event_stream(),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                },
            )

        @self._app.get('/api/config')
        async def get_config():
            cfg = read_config()
            return JSONResponse({'config': cfg})

        @self._app.post('/api/config')
        async def save_config(request: Request):
            payload = await request.json()
            cfg = write_config(payload if isinstance(payload, dict) else {})
            return JSONResponse({'ok': True, 'config': cfg})

        @self._app.get('/api/files')
        async def plugin_files():
            import os
            files_data = []
            for root, dirs, files in os.walk(str(PLUGIN_DIR)):
                if '__pycache__' in root or '.git' in root or 'static' in root:
                    continue
                for f in files:
                    if f.endswith(('.py', '.yaml', '.json', '.md')):
                        try:
                            path = Path(root) / f
                            content = path.read_text(encoding='utf-8-sig')
                            files_data.append({
                                'name': f,
                                'path': str(path.relative_to(PLUGIN_DIR)).replace('\\', '/'),
                                'type': f.split('.')[-1],
                                'size': len(content),
                                'content': content
                            })
                        except Exception:
                            pass
            return JSONResponse({'files': files_data})

    async def start(self):
        if self._server_task and not self._server_task.done():
            return
        
        import socket

        def get_free_port(host, start_port, max_tries=10):
            for port in range(start_port, start_port + max_tries):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind((host, port))
                        return port
                    except OSError:
                        pass
            return None

        free_port = get_free_port(self.host, self.port)
        if not free_port:
            print(f"[multi_model_compute] WebUI 启动失败: 端口 {self.port} 及其附近的端口均被占用。")
            return
        
        self.port = free_port
        self.actual_port = free_port

        cfg = uvicorn.Config(self._app, host=self.host, port=self.port, log_level='warning', loop='asyncio', lifespan='on')
        self._server = uvicorn.Server(cfg)
        
        self._server_task = asyncio.create_task(self._server.serve())
        for _ in range(50):
            if getattr(self._server, 'started', False):
                return
            if self._server_task.done():
                try:
                    e = self._server_task.exception()
                    print(f"WebUI 异常退出: {e}")
                except Exception:
                    pass
                return
            await asyncio.sleep(0.1)

    async def stop(self):
        server = self._server
        task = self._server_task
        if server:
            server.should_exit = True
        if task:
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                if server:
                    server.force_exit = True
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._server = None
        self._server_task = None


def _safe_read(path: Path, default: str = '') -> str:
    try:
        return path.read_text(encoding='utf-8-sig', errors='ignore')
    except Exception:
        return default


def read_config() -> Dict[str, Any]:
    try:
        return json.loads(_safe_read(CONFIG_PATH, '{}'))
    except Exception:
        return {}


def read_metadata() -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in _safe_read(METADATA_PATH).splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            data[k.strip()] = v.strip().strip('"')
    return data


def parse_stage_dict(line: str):
    m = _STAGE_RE.search(line)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None


def parse_backend_timing(line: str):
    m = _BACKEND_RE.search(line)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))
    except Exception:
        return None


def parse_runs() -> List[Dict[str, Any]]:
    lines = _safe_read(RUNTIME_LOG).splitlines()
    runs: List[Dict[str, Any]] = []
    pending = None
    for idx, line in enumerate(lines):
        if 'stage_timings real_multi_call=' in line:
            stage = parse_stage_dict(line)
            if stage:
                pending = {'id': f'run-{idx}', 'seq': len(runs) + 1, 'observed_type': 'direct_log', 'stage': stage, 'timestamp_label': f'log_line_{idx+1}'}
        elif pending and 'backend_stage_timings=' in line and ' timing=' in line:
            parsed = parse_backend_timing(line)
            if not parsed:
                continue
            backend_stage, timing = parsed
            participants = pending['stage'].get('participants_after_circuit_breaker', []) or []
            run = {
                **pending,
                'backend_stage': backend_stage,
                'timing': timing,
                'participants': participants,
                'participant_count': len(participants),
                'total_elapsed_ms': int((timing or {}).get('total_elapsed_ms', 0) or 0),
                'avg_elapsed_ms': int((timing or {}).get('avg_elapsed_ms', 0) or 0),
                'fastest_ms': int((timing or {}).get('fastest_ms', 0) or 0),
                'slowest_ms': int((timing or {}).get('slowest_ms', 0) or 0),
                'timeout_sec': float((timing or {}).get('timeout_sec', 0) or 0),
                'status_inferred': 'pending_result_details' if participants else 'unknown',
                'provider_rows': [{'provider_id': p, 'ok_inferred': False, 'elapsed_ms': int((timing or {}).get('avg_elapsed_ms', 0) or 0), 'failure_type_inferred': 'unknown', 'observed_type': 'inferred_from_timing', 'text_preview': '', 'error': ''} for p in participants],
                'empty_model_output_error_count': 0,
                'failure_types': [],
                'failure_count_inferred': 0,
                'success_count_inferred': 0,
                'reply_observed_type': 'not_recorded_yet',
            }
            lookahead = lines[idx+1: idx+8]
            probe_items = []
            for la in lookahead:
                if 'per_model_results_preview=' in la:
                    m = _PER_MODEL_RE.search(la)
                    if m:
                        try:
                            preview = ast.literal_eval(m.group(1))
                            mapped = []
                            empty_cnt = 0
                            failure_types = []
                            for item in preview:
                                err = str(item.get('error', '') or '')
                                if 'EmptyModelOutputError' in err:
                                    empty_cnt += 1
                                    failure_types.append('EmptyModelOutputError')
                                mapped.append({'provider_id': item.get('provider_id', ''), 'ok_inferred': bool(item.get('ok', False)), 'elapsed_ms': int(item.get('elapsed_ms', 0) or 0), 'failure_type_inferred': 'EmptyModelOutputError' if 'EmptyModelOutputError' in err else ('error' if err else ''), 'observed_type': 'direct_log_preview', 'text_preview': str(item.get('text_preview', '') or ''), 'error': err})
                            run['provider_rows'] = mapped or run['provider_rows']
                            run['empty_model_output_error_count'] = empty_cnt
                            run['failure_types'] = failure_types
                            run['success_count_inferred'] = sum(1 for x in run['provider_rows'] if x.get('ok_inferred'))
                            run['failure_count_inferred'] = sum(1 for x in run['provider_rows'] if not x.get('ok_inferred'))
                            run['status_inferred'] = 'success_observed' if run['success_count_inferred'] > 0 else ('all_failed_observed' if run['failure_count_inferred'] > 0 else run.get('status_inferred', 'unknown'))
                            run['reply_observed_type'] = 'direct_log_preview'
                        except Exception:
                            pass
                elif 'probe_result=' in la:
                    m = _PROBE_RE.search(la)
                    if m:
                        try:
                            item = ast.literal_eval(m.group(1))
                            probe_items.append({'provider_id': item.get('provider_id', ''), 'ok_inferred': bool(item.get('ok', False)), 'elapsed_ms': int(item.get('elapsed_ms', 0) or 0), 'failure_type_inferred': 'EmptyModelOutputError' if 'EmptyModelOutputError' in str(item.get('error', '') or '') else ('error' if item.get('error') else ''), 'observed_type': 'direct_probe_log', 'text_preview': str(item.get('text_preview', '') or ''), 'error': str(item.get('error', '') or '')})
                        except Exception:
                            pass
            if probe_items and run.get('reply_observed_type') != 'direct_log_preview':
                run['provider_rows'] = probe_items
                run['empty_model_output_error_count'] = sum(1 for x in probe_items if 'EmptyModelOutputError' in str(x.get('error', '')))
                run['failure_types'] = list({x['failure_type_inferred'] for x in probe_items if x.get('failure_type_inferred')})
                run['success_count_inferred'] = sum(1 for x in probe_items if x.get('ok_inferred'))
                run['failure_count_inferred'] = sum(1 for x in probe_items if not x.get('ok_inferred'))
                run['status_inferred'] = 'success_observed' if run['success_count_inferred'] > 0 else ('all_failed_observed' if run['failure_count_inferred'] > 0 else run.get('status_inferred', 'unknown'))
                run['reply_observed_type'] = 'direct_probe_log'
            runs.append(run)
            pending = None
    return runs


def summarize(runs: List[Dict[str, Any]], provider: str = 'all', minutes: int = 0) -> Dict[str, Any]:
    filtered = runs
    if provider != 'all':
        filtered = [r for r in filtered if provider in r.get('participants', []) or any(provider == x.get('provider_id') for x in r.get('provider_rows', []))]
    if minutes > 0:
        filtered = filtered[-minutes:] if len(filtered) > minutes else filtered
    cfg = read_config()
    meta = read_metadata()
    enabled_slots = []
    for i in range(1, 6):
        mid = cfg.get(f'model_{i}', '')
        tags = cfg.get(f'model_{i}_tags', '')
        enabled_slots.append({'slot': f'model_{i}', 'provider_id': mid, 'enabled': bool(mid), 'tags': tags})
    provider_stats = defaultdict(lambda: {'provider_id': '', 'runs': 0, 'success_runs': 0, 'failure_runs': 0, 'avg_elapsed_ms': 0, 'recent_elapsed_ms': [], 'consecutive_failures_inferred': 0, 'observed_type': 'inferred'})
    for r in filtered:
        for row in r.get('provider_rows', []):
            p = row.get('provider_id', '')
            if not p:
                continue
            st = provider_stats[p]
            st['provider_id'] = p
            st['runs'] += 1
            if row.get('ok_inferred'):
                st['success_runs'] += 1
            else:
                st['failure_runs'] += 1
            st['recent_elapsed_ms'].append(int(row.get('elapsed_ms', 0) or 0))
    for st in provider_stats.values():
        vals = [v for v in st['recent_elapsed_ms'] if v]
        st['avg_elapsed_ms'] = int(sum(vals) / len(vals)) if vals else 0
        st['success_rate_inferred'] = round((st['success_runs'] / st['runs']) * 100, 1) if st['runs'] else 0
        st['recent_elapsed_ms'] = vals[-10:]
    trend = [{'x': idx + 1, 'elapsed_ms': r.get('total_elapsed_ms', 0), 'participants': r.get('participant_count', 0)} for idx, r in enumerate(filtered[-30:])]
    return {
        'plugin': {'name': meta.get('name', 'astrbot_plugin_multi_model_compute'), 'display_name': meta.get('display_name', ''), 'version': meta.get('version', 'unknown'), 'desc': meta.get('desc', ''), 'observed_type': 'direct_file'},
        'config': {'default_mode': cfg.get('default_mode', 'balanced'), 'default_participant_count': cfg.get('default_participant_count', 0), 'real_call_timeout_sec': cfg.get('real_call_timeout_sec', 0), 'cache_ttl_sec': cfg.get('cache_ttl_sec', 0), 'return_merged_answer': cfg.get('return_merged_answer', False), 'return_candidates': cfg.get('return_candidates', False), 'enabled_slots': enabled_slots, 'observed_type': 'direct_file'},
        'health': {'log_runs_count': len(filtered), 'empty_model_output_error_count': sum(r.get('empty_model_output_error_count', 0) for r in filtered), 'status': 'ok' if filtered else 'no_data', 'observed_type': 'inferred_from_runtime_log'},
        'providers': list(provider_stats.values()),
        'recent_runs': filtered[-50:][::-1],
        'trend': trend,
        'limits': {'unavailable_direct_metrics': ['插件进程内 provider_health 未直接暴露，页面部分统计仍为推导值', '缓存命中/条目数当前未直接从运行时内存读取']}
    }


def write_config(data: Dict[str, Any]) -> Dict[str, Any]:
    current = read_config()
    allowed_scalar = {
        'default_mode', 'default_participant_count', 'real_call_timeout_sec', 'cache_ttl_sec',
        'return_merged_answer', 'return_candidates', 'webui_enabled', 'webui_host', 'webui_port'
    }
    for key in allowed_scalar:
        if key in data:
            current[key] = data[key]
    for i in range(1, 6):
        mk = f'model_{i}'
        tk = f'model_{i}_tags'
        if mk in data:
            current[mk] = str(data.get(mk, '') or '')
        if tk in data:
            current[tk] = str(data.get(tk, '') or '')
    CONFIG_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
    return current
