"""Passive global execution monitor. No workflow nodes or extra listening port."""
import functools
import logging
import os
import sys
from contextvars import ContextVar
from pathlib import Path
from aiohttp import web
from server import PromptServer
from execution import PromptExecutor
from .state import Ledger
from .watcher import OutputWatcher

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
_log = logging.getLogger(__name__)
_server = PromptServer.instance

# Observe before session routing, preserving ComfyUI's original event delivery.
# Idempotent even when the package is loaded again.
if not hasattr(_server, '_comfy_kicker_ledger'):
    _ledger = Ledger()
    _original = PromptExecutor.add_message

    @functools.wraps(_original)
    def _add_message(executor, event, data, broadcast):
        try:
            _ledger.observe(event, data)
        except Exception:
            _log.exception('Comfy Kicker monitoring failed; execution continues')
        return _original(executor, event, data, broadcast)

    PromptExecutor.add_message = _add_message
    _server._comfy_kicker_ledger = _ledger

    _original_send = _server.send_sync

    @functools.wraps(_original_send)
    def _send_sync(event, data, sid=None):
        try:
            if event == 'executed':
                _ledger.observe(event, data)
            if event == 'status' and isinstance(data, dict):
                count = data.get('status', {}).get('exec_info', {}).get('queue_remaining')
                # ComfyUI queue_updated sends status.exec_info.queue_remaining.
                if isinstance(count, int):
                    _ledger.queue_update(count)
        except Exception:
            _log.exception('Comfy Kicker queue monitoring failed; delivery continues')
        return _original_send(event, data, sid)

    _server.send_sync = _send_sync

    # Track write-open paths inside the execution context, without scanning directories.
    # This covers save nodes that return no UI filename descriptors.
    _writes = ContextVar('comfy_kicker_writes', default=None)
    _extensions = {'.txt', '.csv', '.tsv', '.json', '.png', '.jpg', '.jpeg', '.webp',
                   '.bmp', '.tif', '.tiff', '.gif', '.mp4', '.webm', '.mkv', '.mov',
                   '.wav', '.mp3', '.ogg', '.oga', '.flac', '.opus', '.safetensors'}

    _watcher = OutputWatcher(_ledger, _extensions)

    def _audit(event, args):
        paths = _writes.get()
        if paths is None:
            return
        try:
            path = None
            if event == 'open':
                candidate, mode, flags = args
                if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
                    path = candidate
            elif event == 'os.rename':
                path = args[1]
                if len(args) > 3 and args[3] not in (None, -1):
                    return  # a relative dir_fd destination cannot be resolved safely here
            if isinstance(path, (str, bytes)):
                path = os.path.abspath(os.fsdecode(path))
                if os.path.splitext(path)[1].lower() in _extensions:
                    paths.add(path)
        except Exception:
            pass  # audit observers must never interfere with file saving

    sys.addaudithook(_audit)

    def _collect(executor, pid, paths):
        try:
            if pid is None:
                return
            history = getattr(executor, 'history_result', {})
            if isinstance(history, dict):
                _ledger.add_history_outputs(str(pid), history.get('outputs', {}))
            import folder_paths
            temp = Path(folder_paths.get_temp_directory()).resolve()
            for path in paths:
                file = Path(path)
                if file.is_file() and not file.resolve().is_relative_to(temp):
                    _ledger.add_outputs(str(pid), {'filename': file.name, 'filepath': str(file.absolute()), 'type': 'output'})
        except Exception:
            _log.exception('Comfy Kicker output collection failed; execution continues')

    _original_execute = PromptExecutor.execute

    @functools.wraps(_original_execute)
    def _execute(executor, *args, **kwargs):
        pid = kwargs.get('prompt_id', args[1] if len(args) > 1 else None)
        paths = set()
        token = _writes.set(paths)
        try:
            return _original_execute(executor, *args, **kwargs)
        finally:
            _writes.reset(token)
            _collect(executor, pid, paths)

    PromptExecutor.execute = _execute

    if hasattr(PromptExecutor, 'execute_async'):
        _original_async = PromptExecutor.execute_async

        @functools.wraps(_original_async)
        async def _execute_async(executor, *args, **kwargs):
            pid = kwargs.get('prompt_id', args[1] if len(args) > 1 else None)
            paths = _writes.get()
            paths = paths if paths is not None else set()
            token = _writes.set(paths)
            try:
                return await _original_async(executor, *args, **kwargs)
            finally:
                _writes.reset(token)
                _collect(executor, pid, paths)

        PromptExecutor.execute_async = _execute_async

    @_server.routes.post('/comfy-kicker/clear-queue')
    async def clear_queue(request):
        queue = _server.prompt_queue
        # Same pending-heap operation as wipe_queue, with the count captured
        # under its mutex. Currently running jobs are kept separately.
        if not hasattr(queue, 'mutex') or not isinstance(getattr(queue, 'queue', None), list):
            return web.json_response({'error': 'Queue implementation does not support counted clearing'}, status=501)
        with queue.mutex:
            removed = len(queue.queue)
            queue.queue = []
        warning = None
        try:
            notify = getattr(_server, 'queue_updated', None)
            if not callable(notify):
                notify = getattr(queue, 'queue_updated', None)
            if callable(notify):
                notify()
            else:
                warning = 'Queue cleared; queue-status notification callback unavailable.'
        except Exception:
            _log.exception('Queue cleared, but queue-status notification failed')
            warning = 'Queue cleared; queue-status notification failed. Monitoring will refresh on the next poll.'
        return web.json_response({'removed': removed, 'warning': warning})

    @_server.routes.post('/comfy-kicker/output-folders')
    async def output_folders(request):
        data = await request.json()
        directories = data.get('directories') if isinstance(data, dict) else None
        if not isinstance(directories, list) or not all(isinstance(p, str) for p in directories):
            return web.json_response({'error': 'directories must be a list of paths'}, status=400)
        import asyncio
        await asyncio.to_thread(_watcher.configure, directories)
        return web.json_response({'directories': _watcher.directories, 'error': _watcher.error})

    @_server.routes.get('/comfy-kicker/monitor')
    async def monitor(request):
        try:
            since = max(0, int(request.query.get('since', '0')))
        except ValueError:
            return web.json_response({'error': 'since must be an integer'}, status=400)
        result = _ledger.snapshot(since, session=request.query.get('session'),
                                  summary=request.query.get('summary') == '1')
        result['folder_tracking'] = {'directories': _watcher.directories, 'error': _watcher.error}
        # Count only: never copy or traverse queued workflow payloads.
        count_tasks = getattr(_server.prompt_queue, 'get_tasks_remaining', None)
        if callable(count_tasks):
            result['queue'] = count_tasks()
        with _ledger.lock:
            result['pending'] = max(0, result['queue'] - len(_ledger.active))
        # Older implementations use the count observed in queue-status events.
        return web.json_response(result, headers={'Cache-Control': 'no-store'})

    _log.info('Comfy Kicker lifecycle monitor enabled at /comfy-kicker/monitor')
