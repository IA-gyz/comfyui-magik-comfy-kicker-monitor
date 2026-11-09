"""Thread-safe lifecycle ledger; independent of ComfyUI and Qt."""
import threading
import time
import uuid
import posixpath
from collections import OrderedDict


class Ledger:
    def __init__(self, limit=9999, clock=time.monotonic, wall=time.time):
        self.session = uuid.uuid4().hex
        self.limit, self.clock, self.wall = limit, clock, wall
        self.lock = threading.RLock()
        self.active, self.completed = {}, OrderedDict()
        self.sequence = 0
        self.batches_finished = 0
        self.queue_count = 0
        self.batch_had_execution = False
        self.cached_nodes = {}

    def observe(self, event, data):
        if not isinstance(data, dict) or not data.get('prompt_id'):
            return
        pid = str(data['prompt_id'])
        with self.lock:
            if event == 'execution_cached':
                self.cached_nodes[pid] = set(map(str, data.get('nodes') or []))
            elif event == 'executed':
                if str(data.get('node')) not in self.cached_nodes.get(pid, set()):
                    self.add_outputs(pid, data.get('output'))
            elif event == 'execution_start':
                if pid not in self.active and pid not in self.completed:
                    self.sequence += 1
                    self.batch_had_execution = True
                    self.active[pid] = dict(prompt_id=pid, start_time=self.wall(),
                                            started=self.clock(), sequence=self.sequence, files=[])
            elif event in ('execution_success', 'execution_error', 'execution_interrupted'):
                if pid in self.completed:
                    return
                run = self.active.pop(pid, None)
                # A monitor loaded mid-run cannot fabricate its start time.
                if run is None:
                    run = dict(prompt_id=pid, start_time=None, files=[])
                started = run.pop('started', None)
                self.sequence += 1
                run.update(end_time=self.wall(), duration=max(0, self.clock()-started) if started is not None else None,
                           outcome={'execution_success': 'success', 'execution_error': 'error',
                                    'execution_interrupted': 'interrupted'}[event], sequence=self.sequence)
                self.completed[pid] = run
                while len(self.completed) > self.limit:
                    old_pid, _ = self.completed.popitem(last=False)
                    self.cached_nodes.pop(old_pid, None)

    def add_outputs(self, pid, output):
        """Extract standard saved-output descriptors, excluding temporary previews."""
        def filenames(value):
            if isinstance(value, dict):
                name = value.get('filename')
                if isinstance(name, str) and name and value.get('type', 'output') == 'output':
                    # ComfyUI descriptors use '/' for subfolders. On Linux,
                    # backslashes are literal filename characters.
                    yield posixpath.basename(name), value.get("filepath")
                for child in value.values():
                    if isinstance(child, (dict, list, tuple)):
                        yield from filenames(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from filenames(child)
        with self.lock:
            record = self.active.get(pid) or self.completed.get(pid)
            if record is not None:
                files = record.setdefault('files', [])
                for name, path in filenames(output):
                    changed = name not in files
                    if changed:
                        files.append(name)
                    if isinstance(path, str) and path:
                        paths = record.setdefault('file_paths', {}).setdefault(name, [])
                        if path not in paths:
                            paths.append(path)
                            changed = True
                    if changed:
                        self.sequence += 1
                        record['revision'] = self.sequence

    def add_history_outputs(self, pid, outputs):
        with self.lock:
            if isinstance(outputs, dict):
                for node, output in outputs.items():
                    if str(node) not in self.cached_nodes.get(pid, set()):
                        self.add_outputs(pid, output)

    def queue_update(self, count):
        with self.lock:
            self.queue_count = max(0, count)
            if count == 0 and self.batch_had_execution and not self.active:
                self.batches_finished += 1
                self.batch_had_execution = False

    def snapshot(self, since=0, session=None, summary=False):
        with self.lock:
            if summary:
                return dict(protocol=1, session=self.session, sequence=self.sequence,
                            batches_finished=self.batches_finished, queue=self.queue_count, summary=True)
            reset = session is not None and session != self.session
            if reset:
                since = 0
            active = [dict((k, v) for k, v in r.items() if k != 'started') |
                      {'elapsed': max(0, self.clock()-r['started'])} for r in self.active.values()]
            return dict(protocol=1, session=self.session, sequence=self.sequence, batches_finished=self.batches_finished, queue=self.queue_count,
                        reset=reset, retained=list(self.completed),
                        oldest_sequence=next(iter(self.completed.values()))['sequence'] if self.completed else self.sequence,
                        active=active, entries=[dict(r, files=list(r.get('files', [])), file_paths={k: list(v) for k, v in r.get('file_paths', {}).items()}) for r in self.completed.values() if max(r['sequence'], r.get('revision', 0)) > since])
