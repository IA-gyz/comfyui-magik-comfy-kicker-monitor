"""Optional event-based output tracking. Requires watchdog; never polls folders."""
import os
import time
from pathlib import Path


class OutputWatcher:
    def __init__(self, ledger, extensions):
        self.ledger, self.extensions = ledger, extensions
        self.observer = None
        self.directories = []
        self.error = None

    def configure(self, directories):
        directories = list(dict.fromkeys(str(Path(p).expanduser().resolve()) for p in directories))
        if directories == self.directories and not self.error and (self.observer or not directories):
            return
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2)
            self.observer = None
        self.directories, self.error = [], None
        if not directories:
            return
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            owner = self
            class Handler(FileSystemEventHandler):
                def on_closed(self, event):
                    if not event.is_directory:
                        owner.record(event.src_path)
                def on_moved(self, event):
                    if not event.is_directory:
                        owner.record(event.dest_path)
            observer = Observer()
            for directory in directories:
                if not Path(directory).is_dir():
                    raise ValueError('Output folder unavailable: ' + directory)
                observer.schedule(Handler(), directory, recursive=True)
            observer.start()
            self.observer, self.directories = observer, directories
        except Exception as exc:
            self.error = str(exc)

    def record(self, path):
        try:
            file = Path(path)
            if file.suffix.lower() not in self.extensions or not file.is_file():
                return
            ledger = self.ledger
            with ledger.lock:
                candidates = list(ledger.active.values())
                if not candidates and ledger.completed:
                    last = next(reversed(ledger.completed.values()))
                    # Account for delivery shortly after execution completes.
                    modified = file.stat().st_mtime
                    if (last.get('start_time') is not None and time.time() - last['end_time'] <= 2
                            and last['start_time'] <= modified <= last['end_time'] + 0.1):
                        candidates = [last]
                if len(candidates) == 1:
                    ledger.add_outputs(candidates[0]['prompt_id'],
                                       {'filename': file.name, 'filepath': os.path.abspath(path), 'type': 'output'})
        except OSError:
            pass
