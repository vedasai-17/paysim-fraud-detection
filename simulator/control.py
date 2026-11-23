"""Run one stoppable replay per dashboard process."""
from threading import Event, Lock, Thread
from uuid import uuid4

from simulator.replay import replay


class ReplayController:
    def __init__(self):
        self.lock = Lock()
        self.stop_event = Event()
        self.status = "Idle"
        self.sent = 0
        self.error = None
        self.run_id = None

    def snapshot(self):
        with self.lock:
            return self.status, self.sent, self.error, self.run_id

    def start(self, data, url, rate, limit, start_step):
        with self.lock:
            if self.status in ("Running", "Stopping"):
                return
            self.stop_event.clear()
            self.status, self.sent, self.error = "Running", 0, None
            self.run_id = uuid4().hex[:12]
            Thread(target=self._run, args=(data, url, rate, limit, start_step), daemon=True).start()

    def stop(self):
        with self.lock:
            if self.status == "Running":
                self.status = "Stopping"
                self.stop_event.set()

    def _record_event(self, *_):
        with self.lock:
            self.sent += 1

    def _run(self, *args):
        try:
            replay(*args, run_id=self.run_id, stop=self.stop_event, on_event=self._record_event)
        except Exception as exc:
            with self.lock:
                self.status, self.error = "Failed", str(exc)
        else:
            with self.lock:
                self.status = "Stopped" if self.stop_event.is_set() else "Completed"
