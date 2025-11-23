import logging
import time

from collections import deque


LOG = logging.Logger('profiler', level=logging.DEBUG)


class Profiler:
    def __init__(self, window: int = 1000, log_interval_s: float = 5) -> None:
        self.window = window
        self.data = dict()
        self._last_log = time.time()
        self._log_interval = log_interval_s

    def record(self, name: str, dt: float) -> None:
        if not name in self.data:
            self.data[name] = deque(maxlen=1000)
        
        self.data[name].append(dt)
        self.log()

    def average(self, name: str) -> float:
        buf = self.data.get(name, None)
        if not buf:
            return 0
        
        if type(buf[0]) == float:
            return 1000.0 * sum(buf) / len(buf)
        
        return sum(buf) / len(buf)

    def log(self, force: bool = False):
        if time.time() - self._last_log < self._log_interval and not force:
            return

        self._last_log = time.time()
        LOG.log(level=logging.WARNING, msg=" | ".join(f"{k}: {self.average(k):.2f}" for k in self.data.keys()))

    def clean(self, keys: list[str]) -> None:
        for k in keys:
            self.data[k] = []
        
        self.log(force=True)