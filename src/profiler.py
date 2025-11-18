from collections import deque


class Profiler:
    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.data = {
            "capture": deque(maxlen=window),
            "infer": deque(maxlen=window),
            "latency": deque(maxlen=window),
            "interval": deque(maxlen=window),
        }

    def record(self, name: str, dt: float) -> None:
        if not name in self.data:
            self.data[name] = deque(maxlen=100)
        self.data[name].append(dt)


    def avg_ms(self, name: str) -> float:
        buf = self.data.get(name, None)
        if not buf:
            return 0.0
        return 1000.0 * sum(buf) / len(buf)

    def avg_fps(self) -> float:
        buf = self.data["interval"]
        if not buf:
            return 0.0
        avg_interval = sum(buf) / len(buf)
        if avg_interval <= 0.0:
            return 0.0
        return 1.0 / avg_interval

