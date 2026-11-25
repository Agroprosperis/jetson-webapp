import json
import os
import threading
import time

import stage_link
import stage_math
import stage_store


VOLTAGE_REGISTER = 0x0178
SLAVE_X = 1
SLAVE_Y = 2
SETTLE_SECONDS = 0.3
MOTION_BLOCKED = (
    "Рух столу ще не підтверджено на залізі. "
    "Див. docs/stage/README.md"
)


class StageError(Exception):
    pass


class Stage:
    def __init__(self, link):
        self.link = link
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._busy = False
        self._note = ""
        self._voltage = None
        self._voltage_at = 0.0

    @property
    def simulated(self):
        return self.link is None

    @property
    def motion_enabled(self):
        if self.simulated:
            return True
        return os.environ.get("TILLETIA_STAGE_MOTION") == "1"

    def status(self):
        state = stage_store.get_state()
        user = self._user_position(state)
        voltage = self._voltage_cached()
        with self._lock:
            busy = self._busy
            note = self._note
        return {
            "connected": self.link is not None and voltage is not None,
            "simulated": self.simulated,
            "motion_enabled": self.motion_enabled,
            "busy": busy,
            "note": note,
            "voltage_v": voltage,
            "position_mm": user,
            "position_source": "simulated" if self.simulated else "commanded",
            "speed_mm_s": state["speed_mm_s"],
            "step_mm": state["step_mm"],
            "workspace_mm": state["workspace_mm"],
            "zero_set": state["zero_set"],
            "points": state["points"],
            "routes": stage_store.list_routes(),
        }

    def update_settings(self, speed_mm_s=None, step_mm=None, workspace_mm=None):
        fields = {}
        if speed_mm_s is not None:
            fields["speed_mm_s"] = stage_math.clean_speed(speed_mm_s)
        if step_mm is not None:
            step = float(step_mm)
            if step <= 0 or step > 5:
                raise ValueError("Крок від 0 до 5 мм")
            fields["step_mm"] = round(step, 4)
        if workspace_mm is not None:
            box = {
                "min_x": float(workspace_mm["min_x"]),
                "max_x": float(workspace_mm["max_x"]),
                "min_y": float(workspace_mm["min_y"]),
                "max_y": float(workspace_mm["max_y"]),
            }
            if box["min_x"] >= box["max_x"] or box["min_y"] >= box["max_y"]:
                raise ValueError("Робоче вікно порожнє")
            fields.update(box)
        if fields:
            stage_store.update_state(**fields)
        return self.status()

    def jog(self, axis, sign):
        if axis not in ("x", "y"):
            raise ValueError("Вісь x або y")
        sign = 1 if int(sign) >= 0 else -1
        with self._lock:
            if self._busy:
                raise StageError("Стіл уже їде маршрутом")
            state = stage_store.get_state()
            user = self._user_position(state)
            step = state["step_mm"] * sign
            target = {"x": user["x"], "y": user["y"]}
            target[axis] = round(user[axis] + step, 4)
            self._require_inside(target, state["workspace_mm"])
            self._require_motion()
            self._set_machine_target(state, target)
            self._note = ""
        return self.status()

    def stop(self):
        self._stop.set()
        with self._lock:
            self._note = "Зупинено"
            self._busy = False
        return self.status()

    def set_zero(self):
        with self._lock:
            if self._busy:
                raise StageError("Спочатку зупиніть маршрут")
            state = stage_store.get_state()
            stage_store.update_state(
                zero_x=state["machine_mm"]["x"],
                zero_y=state["machine_mm"]["y"],
                zero_set=1,
            )
            self._note = "Нуль поставлено в поточному кадрі"
        return self.status()

    def add_point(self, point=None):
        state = stage_store.get_state()
        if point is None:
            current = self._user_position(state)
        else:
            current = stage_math.clean_points([point])[0]
        points = state["points"] + [current]
        stage_math.clean_points(points)
        stage_store.update_state(points_json=json.dumps(points))
        return self.status()

    def pop_point(self):
        state = stage_store.get_state()
        stage_store.update_state(points_json=json.dumps(state["points"][:-1]))
        return self.status()

    def replace_points(self, points):
        cleaned = stage_math.clean_points(points)
        stage_store.update_state(points_json=json.dumps(cleaned))
        return self.status()

    def fill_goryaev(self, step_mm, width_mm=3.0, height_mm=3.0, second_offset=None):
        points = stage_math.serpentine(width_mm, height_mm, float(step_mm))
        if second_offset is not None:
            dx = float(second_offset["x"])
            dy = float(second_offset["y"])
            points = points + [
                {"x": round(point["x"] + dx, 4), "y": round(point["y"] + dy, 4)}
                for point in points
            ]
            stage_math.clean_points(points)
        stage_store.update_state(points_json=json.dumps(points))
        return self.status()

    def save_route(self, name):
        name = (name or "").strip()
        if not name or len(name) > 80:
            raise ValueError("Дайте маршруту назву до 80 символів")
        state = stage_store.get_state()
        if not state["points"]:
            raise ValueError("Немає точок для збереження")
        stage_store.save_route(name, state["points"], state["speed_mm_s"])
        with self._lock:
            self._note = "Маршрут «%s» збережено" % name
        return self.status()

    def load_route(self, route_id):
        route = stage_store.get_route(route_id)
        if route is None:
            raise StageError("Маршрут не знайдено")
        stage_store.update_state(
            points_json=json.dumps(route["points"]),
            speed_mm_s=route["speed_mm_s"],
        )
        return self.status()

    def delete_route(self, route_id):
        if not stage_store.delete_route(route_id):
            raise StageError("Маршрут не знайдено")
        return self.status()

    def run_points(self, points=None, speed_mm_s=None):
        state = stage_store.get_state()
        points = state["points"] if points is None else stage_math.clean_points(points)
        if not points:
            raise ValueError("Маршрут порожній")
        speed = state["speed_mm_s"] if speed_mm_s is None else stage_math.clean_speed(speed_mm_s)
        for point in points:
            self._require_inside(point, state["workspace_mm"])
        self._require_motion()
        with self._lock:
            if self._busy:
                raise StageError("Стіл уже їде")
            self._stop.clear()
            self._busy = True
            self._note = "Їде маршрут"
        threading.Thread(
            target=self._run_worker,
            args=(points, speed),
            daemon=True,
        ).start()
        return self.status()

    def _run_worker(self, points, speed):
        try:
            for point in points:
                if self._stop.is_set():
                    break
                self._move_user_to(point)
                if self._stop.wait(SETTLE_SECONDS):
                    break
            with self._lock:
                if self._stop.is_set():
                    self._note = "Маршрут зупинено"
                else:
                    self._note = "Маршрут пройдено"
        except Exception as exc:
            with self._lock:
                self._note = str(exc)
        finally:
            with self._lock:
                self._busy = False

    def _move_user_to(self, target):
        with self._lock:
            if self._stop.is_set():
                return
            state = stage_store.get_state()
            self._require_inside(target, state["workspace_mm"])
            self._require_motion()
            self._set_machine_target(state, target)

    def _set_machine_target(self, state, user_target):
        zero_x = state["zero_mm"]["x"] if state["zero_set"] else 0.0
        zero_y = state["zero_mm"]["y"] if state["zero_set"] else 0.0
        stage_store.update_state(
            machine_x=round(user_target["x"] + zero_x, 4),
            machine_y=round(user_target["y"] + zero_y, 4),
        )

    def _require_motion(self):
        if not self.motion_enabled:
            raise StageError(MOTION_BLOCKED)

    def _require_inside(self, point, workspace):
        if not stage_math.inside_workspace(point["x"], point["y"], workspace):
            raise StageError("Точка поза робочим вікном столу")

    def _user_position(self, state):
        machine = state["machine_mm"]
        if not state["zero_set"]:
            return {"x": machine["x"], "y": machine["y"]}
        return {
            "x": round(machine["x"] - state["zero_mm"]["x"], 4),
            "y": round(machine["y"] - state["zero_mm"]["y"], 4),
        }

    def _voltage_cached(self):
        if self.link is None:
            return None
        now = time.time()
        if self._voltage is not None and now - self._voltage_at < 2:
            return self._voltage
        try:
            raw_x = self.link.read_holding(SLAVE_X, VOLTAGE_REGISTER, 1)[0]
            raw_y = self.link.read_holding(SLAVE_Y, VOLTAGE_REGISTER, 1)[0]
        except (OSError, stage_link.ModbusError):
            self._voltage = None
            return None
        self._voltage = {"x": raw_x / 10.0, "y": raw_y / 10.0}
        self._voltage_at = now
        return self._voltage


_stage = None
_stage_lock = threading.Lock()


def get_stage():
    global _stage
    with _stage_lock:
        if _stage is None:
            port = os.environ.get("TILLETIA_STAGE_PORT", "/dev/ttyUSB0")
            link = None
            if os.path.exists(port):
                link = stage_link.ModbusRtu(port)
            _stage = Stage(link)
        return _stage
