class GridBuilder:
    def __init__(
        self,
        narrow_gap_tolerance_px: float = 10.0,
        wide_gap_tolerance_px: float = 20.0,
        grid_select_min_fraction: float = 0.5,
        track_memory_suppress_min_px: float = 8.0,
        track_memory_suppress_ratio: float = 0.35,
    ) -> None:
        self._narrow_gap_tolerance_px = narrow_gap_tolerance_px
        self._wide_gap_tolerance_px = wide_gap_tolerance_px
        self._grid_select_min_fraction = grid_select_min_fraction
        self._track_memory_suppress_min_px = track_memory_suppress_min_px
        self._track_memory_suppress_ratio = track_memory_suppress_ratio

    def __call__(
        self,
        clustered_lines: list[dict[str, any]],
        gap_info: dict[str, list[float]],
        tracker_state: dict[str, any],
    ) -> dict[str, dict[str, list[dict[str, any]]]]:
        current_grid = self.build_current(clustered_lines, gap_info)
        return self.integrate_predictions(current_grid, gap_info, tracker_state)

    def build_current(
        self,
        clustered_lines: list[dict[str, any]],
        gap_info: dict[str, list[float]],
    ) -> dict[str, dict[str, list[dict[str, any]]]]:
        result: dict[str, dict[str, list[dict[str, any]]]] = {
            "accepted": {"vertical": [], "horizontal": []},
            "rejected": {"vertical": [], "horizontal": []},
        }
        for orientation in ("vertical", "horizontal"):
            lines = [line for line in clustered_lines if line["orientation"] == orientation]
            lines = sorted(lines, key=lambda item: item["rho"])
            used_modes = list(gap_info[orientation])
            selected = self._select(lines, used_modes)
            selected_ids = {id(item) for item in selected}
            rejected = [item for item in lines if id(item) not in selected_ids]
            result["accepted"][orientation] = [self._to_public_line(item, orientation) for item in selected]
            result["rejected"][orientation] = [self._to_public_line(item, orientation) for item in rejected]
        return result

    def integrate_predictions(
        self,
        current_grid: dict[str, dict[str, list[dict[str, any]]]],
        gap_info: dict[str, list[float]],
        tracker_state: dict[str, any],
    ) -> dict[str, dict[str, list[dict[str, any]]]]:
        result: dict[str, dict[str, list[dict[str, any]]]] = {
            "accepted": {"vertical": [], "horizontal": []},
            "rejected": {
                "vertical": [dict(line) for line in current_grid["rejected"]["vertical"]],
                "horizontal": [dict(line) for line in current_grid["rejected"]["horizontal"]],
            },
            "predicted": {"vertical": [], "horizontal": []},
        }
        for orientation in ("vertical", "horizontal"):
            accepted, predicted = self._integrate_predicted(
                tracker_state["predicted"][orientation],
                current_grid["accepted"][orientation],
                list(gap_info[orientation]),
                orientation,
            )
            result["accepted"][orientation] = accepted
            result["predicted"][orientation] = predicted
        return result

    def _select(
        self,
        lines: list[dict[str, any]],
        modes: list[float],
    ) -> list[dict[str, any]]:
        if not lines:
            return []
        if not modes:
            return list(lines)
        if len(lines) < 3:
            return list(lines)
        best_len = [1] * len(lines)
        best_err = [0.0] * len(lines)
        parent = [-1] * len(lines)
        for idx in range(len(lines)):
            for prev_idx in range(idx):
                gap = lines[idx]["rho"] - lines[prev_idx]["rho"]
                target = min(modes, key=lambda value: abs(value - gap))
                err = abs(gap - target)
                if err > self._mode_tolerance(target, modes):
                    continue
                candidate_len = best_len[prev_idx] + 1
                candidate_err = best_err[prev_idx] + err
                if candidate_len > best_len[idx] or (
                    candidate_len == best_len[idx] and candidate_err < best_err[idx]
                ):
                    best_len[idx] = candidate_len
                    best_err[idx] = candidate_err
                    parent[idx] = prev_idx
        best_idx = 0
        for idx in range(1, len(lines)):
            if best_len[idx] > best_len[best_idx]:
                best_idx = idx
            elif best_len[idx] == best_len[best_idx] and best_err[idx] < best_err[best_idx]:
                best_idx = idx
        selected_idx = []
        cursor = best_idx
        while cursor >= 0:
            selected_idx.append(cursor)
            cursor = parent[cursor]
        selected_idx.reverse()
        return [lines[idx] for idx in selected_idx]

    def _integrate_predicted(
        self,
        predicted_lines: list[dict[str, any]],
        accepted_lines: list[dict[str, any]],
        modes: list[float],
        orientation: str,
    ) -> tuple[list[dict[str, any]], list[dict[str, any]]]:
        scene = [self._to_public_line(item, orientation) for item in accepted_lines]
        scene = sorted(scene, key=lambda item: item["rho"])
        if not predicted_lines:
            return scene, []
        if modes:
            suppress_dist = max(
                self._track_memory_suppress_min_px,
                self._track_memory_suppress_ratio * min(modes),
            )
        else:
            suppress_dist = self._track_memory_suppress_min_px
        kept_predicted: list[dict[str, any]] = []
        for line in predicted_lines:
            if any(abs(line["rho"] - item["rho"]) <= suppress_dist for item in scene):
                continue
            if not self._placement_is_consistent(line["rho"], scene, modes):
                kept_predicted.append(line)
                continue
            scene.append(self._to_public_line(line, orientation))
            scene.sort(key=lambda item: item["rho"])
        return scene, kept_predicted

    def _placement_is_consistent(
        self,
        rho: float,
        scene: list[dict[str, any]],
        modes: list[float],
    ) -> bool:
        if not modes or len(scene) < 2:
            return True
        insert_idx = 0
        while insert_idx < len(scene) and scene[insert_idx]["rho"] < rho:
            insert_idx += 1
        left = scene[insert_idx - 1] if insert_idx > 0 else None
        right = scene[insert_idx] if insert_idx < len(scene) else None
        if left is not None and right is not None:
            return self._gap_matches(rho - left["rho"], modes) and self._gap_matches(right["rho"] - rho, modes)
        if left is not None:
            return self._gap_matches(rho - left["rho"], modes)
        if right is not None:
            return self._gap_matches(right["rho"] - rho, modes)
        return True

    def _gap_matches(self, gap: float, modes: list[float]) -> bool:
        if gap <= 0 or not modes:
            return False
        target = min(modes, key=lambda mode: abs(gap - mode))
        return abs(gap - target) <= self._mode_tolerance(target, modes)

    def _mode_tolerance(self, target_mode: float, modes: list[float]) -> float:
        if len(modes) <= 1:
            return self._narrow_gap_tolerance_px
        ordered_modes = sorted(float(mode) for mode in modes)
        if abs(target_mode - ordered_modes[0]) <= abs(target_mode - ordered_modes[-1]):
            return self._narrow_gap_tolerance_px
        return self._wide_gap_tolerance_px

    def _to_public_line(self, item: dict[str, any], orientation: str) -> dict[str, any]:
        return {
            "id": item.get("id"),
            "orientation": orientation,
            "rho": float(item["rho"]),
            "theta": float(item["theta"]),
            "conf": float(item.get("conf", 0.0)),
            "miss": int(item.get("miss", 0)),
        }
