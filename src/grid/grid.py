from __future__ import annotations

import math
from typing import Any


class GridBuilder:
    def __init__(
        self,
        narrow_gap_tolerance_px: float = 10.0,
        wide_gap_tolerance_px: float = 20.0,
        gap_multiple_max: int = 4,
        grid_select_min_fraction: float = 0.5,
        track_memory_suppress_min_px: float = 8.0,
        track_memory_suppress_ratio: float = 0.35,
        slot_match_max_dist_px: float = 28.0,
        slot_rho_smooth: float = 0.65,
        slot_theta_smooth: float = 0.55,
        slot_conf_ema_alpha: float = 0.35,
        slot_conf_decay: float = 0.92,
        slot_max_miss: int = 18,
        slot_min_conf: float = 0.015,
        slot_new_edge_min_conf: float = 0.08,
        predicted_insert_min_conf: float = 0.03,
    ) -> None:
        self._narrow_gap_tolerance_px = narrow_gap_tolerance_px
        self._wide_gap_tolerance_px = wide_gap_tolerance_px
        self._gap_multiple_max = max(int(gap_multiple_max), 1)
        self._grid_select_min_fraction = grid_select_min_fraction
        self._track_memory_suppress_min_px = track_memory_suppress_min_px
        self._track_memory_suppress_ratio = track_memory_suppress_ratio
        self._slot_match_max_dist_px = slot_match_max_dist_px
        self._slot_rho_smooth = slot_rho_smooth
        self._slot_theta_smooth = slot_theta_smooth
        self._slot_conf_ema_alpha = slot_conf_ema_alpha
        self._slot_conf_decay = slot_conf_decay
        self._slot_max_miss = max(int(slot_max_miss), 0)
        self._slot_min_conf = max(float(slot_min_conf), 0.0)
        self._slot_new_edge_min_conf = max(float(slot_new_edge_min_conf), 0.0)
        self._predicted_insert_min_conf = max(float(predicted_insert_min_conf), 0.0)
        self._slots: dict[str, list[dict[str, Any]]] = {
            "vertical": [],
            "horizontal": [],
        }
        self._next_slot_id: dict[str, int] = {
            "vertical": 1,
            "horizontal": 1,
        }

    def __call__(
        self,
        clustered_lines: list[dict[str, Any]],
        gap_info: dict[str, list[float]],
        tracker_state: dict[str, Any],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        current_grid = self.build_current(clustered_lines, gap_info)
        return self.integrate_predictions(current_grid, gap_info, tracker_state)

    def build_current(
        self,
        clustered_lines: list[dict[str, Any]],
        gap_info: dict[str, list[float]],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        result: dict[str, dict[str, list[dict[str, Any]]]] = {
            "accepted": {"vertical": [], "horizontal": []},
            "rejected": {"vertical": [], "horizontal": []},
        }
        for orientation in ("vertical", "horizontal"):
            lines = [
                dict(line)
                for line in clustered_lines
                if line["orientation"] == orientation
            ]
            lines.sort(key=lambda item: item["rho"])
            modes = list(gap_info[orientation])
            if not self._slots[orientation]:
                accepted, rejected = self._bootstrap_orientation(orientation, lines, modes)
            else:
                accepted, rejected = self._match_orientation(orientation, lines, modes)
            result["accepted"][orientation] = accepted
            result["rejected"][orientation] = rejected
        return result

    def integrate_predictions(
        self,
        current_grid: dict[str, dict[str, list[dict[str, Any]]]],
        gap_info: dict[str, list[float]],
        tracker_state: dict[str, Any],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        result: dict[str, dict[str, list[dict[str, Any]]]] = {
            "accepted": {"vertical": [], "horizontal": []},
            "rejected": {
                "vertical": [dict(line) for line in current_grid["rejected"]["vertical"]],
                "horizontal": [dict(line) for line in current_grid["rejected"]["horizontal"]],
            },
            "predicted": {"vertical": [], "horizontal": []},
        }
        for orientation in ("vertical", "horizontal"):
            modes = list(gap_info[orientation])
            reference = [
                dict(line)
                for line in tracker_state["reference"][orientation]
            ]
            reference.sort(key=lambda item: item["rho"])
            if reference:
                scene_base = reference
            else:
                scene_base = [
                    self._to_public_line(item, orientation)
                    for item in current_grid["accepted"][orientation]
                ]
                scene_base.sort(key=lambda item: item["rho"])
            predicted = [
                self._to_public_line(item, orientation)
                for item in tracker_state["predicted"][orientation]
            ]
            predicted.sort(key=lambda item: item["rho"])
            merged, kept_predicted = self._integrate_predicted(
                predicted,
                scene_base,
                modes,
            )
            result["accepted"][orientation] = merged
            result["predicted"][orientation] = kept_predicted
        return result

    def _bootstrap_orientation(
        self,
        orientation: str,
        lines: list[dict[str, Any]],
        modes: list[float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not lines:
            self._slots[orientation] = []
            return [], []
        selected = self._bootstrap_select(lines, modes)
        selected = self._expand_bootstrap_selection(lines, selected, modes)
        selected, _ = self._collapse_close_lines(selected, modes)
        selected_ids = {id(item) for item in selected}
        rejected = [item for item in lines if id(item) not in selected_ids]
        self._slots[orientation] = []
        self._next_slot_id[orientation] = 1
        for item in selected:
            self._create_slot(orientation, item)
        return (
            [self._to_public_line(item, orientation) for item in selected],
            [self._to_public_line(item, orientation) for item in rejected],
        )

    def _bootstrap_select(
        self,
        lines: list[dict[str, Any]],
        modes: list[float],
    ) -> list[dict[str, Any]]:
        if not lines:
            return []
        if not modes or len(lines) < 2:
            return list(lines)

        count = len(lines)
        best_score = [self._line_support(line) for line in lines]
        best_err = [0.0] * count
        best_len = [1] * count
        parent = [-1] * count

        for idx in range(count):
            for prev_idx in range(idx):
                gap = lines[idx]["rho"] - lines[prev_idx]["rho"]
                match = self._closest_gap_match(gap, modes)
                if match is None:
                    continue
                err = match["err"]
                candidate_score = best_score[prev_idx] + self._line_support(lines[idx])
                candidate_len = best_len[prev_idx] + 1
                candidate_err = best_err[prev_idx] + err
                if self._is_better_chain(
                    candidate_score,
                    candidate_len,
                    candidate_err,
                    best_score[idx],
                    best_len[idx],
                    best_err[idx],
                ):
                    best_score[idx] = candidate_score
                    best_len[idx] = candidate_len
                    best_err[idx] = candidate_err
                    parent[idx] = prev_idx

        best_idx = 0
        for idx in range(1, count):
            if self._is_better_chain(
                best_score[idx],
                best_len[idx],
                best_err[idx],
                best_score[best_idx],
                best_len[best_idx],
                best_err[best_idx],
            ):
                best_idx = idx

        selected_idx: list[int] = []
        cursor = best_idx
        while cursor >= 0:
            selected_idx.append(cursor)
            cursor = parent[cursor]
        selected_idx.reverse()
        if not selected_idx:
            return []

        selected = [lines[idx] for idx in selected_idx]
        min_required = max(2, int(math.ceil(len(lines) * self._grid_select_min_fraction)))
        if len(selected) < min_required and len(lines) <= 6:
            return list(lines)
        return selected

    def _expand_bootstrap_selection(
        self,
        lines: list[dict[str, Any]],
        selected: list[dict[str, Any]],
        modes: list[float],
    ) -> list[dict[str, Any]]:
        if not lines:
            return []
        if not selected or not modes:
            return list(selected)

        scene = sorted(list(selected), key=lambda item: item["rho"])
        selected_ids = {id(item) for item in scene}
        remaining = [item for item in lines if id(item) not in selected_ids]

        def _remaining_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(items, key=lambda item: (-self._line_support(item), float(item["rho"])))

        remaining = _remaining_order(remaining)
        while remaining:
            next_remaining: list[dict[str, Any]] = []
            added = False
            for item in remaining:
                if self._candidate_fits_scene(item, scene, modes):
                    scene.append(item)
                    scene.sort(key=lambda line: line["rho"])
                    added = True
                else:
                    next_remaining.append(item)
            if not added:
                break
            remaining = _remaining_order(next_remaining)

        return scene

    def _match_orientation(
        self,
        orientation: str,
        lines: list[dict[str, Any]],
        modes: list[float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._slots[orientation] = sorted(self._slots[orientation], key=lambda item: item["rho"])
        slots = self._slots[orientation]
        if not slots:
            return self._bootstrap_orientation(orientation, lines, modes)

        match_dist = self._slot_match_distance(modes)
        matches, matched_slot_indices, matched_line_indices = self._match_slots_to_lines(
            slots,
            lines,
            match_dist,
        )

        accepted: list[dict[str, Any]] = []
        for slot_idx, line_idx in matches:
            slot = slots[slot_idx]
            line = lines[line_idx]
            self._update_slot_from_line(slot, line)
            accepted.append(self._to_public_line(line, orientation))

        for slot_idx, slot in enumerate(slots):
            if slot_idx in matched_slot_indices:
                continue
            self._age_slot(slot)

        added_edge_indices = self._append_edge_slots(
            orientation,
            lines,
            sorted(idx for idx in range(len(lines)) if idx not in matched_line_indices),
            accepted,
            modes,
        )
        self._collapse_close_slots(orientation, modes)
        accepted, demoted = self._collapse_close_public_lines(accepted, modes)
        rejected = [
            self._to_public_line(lines[idx], orientation)
            for idx in range(len(lines))
            if idx not in matched_line_indices and idx not in added_edge_indices
        ]
        rejected.extend(demoted)

        self._prune_slots(orientation)
        accepted.sort(key=lambda item: item["rho"])
        rejected.sort(key=lambda item: item["rho"])
        return accepted, rejected

    def _match_slots_to_lines(
        self,
        slots: list[dict[str, Any]],
        lines: list[dict[str, Any]],
        match_dist: float,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not slots or not lines:
            return [], set(), set()

        slot_count = len(slots)
        line_count = len(lines)
        dp_score = [[0.0] * (line_count + 1) for _ in range(slot_count + 1)]
        dp_matches = [[0] * (line_count + 1) for _ in range(slot_count + 1)]
        choice = [[0] * (line_count + 1) for _ in range(slot_count + 1)]

        for i in range(1, slot_count + 1):
            slot = slots[i - 1]
            slot_skip_penalty = self._skip_slot_penalty(slot)
            for j in range(1, line_count + 1):
                line = lines[j - 1]
                best_score = dp_score[i - 1][j] - slot_skip_penalty
                best_matches = dp_matches[i - 1][j]
                best_choice = 1

                skip_line_score = dp_score[i][j - 1] - self._skip_line_penalty(line)
                skip_line_matches = dp_matches[i][j - 1]
                if self._is_better_match_state(
                    skip_line_score,
                    skip_line_matches,
                    best_score,
                    best_matches,
                ):
                    best_score = skip_line_score
                    best_matches = skip_line_matches
                    best_choice = 2

                dist = abs(float(slot["rho"]) - float(line["rho"]))
                if dist <= match_dist:
                    match_score = dp_score[i - 1][j - 1] + self._match_reward(slot, line, dist)
                    match_matches = dp_matches[i - 1][j - 1] + 1
                    if self._is_better_match_state(
                        match_score,
                        match_matches,
                        best_score,
                        best_matches,
                    ):
                        best_score = match_score
                        best_matches = match_matches
                        best_choice = 3

                dp_score[i][j] = best_score
                dp_matches[i][j] = best_matches
                choice[i][j] = best_choice

        matches: list[tuple[int, int]] = []
        i = slot_count
        j = line_count
        while i > 0 and j > 0:
            decision = choice[i][j]
            if decision == 3:
                matches.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif decision == 2:
                j -= 1
            else:
                i -= 1
        matches.reverse()
        matched_slot_indices = {slot_idx for slot_idx, _ in matches}
        matched_line_indices = {line_idx for _, line_idx in matches}
        return matches, matched_slot_indices, matched_line_indices

    def _integrate_predicted(
        self,
        predicted_lines: list[dict[str, Any]],
        accepted_lines: list[dict[str, Any]],
        modes: list[float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scene = [dict(line) for line in accepted_lines]
        scene.sort(key=lambda item: item["rho"])
        if not predicted_lines:
            return scene, []

        suppress_dist = self._band_merge_distance(modes)

        kept_predicted: list[dict[str, Any]] = []
        for line in predicted_lines:
            public_line = dict(line)
            if any(abs(public_line["rho"] - item["rho"]) <= suppress_dist for item in scene):
                continue
            if public_line["conf"] < self._predicted_insert_min_conf:
                kept_predicted.append(public_line)
                continue
            if modes and not self._candidate_fits_scene(public_line, scene, modes):
                kept_predicted.append(public_line)
                continue
            scene.append(public_line)
            scene.sort(key=lambda item: item["rho"])
        return scene, kept_predicted

    def _append_edge_slots(
        self,
        orientation: str,
        lines: list[dict[str, Any]],
        candidate_indices: list[int],
        accepted: list[dict[str, Any]],
        modes: list[float],
    ) -> set[int]:
        if not candidate_indices:
            return set()

        scene = [dict(line) for line in accepted]
        scene.sort(key=lambda item: item["rho"])
        added_indices: set[int] = set()

        for line_idx in sorted(candidate_indices, key=lambda idx: lines[idx]["rho"]):
            line = lines[line_idx]
            if self._line_support(line) < self._slot_new_edge_min_conf:
                continue
            public_line = self._to_public_line(line, orientation)
            if not scene:
                self._create_slot(orientation, line)
                scene.append(public_line)
                added_indices.add(line_idx)
                continue
            if any(
                abs(public_line["rho"] - item["rho"]) <= self._track_memory_suppress_min_px
                for item in scene
            ):
                continue
            left_edge = scene[0]
            right_edge = scene[-1]
            rho = public_line["rho"]
            if rho < left_edge["rho"]:
                if self._edge_gap_is_consistent(left_edge["rho"] - rho, modes):
                    self._create_slot(orientation, line)
                    scene.append(public_line)
                    scene.sort(key=lambda item: item["rho"])
                    added_indices.add(line_idx)
                continue
            if rho > right_edge["rho"]:
                if self._edge_gap_is_consistent(rho - right_edge["rho"], modes):
                    self._create_slot(orientation, line)
                    scene.append(public_line)
                    scene.sort(key=lambda item: item["rho"])
                    added_indices.add(line_idx)

        accepted[:] = scene
        return added_indices

    def _edge_gap_is_consistent(self, gap: float, modes: list[float]) -> bool:
        if not modes:
            return True
        return self._gap_matches(gap, modes)

    def _gap_matches(self, gap: float, modes: list[float]) -> bool:
        return self._closest_gap_match(gap, modes) is not None

    def _closest_gap_match(
        self,
        gap: float,
        modes: list[float],
    ) -> dict[str, float | int] | None:
        if gap <= 0.0 or not modes:
            return None

        best_match: dict[str, float | int] | None = None
        for mode in sorted(float(value) for value in modes if value > 0.0):
            base_tolerance = self._mode_tolerance(mode, modes)
            for multiple in range(1, self._gap_multiple_max + 1):
                target = float(multiple) * mode
                tolerance = max(base_tolerance * float(multiple), 0.12 * target)
                err = abs(gap - target)
                if err > tolerance:
                    continue
                candidate = {
                    "mode": mode,
                    "multiple": multiple,
                    "target": target,
                    "err": err,
                    "tolerance": tolerance,
                }
                if best_match is None:
                    best_match = candidate
                    continue
                if candidate["err"] < float(best_match["err"]) - 1e-6:
                    best_match = candidate
                    continue
                if (
                    abs(candidate["err"] - float(best_match["err"])) <= 1e-6
                    and int(candidate["multiple"]) < int(best_match["multiple"])
                ):
                    best_match = candidate
        return best_match

    def _candidate_fits_scene(
        self,
        line: dict[str, Any],
        scene: list[dict[str, Any]],
        modes: list[float],
    ) -> bool:
        if not scene or not modes:
            return True

        ordered = sorted(scene, key=lambda item: item["rho"])
        rho = float(line["rho"])
        insert_at = 0
        while insert_at < len(ordered) and float(ordered[insert_at]["rho"]) < rho:
            insert_at += 1

        if insert_at > 0:
            left_gap = rho - float(ordered[insert_at - 1]["rho"])
            if left_gap > 0.0 and not self._gap_matches(left_gap, modes):
                return False

        if insert_at < len(ordered):
            right_gap = float(ordered[insert_at]["rho"]) - rho
            if right_gap > 0.0 and not self._gap_matches(right_gap, modes):
                return False

        return True

    def _mode_tolerance(self, target_mode: float, modes: list[float]) -> float:
        if len(modes) <= 1:
            return self._narrow_gap_tolerance_px
        ordered_modes = sorted(float(mode) for mode in modes)
        if abs(target_mode - ordered_modes[0]) <= abs(target_mode - ordered_modes[-1]):
            return self._narrow_gap_tolerance_px
        return self._wide_gap_tolerance_px

    def _slot_match_distance(self, modes: list[float]) -> float:
        if not modes:
            return self._slot_match_max_dist_px
        return max(self._slot_match_max_dist_px, 0.55 * min(modes))

    def _band_merge_distance(self, modes: list[float]) -> float:
        if not modes:
            return self._track_memory_suppress_min_px
        return max(
            self._track_memory_suppress_min_px,
            self._track_memory_suppress_ratio * min(modes),
        )

    def _collapse_close_lines(
        self,
        lines: list[dict[str, Any]],
        modes: list[float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if len(lines) < 2:
            return list(lines), []
        merge_dist = self._band_merge_distance(modes)
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for group in self._group_close_items(lines, merge_dist):
            if len(group) == 1:
                kept.append(group[0])
                continue
            best = max(group, key=self._line_support)
            kept.append(best)
            dropped.extend(item for item in group if item is not best)
        kept.sort(key=lambda item: item["rho"])
        dropped.sort(key=lambda item: item["rho"])
        return kept, dropped

    def _collapse_close_public_lines(
        self,
        lines: list[dict[str, Any]],
        modes: list[float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if len(lines) < 2:
            return list(lines), []
        merge_dist = self._band_merge_distance(modes)
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for group in self._group_close_items(lines, merge_dist):
            if len(group) == 1:
                kept.append(group[0])
                continue
            best = max(group, key=self._public_line_rank)
            kept.append(best)
            dropped.extend(dict(item) for item in group if item is not best)
        kept.sort(key=lambda item: item["rho"])
        dropped.sort(key=lambda item: item["rho"])
        return kept, dropped

    def _collapse_close_slots(self, orientation: str, modes: list[float]) -> None:
        slots = self._slots[orientation]
        if len(slots) < 2:
            return

        merge_dist = self._band_merge_distance(modes)
        merged_slots: list[dict[str, Any]] = []
        for group in self._group_close_items(slots, merge_dist):
            if len(group) == 1:
                merged_slots.append(group[0])
                continue

            best = max(group, key=self._slot_rank)
            weights = [
                max(float(item["conf_ema"]), float(item["last_conf"]), self._slot_min_conf, 1e-3)
                for item in group
            ]
            total_weight = sum(weights)
            best["rho"] = sum(
                weight * float(item["rho"])
                for item, weight in zip(group, weights)
            ) / total_weight
            best["theta"] = self._angle_average(
                [float(item["theta"]) for item in group],
                weights,
            )
            best["conf_ema"] = max(float(item["conf_ema"]) for item in group)
            best["last_conf"] = max(float(item["last_conf"]) for item in group)
            best["hits"] = max(int(item["hits"]) for item in group)
            best["miss"] = min(int(item["miss"]) for item in group)
            merged_slots.append(best)

        self._slots[orientation] = sorted(merged_slots, key=lambda item: item["rho"])

    def _group_close_items(
        self,
        items: list[dict[str, Any]],
        merge_dist: float,
    ) -> list[list[dict[str, Any]]]:
        if not items:
            return []

        ordered = sorted(items, key=lambda item: item["rho"])
        groups: list[list[dict[str, Any]]] = [[ordered[0]]]
        for item in ordered[1:]:
            prev = groups[-1][-1]
            if float(item["rho"]) - float(prev["rho"]) <= merge_dist:
                groups[-1].append(item)
            else:
                groups.append([item])
        return groups

    def _match_reward(
        self,
        slot: dict[str, Any],
        line: dict[str, Any],
        dist: float,
    ) -> float:
        return (
            2.0 * self._line_support(line)
            + 0.35 * float(slot["conf_ema"])
            - 0.06 * float(dist)
        )

    def _skip_slot_penalty(self, slot: dict[str, Any]) -> float:
        return 0.35 * max(float(slot["conf_ema"]), float(slot["last_conf"]))

    def _skip_line_penalty(self, line: dict[str, Any]) -> float:
        return 0.05 * self._line_support(line)

    def _public_line_rank(self, line: dict[str, Any]) -> float:
        return self._line_support(line) - 0.03 * float(line.get("miss", 0))

    def _slot_rank(self, slot: dict[str, Any]) -> float:
        return (
            max(float(slot["conf_ema"]), float(slot["last_conf"]))
            + 0.02 * min(int(slot["hits"]), 10)
            - 0.03 * int(slot["miss"])
        )

    def _is_better_match_state(
        self,
        cand_score: float,
        cand_matches: int,
        best_score: float,
        best_matches: int,
    ) -> bool:
        if cand_score > best_score + 1e-6:
            return True
        if cand_score < best_score - 1e-6:
            return False
        return cand_matches > best_matches

    def _is_better_chain(
        self,
        cand_score: float,
        cand_len: int,
        cand_err: float,
        best_score: float,
        best_len: int,
        best_err: float,
    ) -> bool:
        if cand_score > best_score + 1e-6:
            return True
        if cand_score < best_score - 1e-6:
            return False
        if cand_len != best_len:
            return cand_len > best_len
        return cand_err < best_err - 1e-6

    def _create_slot(
        self,
        orientation: str,
        line: dict[str, Any],
    ) -> dict[str, Any]:
        slot = {
            "slot_id": self._next_slot_id[orientation],
            "orientation": orientation,
            "rho": float(line["rho"]),
            "theta": float(line["theta"]),
            "conf_ema": self._line_support(line),
            "last_conf": self._line_support(line),
            "hits": 1,
            "miss": 0,
        }
        self._next_slot_id[orientation] += 1
        self._slots[orientation].append(slot)
        self._slots[orientation].sort(key=lambda item: item["rho"])
        return slot

    def _update_slot_from_line(
        self,
        slot: dict[str, Any],
        line: dict[str, Any],
    ) -> None:
        slot["rho"] = (
            (1.0 - self._slot_rho_smooth) * float(slot["rho"])
            + self._slot_rho_smooth * float(line["rho"])
        )
        slot["theta"] = self._angle_blend(
            float(slot["theta"]),
            float(line["theta"]),
            self._slot_theta_smooth,
        )
        line_conf = self._line_support(line)
        slot["conf_ema"] = (
            (1.0 - self._slot_conf_ema_alpha) * float(slot["conf_ema"])
            + self._slot_conf_ema_alpha * line_conf
        )
        slot["last_conf"] = line_conf
        slot["hits"] = int(slot["hits"]) + 1
        slot["miss"] = 0

    def _age_slot(self, slot: dict[str, Any]) -> None:
        slot["miss"] = int(slot["miss"]) + 1
        slot["last_conf"] *= self._slot_conf_decay
        slot["conf_ema"] *= self._slot_conf_decay

    def _prune_slots(self, orientation: str) -> None:
        self._slots[orientation] = [
            slot
            for slot in self._slots[orientation]
            if int(slot["miss"]) <= self._slot_max_miss
            and (
                int(slot["hits"]) >= 2
                or int(slot["miss"]) <= 2
                or float(slot["conf_ema"]) >= self._slot_min_conf
            )
        ]
        self._slots[orientation].sort(key=lambda item: item["rho"])

    def _line_support(self, item: dict[str, Any]) -> float:
        return max(float(item.get("conf", 0.0)), 0.0)

    def _angle_blend(
        self,
        theta_old: float,
        theta_new: float,
        alpha: float,
    ) -> float:
        c_old = math.cos(2.0 * theta_old)
        s_old = math.sin(2.0 * theta_old)
        c_new = math.cos(2.0 * theta_new)
        s_new = math.sin(2.0 * theta_new)
        c_mix = (1.0 - alpha) * c_old + alpha * c_new
        s_mix = (1.0 - alpha) * s_old + alpha * s_new
        return 0.5 * math.atan2(s_mix, c_mix)

    def _angle_average(
        self,
        angles: list[float],
        weights: list[float],
    ) -> float:
        c_sum = 0.0
        s_sum = 0.0
        for angle, weight in zip(angles, weights):
            c_sum += weight * math.cos(2.0 * angle)
            s_sum += weight * math.sin(2.0 * angle)
        return 0.5 * math.atan2(s_sum, c_sum)

    def _to_public_line(self, item: dict[str, Any], orientation: str) -> dict[str, Any]:
        return {
            "id": item.get("id"),
            "orientation": orientation,
            "rho": float(item["rho"]),
            "theta": float(item["theta"]),
            "conf": float(
                item.get(
                    "conf",
                    item.get("last_conf", item.get("conf_ema", 0.0)),
                )
            ),
            "miss": int(item.get("miss", 0)),
        }
