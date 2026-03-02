import math
import cv2
import numpy as np


class GridTracker:
    """Track grid lines across frames using global optical-flow motion and line matching."""

    def __init__(
        self,
        track_match_max_dist_px: float = 18.0,
        track_shift_max_px: float = 45.0,
        track_max_miss: int = 15,
        track_rho_smooth: float = 0.65,
        track_theta_smooth: float = 0.55,
        track_conf_decay: float = 0.92,
        track_min_conf: float = 0.02,
        track_warmup_frames: int = 3,
        track_memory_max_miss: int = 15,
        track_memory_min_conf: float = 0.0,
        track_align_max_corr_px: float = 6.0,
        track_align_min_cands: int = 2,
        track_flow_min_side_px: int = 320,
    ) -> None:
        """
        Initialize the grid tracker.

        Args:
            track_match_max_dist_px: Maximum allowed distance, in pixels,
                between a predicted track position and a current candidate for
                the pair to be considered a valid match.
            track_shift_max_px: Maximum absolute global shift, in pixels,
                applied from optical flow in a single frame.
            track_max_miss: Maximum number of consecutive missed frames a track
                may survive before it is removed.
            track_rho_smooth: Smoothing factor used when updating the tracked
                line position from the predicted position and the matched
                candidate.
            track_theta_smooth: Smoothing factor used when updating the tracked
                line angle from the previous track angle and the matched
                candidate angle.
            track_conf_decay: Multiplicative confidence decay applied when a
                track is not matched in the current frame.
            track_min_conf: Minimum confidence required for a track to appear
                in the reference output.
            track_warmup_frames: Number of confirmations required before a
                track is treated as fully established.
            track_memory_max_miss: Maximum miss count for a track to remain
                eligible for the predicted carry-over output.
            track_memory_min_conf: Minimum confidence required for a missed
                track to be emitted as a predicted line.
            track_align_max_corr_px: Maximum absolute correction, in pixels,
                allowed for the tracker-to-candidate alignment shift.
            track_align_min_cands: Minimum number of current candidates needed
                before alignment correction is estimated.
            track_flow_min_side_px: Short-side resolution used for the
                optical-flow pass. The estimated motion is scaled back to the
                analysis frame size before it is applied to tracks.
        """
        self._track_match_max_dist_px = track_match_max_dist_px
        self._track_shift_max_px = track_shift_max_px
        self._track_max_miss = track_max_miss
        self._track_rho_smooth = track_rho_smooth
        self._track_theta_smooth = track_theta_smooth
        self._track_conf_decay = track_conf_decay
        self._track_min_conf = track_min_conf
        self._track_warmup_frames = track_warmup_frames
        self._track_memory_max_miss = track_memory_max_miss
        self._track_memory_min_conf = track_memory_min_conf
        self._track_align_max_corr_px = track_align_max_corr_px
        self._track_align_min_cands = track_align_min_cands
        self._track_flow_min_side_px = max(int(track_flow_min_side_px), 0)
        self._state: dict[str, dict[str, any]] = {
            "vertical": {"tracks": [], "next_id": 1, "frames_seen": 0},
            "horizontal": {"tracks": [], "next_id": 1, "frames_seen": 0},
        }
        self._prev_gray: np.ndarray | None = None
        self._last_debug: dict[str, any] = {}
        self._last_flow_debug: dict[str, any] = {}

    @property
    def warmup_frames(self) -> int:
        return int(self._track_warmup_frames)

    def estimate_flow_shift(self, analysis_gray: np.ndarray) -> dict[str, float]:
        analysis_h, analysis_w = analysis_gray.shape[:2]
        flow_gray, flow_scale_x, flow_scale_y = self._prepare_flow_frame(analysis_gray)
        flow_dx_raw, flow_dy_raw, flow_debug = self._estimate_optical_flow_shift(self._prev_gray, flow_gray)
        flow_dx_scaled = flow_dx_raw * flow_scale_x
        flow_dy_scaled = flow_dy_raw * flow_scale_y
        flow_dx = self._clamp(flow_dx_scaled, -self._track_shift_max_px, self._track_shift_max_px)
        flow_dy = self._clamp(flow_dy_scaled, -self._track_shift_max_px, self._track_shift_max_px)
        self._prev_gray = flow_gray.copy()
        flow_h, flow_w = flow_gray.shape[:2]
        self._last_flow_debug = {
            "analysis_shape": {"height": int(analysis_h), "width": int(analysis_w)},
            "flow_shape": {"height": int(flow_h), "width": int(flow_w)},
            "scale_x": float(flow_scale_x),
            "scale_y": float(flow_scale_y),
            "raw_dx": float(flow_dx_raw),
            "raw_dy": float(flow_dy_raw),
            "scaled_dx_before_clamp": float(flow_dx_scaled),
            "scaled_dy_before_clamp": float(flow_dy_scaled),
            "dx": float(flow_dx),
            "dy": float(flow_dy),
            **flow_debug,
        }
        return {"dx": float(flow_dx), "dy": float(flow_dy)}

    def __call__(
        self,
        clustered_lines: list[dict[str, any]],
        analysis_gray: np.ndarray | None = None,
        flow_shift: dict[str, float] | None = None,
    ) -> dict[str, any]:
        if flow_shift is None:
            if analysis_gray is None:
                raise ValueError("analysis_gray is required when flow_shift is not provided.")
            flow_data = self.estimate_flow_shift(analysis_gray)
        else:
            flow_data = {
                "dx": float(flow_shift.get("dx", 0.0)),
                "dy": float(flow_shift.get("dy", 0.0)),
            }
        flow_dx = float(flow_data["dx"])
        flow_dy = float(flow_data["dy"])
        predicted: dict[str, list[dict[str, any]]] = {}
        reference: dict[str, list[dict[str, any]]] = {}
        call_debug = {
            "flow_shift": {"dx": float(flow_dx), "dy": float(flow_dy)},
            "flow_estimation": dict(self._last_flow_debug),
            "orientations": {},
        }
        
        for orientation in ("vertical", "horizontal"):
            lines = [line for line in clustered_lines if line["orientation"] == orientation]
            flow_shift = flow_dx if orientation == "vertical" else flow_dy
            updated = self._update_orientation(orientation, lines, flow_shift)
            predicted[orientation] = updated["predicted"]
            reference[orientation] = updated["reference"]
            call_debug["orientations"][orientation] = updated["debug"]
        
        self._last_debug = call_debug
        
        return {
            "predicted": predicted,
            "reference": reference,
            "flow_shift": {"dx": float(flow_dx), "dy": float(flow_dy)},
            "debug": call_debug,
        }

    def debug_snapshot(self) -> dict[str, any]:
        return {
            "last_call": dict(self._last_debug),
            "state": {
                "vertical": self._snapshot_orientation_state("vertical"),
                "horizontal": self._snapshot_orientation_state("horizontal"),
            },
        }

    def _prepare_flow_frame(
        self,
        analysis_gray: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        height, width = analysis_gray.shape[:2]
        short_side = min(height, width)
        target_min_side = self._track_flow_min_side_px

        if target_min_side <= 0 or short_side <= 0 or short_side <= target_min_side:
            return analysis_gray, 1.0, 1.0

        scale = float(target_min_side) / float(short_side)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        resized = cv2.resize(analysis_gray, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        scale_x = float(width) / float(resized_w)
        scale_y = float(height) / float(resized_h)
        return resized, scale_x, scale_y

    def _update_orientation(
        self,
        orientation: str,
        candidates: list[dict[str, any]],
        flow_shift: float,
    ) -> dict[str, list[dict[str, any]]]:
        st = self._state[orientation]
        st["frames_seen"] += 1
        tracks = st["tracks"]
        original_track_count = len(tracks)
        tracks_before = self._snapshot_tracks(tracks)
        candidates_debug = [self._public_candidate(candidate, idx) for idx, candidate in enumerate(candidates)]
        
        for track in tracks:
            track["rho_pred"] = track["rho"] + self._clamp(
                float(flow_shift),
                -self._track_shift_max_px,
                self._track_shift_max_px,
            )
        tracks_after_flow = self._snapshot_tracks(tracks)
        
        align_shift = self._estimate_alignment_delta(tracks, candidates)
        if abs(align_shift) > 1e-6:
            for track in tracks:
                track["rho_pred"] += align_shift
        tracks_after_alignment = self._snapshot_tracks(tracks)
        
        matches, used_tracks, used_candidates = self._match(tracks, candidates)
        match_debug = []
        for track_idx, candidate_idx in matches:
            track = tracks[track_idx]
            candidate = candidates[candidate_idx]
            pred_rho = track.get("rho_pred", track["rho"])
            match_debug.append(
                {
                    "track_index": int(track_idx),
                    "track_id": int(track["id"]),
                    "candidate_index": int(candidate_idx),
                    "track_rho_pred": float(pred_rho),
                    "candidate_rho": float(candidate["rho"]),
                    "delta": float(candidate["rho"] - pred_rho),
                }
            )
            track["rho"] = (1.0 - self._track_rho_smooth) * pred_rho + self._track_rho_smooth * candidate["rho"]
            track["theta"] = self._angle_blend(track["theta"], candidate["theta"], self._track_theta_smooth)
            track["conf"] = 0.7 * track["conf"] + 0.3 * candidate["conf"]
            track["miss"] = 0
            track["hits"] += 1
        
        for idx, track in enumerate(tracks):
            if idx in used_tracks:
                continue
            track["rho"] = track.get("rho_pred", track["rho"])
            track["miss"] += 1
            track["conf"] *= self._track_conf_decay
        
        for idx, candidate in enumerate(candidates):
            if idx in used_candidates:
                continue
            new_track_id = st["next_id"]
            tracks.append(
                {
                    "id": new_track_id,
                    "orientation": orientation,
                    "rho": float(candidate["rho"]),
                    "theta": float(candidate["theta"]),
                    "conf": float(candidate["conf"]),
                    "hits": 1,
                    "miss": 0,
                    "rho_pred": float(candidate["rho"]),
                }
            )
            st["next_id"] += 1
        
        tracks_before_prune = self._snapshot_tracks(tracks)
        
        kept_tracks = [
            track
            for track in tracks
            if track["miss"] <= self._track_max_miss
            and (track["hits"] >= self._track_warmup_frames or track["miss"] == 0)
        ]
        kept_ids = {int(track["id"]) for track in kept_tracks}
        dropped_track_ids = [
            int(track["id"]) for track in tracks if int(track["id"]) not in kept_ids
        ]
        st["tracks"] = kept_tracks
        
        warming_up = st["frames_seen"] <= self._track_warmup_frames
        reference_tracks = [
            track.copy()
            for track in sorted(st["tracks"], key=lambda item: item["rho"])
            if track["miss"] <= 1
            and track["conf"] >= self._track_min_conf
            and (warming_up or track["hits"] >= self._track_warmup_frames)
        ]
        
        if warming_up:
            predicted_tracks = []
        else:
            predicted_tracks = [
                track.copy()
                for track in sorted(st["tracks"], key=lambda item: item["rho"])
                if 0 < track["miss"] <= self._track_memory_max_miss
                and track["conf"] >= self._track_memory_min_conf
                and track["hits"] >= self._track_warmup_frames
            ]
        matched_track_ids = [int(tracks[track_idx]["id"]) for track_idx, _ in matches]
        unmatched_track_ids = [
            int(tracks[idx]["id"]) for idx in range(original_track_count) if idx not in used_tracks
        ]
        debug_info = {
            "orientation": orientation,
            "frames_seen": int(st["frames_seen"]),
            "warming_up": bool(warming_up),
            "flow_shift": float(flow_shift),
            "align_shift": float(align_shift),
            "candidates": candidates_debug,
            "tracks_before": tracks_before,
            "tracks_after_flow": tracks_after_flow,
            "tracks_after_alignment": tracks_after_alignment,
            "matches": match_debug,
            "matched_track_ids": matched_track_ids,
            "matched_candidate_indices": sorted(int(idx) for idx in used_candidates),
            "unmatched_track_ids": unmatched_track_ids,
            "unmatched_candidate_indices": [
                int(idx) for idx in range(len(candidates)) if idx not in used_candidates
            ],
            "tracks_before_prune": tracks_before_prune,
            "dropped_track_ids": dropped_track_ids,
            "tracks_after": self._snapshot_tracks(st["tracks"]),
            "reference_output": [self._public_track(track) for track in reference_tracks],
            "predicted_output": [self._public_track(track) for track in predicted_tracks],
        }
        return {"reference": reference_tracks, "predicted": predicted_tracks, "debug": debug_info}

    def _estimate_alignment_delta(
        self,
        tracks: list[dict[str, any]],
        candidates: list[dict[str, any]],
    ) -> float:
        if not tracks or len(candidates) < self._track_align_min_cands:
            return 0.0
        prelim_matches, _, _ = self._match(
            tracks,
            candidates,
            max_dist_px=self._track_match_max_dist_px + self._track_align_max_corr_px,
        )
        deltas = [
            float(candidates[candidate_idx]["rho"] - tracks[track_idx].get("rho_pred", tracks[track_idx]["rho"]))
            for track_idx, candidate_idx in prelim_matches
        ]
        if len(deltas) < self._track_align_min_cands:
            return 0.0
        
        delta = self._median_value(deltas)
        return self._clamp(delta, -self._track_align_max_corr_px, self._track_align_max_corr_px)

    def _match(
        self,
        tracks: list[dict[str, any]],
        candidates: list[dict[str, any]],
        max_dist_px: float | None = None,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not tracks or not candidates:
            return [], set(), set()

        dist_limit = self._track_match_max_dist_px if max_dist_px is None else float(max_dist_px)
        track_order = sorted(
            range(len(tracks)),
            key=lambda idx: tracks[idx].get("rho_pred", tracks[idx]["rho"]),
        )
        candidate_order = sorted(
            range(len(candidates)),
            key=lambda idx: candidates[idx]["rho"],
        )
        track_count = len(track_order)
        candidate_count = len(candidate_order)
        dp_matches = [[0] * (candidate_count + 1) for _ in range(track_count + 1)]
        dp_cost = [[0.0] * (candidate_count + 1) for _ in range(track_count + 1)]
        choice = [[0] * (candidate_count + 1) for _ in range(track_count + 1)]

        for i in range(1, track_count + 1):
            for j in range(1, candidate_count + 1):
                best_matches = dp_matches[i - 1][j]
                best_cost = dp_cost[i - 1][j]
                best_choice = 1

                skip_candidate_matches = dp_matches[i][j - 1]
                skip_candidate_cost = dp_cost[i][j - 1]
                if self._is_better_match_state(
                    skip_candidate_matches,
                    skip_candidate_cost,
                    best_matches,
                    best_cost,
                ):
                    best_matches = skip_candidate_matches
                    best_cost = skip_candidate_cost
                    best_choice = 2

                track_idx = track_order[i - 1]
                candidate_idx = candidate_order[j - 1]
                pred_rho = tracks[track_idx].get("rho_pred", tracks[track_idx]["rho"])
                dist = abs(pred_rho - candidates[candidate_idx]["rho"])
                if dist <= dist_limit:
                    match_matches = dp_matches[i - 1][j - 1] + 1
                    match_cost = dp_cost[i - 1][j - 1] + dist
                    if self._is_better_match_state(
                        match_matches,
                        match_cost,
                        best_matches,
                        best_cost,
                    ):
                        best_matches = match_matches
                        best_cost = match_cost
                        best_choice = 3

                dp_matches[i][j] = best_matches
                dp_cost[i][j] = best_cost
                choice[i][j] = best_choice

        matches = []
        i = track_count
        j = candidate_count
        while i > 0 and j > 0:
            decision = choice[i][j]
            if decision == 3:
                matches.append((track_order[i - 1], candidate_order[j - 1]))
                i -= 1
                j -= 1
            elif decision == 2:
                j -= 1
            else:
                i -= 1

        matches.reverse()
        used_tracks = {track_idx for track_idx, _ in matches}
        used_candidates = {candidate_idx for _, candidate_idx in matches}
        return matches, used_tracks, used_candidates

    def _is_better_match_state(
        self,
        cand_matches: int,
        cand_cost: float,
        best_matches: int,
        best_cost: float,
    ) -> bool:
        if cand_matches != best_matches:
            return cand_matches > best_matches
        return cand_cost < best_cost - 1e-6

    def _estimate_optical_flow_shift(
        self,
        prev_gray: np.ndarray | None,
        curr_gray: np.ndarray | None,
    ) -> tuple[float, float, dict[str, any]]:
        if prev_gray is None or curr_gray is None:
            return 0.0, 0.0, {"status": "missing_previous_frame", "feature_count": 0, "tracked_count": 0}
        
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        
        feature_count = 0 if prev_pts is None else int(len(prev_pts))
        if prev_pts is None or feature_count < 16:
            return (
                0.0,
                0.0,
                {
                    "status": "insufficient_features",
                    "feature_count": feature_count,
                    "tracked_count": 0,
                },
            )
        
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        
        if curr_pts is None or status is None:
            return (
                0.0,
                0.0,
                {
                    "status": "tracking_failed",
                    "feature_count": feature_count,
                    "tracked_count": 0,
                },
            )
        mask = status.reshape(-1).astype(bool)
        tracked_prev = prev_pts.reshape(-1, 2)[mask]
        tracked_curr = curr_pts.reshape(-1, 2)[mask]
        tracked_count = int(len(tracked_prev))
        
        if tracked_count < 16:
            return (
                0.0,
                0.0,
                {
                    "status": "insufficient_tracked_points",
                    "feature_count": feature_count,
                    "tracked_count": tracked_count,
                },
            )
        
        affine, _ = cv2.estimateAffinePartial2D(
            tracked_prev,
            tracked_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        
        if affine is not None:
            dx = float(affine[0, 2])
            dy = float(affine[1, 2])
            method = "affine_partial_2d"
        else:
            dx = self._median_value([float(curr[0] - prev[0]) for prev, curr in zip(tracked_prev, tracked_curr)])
            dy = self._median_value([float(curr[1] - prev[1]) for prev, curr in zip(tracked_prev, tracked_curr)])
            method = "median_delta"
        
        return (
            dx,
            dy,
            {
                "status": "ok",
                "feature_count": feature_count,
                "tracked_count": tracked_count,
                "method": method,
            },
        )

    def _snapshot_orientation_state(self, orientation: str) -> dict[str, any]:
        st = self._state[orientation]
        return {
            "frames_seen": int(st["frames_seen"]),
            "next_id": int(st["next_id"]),
            "tracks": self._snapshot_tracks(st["tracks"]),
        }

    def _snapshot_tracks(self, tracks: list[dict[str, any]]) -> list[dict[str, any]]:
        return [self._public_track(track, idx) for idx, track in enumerate(tracks)]

    def _public_track(
        self,
        track: dict[str, any],
        track_index: int | None = None,
    ) -> dict[str, any]:
        public = {
            "id": int(track["id"]) if track.get("id") is not None else None,
            "orientation": track.get("orientation"),
            "rho": float(track["rho"]),
            "rho_pred": float(track.get("rho_pred", track["rho"])),
            "theta": float(track["theta"]),
            "conf": float(track["conf"]),
            "hits": int(track.get("hits", 0)),
            "miss": int(track.get("miss", 0)),
        }
        if track_index is not None:
            public["track_index"] = int(track_index)
        return public

    def _public_candidate(
        self,
        candidate: dict[str, any],
        candidate_index: int,
    ) -> dict[str, any]:
        return {
            "candidate_index": int(candidate_index),
            "orientation": candidate.get("orientation"),
            "rho": float(candidate["rho"]),
            "theta": float(candidate["theta"]),
            "conf": float(candidate.get("conf", 0.0)),
        }

    def _median_value(self, values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2 == 1:
            return float(sorted_values[mid])
        return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])

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

    def _clamp(
        self,
        value: float,
        low: float,
        high: float,
    ) -> float:
        return max(low, min(high, value))
