import math
import cv2
import numpy as np


class GridTracker:
    """Track grid lines across frames using global optical-flow motion and line matching."""

    def __init__(
        self,
        track_match_max_dist_px: float = 18.0,
        track_shift_max_px: float = 45.0,
        track_max_miss: int = 16,
        track_rho_smooth: float = 0.65,
        track_theta_smooth: float = 0.55,
        track_conf_decay: float = 0.92,
        track_min_conf: float = 0.02,
        track_warmup_frames: int = 3,
        track_memory_max_miss: int = 8,
        track_memory_min_conf: float = 0.05,
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

    @property
    def warmup_frames(self) -> int:
        return int(self._track_warmup_frames)

    def __call__(
        self,
        clustered_lines: list[dict[str, any]],
        analysis_gray: np.ndarray,
    ) -> dict[str, any]:
        flow_gray, flow_scale_x, flow_scale_y = self._prepare_flow_frame(analysis_gray)
        flow_dx, flow_dy = self._estimate_optical_flow_shift(self._prev_gray, flow_gray)
        flow_dx = self._clamp(flow_dx * flow_scale_x, -self._track_shift_max_px, self._track_shift_max_px)
        flow_dy = self._clamp(flow_dy * flow_scale_y, -self._track_shift_max_px, self._track_shift_max_px)
        self._prev_gray = flow_gray.copy()
        predicted: dict[str, list[dict[str, any]]] = {}
        reference: dict[str, list[dict[str, any]]] = {}
        
        for orientation in ("vertical", "horizontal"):
            lines = [line for line in clustered_lines if line["orientation"] == orientation]
            flow_shift = flow_dx if orientation == "vertical" else flow_dy
            updated = self._update_orientation(orientation, lines, flow_shift)
            predicted[orientation] = updated["predicted"]
            reference[orientation] = updated["reference"]
        
        return {
            "predicted": predicted,
            "reference": reference,
            "flow_shift": {"dx": float(flow_dx), "dy": float(flow_dy)},
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
        
        for track in tracks:
            track["rho_pred"] = track["rho"] + self._clamp(
                float(flow_shift),
                -self._track_shift_max_px,
                self._track_shift_max_px,
            )
        
        align_shift = self._estimate_alignment_delta(tracks, candidates)
        if abs(align_shift) > 1e-6:
            for track in tracks:
                track["rho_pred"] += align_shift
        
        matches, used_tracks, used_candidates = self._match(tracks, candidates)
        for track_idx, candidate_idx in matches:
            track = tracks[track_idx]
            candidate = candidates[candidate_idx]
            pred_rho = track.get("rho_pred", track["rho"])
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
            tracks.append(
                {
                    "id": st["next_id"],
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
        
        st["tracks"] = [
            track
            for track in tracks
            if track["miss"] <= self._track_max_miss
            and (track["hits"] >= self._track_warmup_frames or track["miss"] == 0)
        ]
        
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
                for track in sorted(st["tracks"], key=lambda item: (item["miss"], -item["conf"]))
                if 0 < track["miss"] <= self._track_memory_max_miss
                and track["conf"] >= self._track_memory_min_conf
                and track["hits"] >= self._track_warmup_frames
            ]
        return {"reference": reference_tracks, "predicted": predicted_tracks}

    def _estimate_alignment_delta(
        self,
        tracks: list[dict[str, any]],
        candidates: list[dict[str, any]],
    ) -> float:
        if not tracks or len(candidates) < self._track_align_min_cands:
            return 0.0
        cand_rhos = [candidate["rho"] for candidate in candidates]
        deltas: list[float] = []

        for track in tracks:
            pred_rho = track.get("rho_pred", track["rho"])
            nearest = min(cand_rhos, key=lambda value: abs(value - pred_rho))
            deltas.append(nearest - pred_rho)

        if not deltas:
            return 0.0
        
        delta = self._median_value(deltas)
        return self._clamp(delta, -self._track_align_max_corr_px, self._track_align_max_corr_px)

    def _match(
        self,
        tracks: list[dict[str, any]],
        candidates: list[dict[str, any]],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        pairs: list[tuple[float, int, int]] = []
        
        for track_idx, track in enumerate(tracks):
            pred_rho = track.get("rho_pred", track["rho"])
            for candidate_idx, candidate in enumerate(candidates):
                dist = abs(pred_rho - candidate["rho"])
                if dist <= self._track_match_max_dist_px:
                    pairs.append((dist, track_idx, candidate_idx))
        
        pairs.sort(key=lambda item: item[0])
        used_tracks = set()
        used_candidates = set()
        matches = []
        
        for _, track_idx, candidate_idx in pairs:
            if track_idx in used_tracks or candidate_idx in used_candidates:
                continue
            used_tracks.add(track_idx)
            used_candidates.add(candidate_idx)
            matches.append((track_idx, candidate_idx))
        
        return matches, used_tracks, used_candidates

    def _estimate_optical_flow_shift(
        self,
        prev_gray: np.ndarray | None,
        curr_gray: np.ndarray | None,
    ) -> tuple[float, float]:
        if prev_gray is None or curr_gray is None:
            return 0.0, 0.0
        
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        
        if prev_pts is None or len(prev_pts) < 16:
            return 0.0, 0.0
        
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
            return 0.0, 0.0
        mask = status.reshape(-1).astype(bool)
        tracked_prev = prev_pts.reshape(-1, 2)[mask]
        tracked_curr = curr_pts.reshape(-1, 2)[mask]
        
        if len(tracked_prev) < 16:
            return 0.0, 0.0
        
        affine, _ = cv2.estimateAffinePartial2D(
            tracked_prev,
            tracked_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        
        if affine is not None:
            dx = float(affine[0, 2])
            dy = float(affine[1, 2])
        else:
            dx = self._median_value([float(curr[0] - prev[0]) for prev, curr in zip(tracked_prev, tracked_curr)])
            dy = self._median_value([float(curr[1] - prev[1]) for prev, curr in zip(tracked_prev, tracked_curr)])
        
        return dx, dy

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
