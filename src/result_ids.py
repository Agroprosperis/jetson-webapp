import os
from datetime import datetime


def generate_unique_result_run_id(base_run_id, hq_output_dir, *, now_ms=None):
    candidate = str(base_run_id or "").strip()
    if not candidate:
        raise ValueError("Result run ID is required.")

    run_dir = os.path.join(hq_output_dir, candidate)
    if not os.path.exists(run_dir):
        return candidate

    stamp_value = now_ms
    if stamp_value is None:
        stamp_value = int(datetime.utcnow().timestamp() * 1000)

    unique_candidate = f"{candidate}-{int(stamp_value)}"
    suffix = 1
    while os.path.exists(os.path.join(hq_output_dir, unique_candidate)):
        unique_candidate = f"{candidate}-{int(stamp_value)}-{suffix}"
        suffix += 1

    return unique_candidate
