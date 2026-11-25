MAX_SPEED_MM_S = 5.0
MIN_SPEED_MM_S = 0.1
MAX_POINTS = 600


def axis_stops(length_mm, step_mm):
    if step_mm <= 0:
        raise ValueError("Крок має бути більшим за нуль")
    if length_mm < 0:
        raise ValueError("Розмір сітки не може бути від'ємним")
    stops = []
    value = 0.0
    while value < length_mm - 1e-9:
        stops.append(round(value, 4))
        value += step_mm
    if not stops or abs(stops[-1] - length_mm) > 1e-6:
        stops.append(round(length_mm, 4))
    return stops


def serpentine(width_mm, height_mm, step_mm):
    xs = axis_stops(width_mm, step_mm)
    ys = axis_stops(height_mm, step_mm)
    points = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append({"x": x, "y": y})
    if len(points) > MAX_POINTS:
        raise ValueError("Забагато точок для цього кроку")
    return points


def clean_points(points):
    if not isinstance(points, list):
        raise ValueError("Точки мають бути списком")
    if len(points) > MAX_POINTS:
        raise ValueError("Забагато точок")
    cleaned = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("Точка має містити x і y")
        cleaned.append({
            "x": round(float(point["x"]), 4),
            "y": round(float(point["y"]), 4),
        })
    return cleaned


def clean_speed(speed_mm_s):
    speed = float(speed_mm_s)
    if speed < MIN_SPEED_MM_S or speed > MAX_SPEED_MM_S:
        raise ValueError("Швидкість від %.1f до %.0f мм/с" % (MIN_SPEED_MM_S, MAX_SPEED_MM_S))
    return round(speed, 3)


def inside_workspace(x, y, workspace):
    return (
        workspace["min_x"] <= x <= workspace["max_x"]
        and workspace["min_y"] <= y <= workspace["max_y"]
    )
