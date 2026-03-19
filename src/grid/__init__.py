from importlib import import_module

_EXPORTS = {
    "FramePreprocessor": ".preprocess",
    "Line": ".line",
    "LineDetector": ".line_detection",
    "TemporalLineAccumulator": ".accumulation",
    "LineClusterizer": ".clusterization",
    "LineRegularizer": ".regularization",
    "GapAnalyzer": ".gap_analysis",
    "GridTracker": ".tracker",
    "GridBuilder": ".grid",
    "GridRenderer": ".rendering",
}

__all__ = [
    "FramePreprocessor",
    "Line",
    "LineDetector",
    "TemporalLineAccumulator",
    "LineClusterizer",
    "LineRegularizer",
    "GapAnalyzer",
    "GridTracker",
    "GridBuilder",
    "GridRenderer",
]


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
