"""Generate wake word datasets and benchmark any wake word model."""

from .bench import BenchConfigRun, BenchResult, benchmark, compare, evaluate, run_detector
from .config import BenchConfig, load_config, make_config
from .dataset import DEFAULT_DATASET
from .detectors import Detector, Prediction, SlidingWindowDetector, load_detector
from .generate import generate
from .loader import Sample, WakeWordData

__all__ = [
    "DEFAULT_DATASET",
    "BenchConfig",
    "BenchConfigRun",
    "BenchResult",
    "Detector",
    "Prediction",
    "Sample",
    "SlidingWindowDetector",
    "WakeWordData",
    "benchmark",
    "compare",
    "evaluate",
    "generate",
    "load_config",
    "load_detector",
    "make_config",
    "run_detector",
]
