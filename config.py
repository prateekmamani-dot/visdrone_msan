"""Shared constants for the project."""

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
# torchvision detectors reserve label 0 for background, so VisDrone
# categories 1..10 map directly onto labels 1..10.
NUM_CLASSES = len(CLASS_NAMES) + 1

# Anchors are smaller than the torchvision default (32..512) because most
# VisDrone objects are under 32x32 pixels.
ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
ANCHOR_RATIOS = ((0.5, 1.0, 2.0),) * 5

MIN_SIZE = 800
MAX_SIZE = 1333
