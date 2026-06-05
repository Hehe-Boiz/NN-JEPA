"""Data pipeline utilities for NN-JEPA."""

from .preprocess import preprocess_all_sessions
from .settings import ACTION_COLUMNS, STATE_COLUMNS

__all__ = [
    "ACTION_COLUMNS",
    "STATE_COLUMNS",
    "preprocess_all_sessions",
    "DrivingJEPADataset",
    "create_dataloaders",
]


def __getattr__(name: str):
    if name == "DrivingJEPADataset":
        from .dataset import DrivingJEPADataset

        return DrivingJEPADataset
    if name == "create_dataloaders":
        from .dataset import create_dataloaders

        return create_dataloaders
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
