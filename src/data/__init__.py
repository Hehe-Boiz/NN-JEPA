"""Data pipeline utilities for NN-JEPA."""

from .preprocess import preprocess_all_sessions
from .settings import ACTION_COLUMNS, STATE_COLUMNS

__all__ = [
    "ACTION_COLUMNS",
    "STATE_COLUMNS",
    "preprocess_all_sessions",
    "DrivingJEPADataset",
    "create_dataloaders",
    "RCJepaACSequenceDataset",
    "create_ac_sequence_dataloaders",
    "RCJepaACFeatureSequenceDataset",
    "create_ac_feature_sequence_dataloaders",
]


def __getattr__(name: str):
    if name == "DrivingJEPADataset":
        from .dataset import DrivingJEPADataset

        return DrivingJEPADataset
    if name == "create_dataloaders":
        from .dataset import create_dataloaders

        return create_dataloaders
    if name == "RCJepaACSequenceDataset":
        from .sequence_dataset import RCJepaACSequenceDataset

        return RCJepaACSequenceDataset
    if name == "create_ac_sequence_dataloaders":
        from .sequence_dataset import create_ac_sequence_dataloaders

        return create_ac_sequence_dataloaders
    if name == "RCJepaACFeatureSequenceDataset":
        from .feature_sequence_dataset import RCJepaACFeatureSequenceDataset

        return RCJepaACFeatureSequenceDataset
    if name == "create_ac_feature_sequence_dataloaders":
        from .feature_sequence_dataset import create_ac_feature_sequence_dataloaders

        return create_ac_feature_sequence_dataloaders
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
