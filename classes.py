"""For specifiying schema of method input/output variables."""

from typing import TypedDict
from torch import Tensor
from typing import Tuple
from pandas import DataFrame
from typing import Dict
from dataset import ComplaintsDataset
from torch.utils.data import DataLoader


class ExperimentDataSplit(TypedDict, total=False):
    train: DataFrame
    valid: DataFrame
    test: DataFrame
    split_size: Tuple[int, int, int]
    dataset_train: ComplaintsDataset
    dataset_valid: ComplaintsDataset
    dataset_test: ComplaintsDataset
    class_weights_train: Tensor


class ExperimentData(TypedDict, total=False):
    data: DataFrame
    num_labels: int
    target2label: Dict[int, str]
    stratify_sklearn: ExperimentDataSplit
    stratify_none: ExperimentDataSplit


class DataLoaderDict(TypedDict, total=False):
    train: DataLoader
    valid: DataLoader
    test: DataLoader
