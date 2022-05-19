"""Place where I store all of the functions that are useful in multiple scripts"""

# Constants needed to parse IRS 990s
from constants import *

# Custom class to define schema of row data
from classes import *

# Environment
import os

# Type hints
from typeguard import typechecked
from typing import (
    Sequence,
    Union,
    Dict,
    Deque,
    Iterable,
    Mapping,
    Tuple,
    List,
    Optional,
    Any,
)


# Pretty print
from pprint import pprint

# Data analysis
import pandas as pd
import numpy as np
import datetime
from datetime import date
import time
from collections import Counter
import random

# Ignore excessive warnings
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Graph
import networkx as nx

# Writing to parquet
import pyarrow as pa
import pyarrow.parquet as pq

# Neural network
import torch

# Progress bars
from tqdm.notebook import tqdm

# ML modules
# Splitting data
from sklearn.model_selection import train_test_split

# Preparing data
from sklearn import preprocessing


# Get root directory and other directory paths to use in scripts
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.curdir))
DATA_PATH = f"{PROJECT_ROOT+'/data/'}"


@typechecked
def Merge(dict1: Mapping, dict2: Mapping) -> None:
    """Merge two dictionaries"""
    return dict1.update(dict2)


@typechecked
def save_to_parquet(
    data: pd.DataFrame, cols: Sequence[str], loc: str, filename: str
) -> None:
    """Save the processed dataframes into parquet files.

    Args:
        data (pd.DataFrame): Input Pandas DataFrame.
        cols (Sequence[str]): Column names.
        loc (str): Folder location.
        filename (str): Filename.
    """
    schema = pa.schema({val: pa.string() for val in cols})
    table = pa.Table.from_pandas(data, schema=schema)
    pq.write_table(
        table, where=f"{loc}{filename}.parquet", compression="snappy",
    )


@typechecked
def load_parquet(
    loc: str, filename: str, frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> pd.DataFrame:
    """Load in the cleaned grants data.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): Print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Gold grants DataFrame.
    """
    # Load in data
    filepath = f"{loc}{filename}.parquet"

    df = (
        pd.read_parquet(filepath)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
        .replace("", pd.NA)
    )

    # Check it out
    if verbose:
        print(f"Sampling {round(frac*100,2)}% of {filename.upper()} data...")
        print(f"\tShape: {df.shape}")

    # Return dataframe with columns sorted alphabetically
    return df[sorted(df.columns)]


@typechecked
def load_model_data(
    skip_na: bool = True, frac: float = 1.0, seed: int = SEED,
) -> pd.DataFrame:
    data = load_parquet(loc=DATA_PATH, filename="processed", frac=frac, seed=seed)
    if skip_na:
        from ast import literal_eval

        print(f"Size of data before: {data.shape}")
        data.loc[:, "narrative_na"] = data["narrative_na"].apply(literal_eval)
        data = data[~data["narrative_na"]].copy().reset_index(drop=True)
        print(f"Size of data after: {data.shape}")
    return data[["label", "sequence"]]


save_to_parquet(
    data=processed_df,
    cols=processed_df.columns.tolist(),
    loc="../data/",
    filename="processed",
)
