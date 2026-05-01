import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def load_parquet(fpath: str) -> pd.DataFrame:
    """Read a parquet file into a DataFrame (pyarrow engine)."""
    return pd.read_parquet(fpath, engine="pyarrow")


def save_parquet(df: pd.DataFrame, fpath: str) -> None:
    """Write a DataFrame to parquet, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    df.to_parquet(fpath, engine="pyarrow")


def load_json_gz(fpath: str) -> pd.DataFrame:
    """Load a gzipped JSON file (JSONL first, fallback to a JSON array)."""
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File not found: {fpath}")
    try:
        return pd.read_json(fpath, compression="gzip", lines=True)
    except ValueError:
        return pd.read_json(fpath, compression="gzip", lines=False)


def load_yaml(fpath: str) -> dict:
    """Load a YAML config file as a dict."""
    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fix random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_metrics(
    preds: np.ndarray | torch.Tensor,
    trues: np.ndarray | torch.Tensor,
) -> tuple[float, float, float, float]:
    """Calculate regression metrics: MAE, MSE, RMSE, MAPE."""
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()
    preds = np.asarray(preds).squeeze()
    trues = np.asarray(trues).squeeze()

    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(trues, preds) * 100
    return mae, mse, rmse, mape
