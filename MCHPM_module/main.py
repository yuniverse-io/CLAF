import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model.mchpm import MCHPM, predict, train
from src.data_processing import DataProcessor, get_data_loader, standardize_peripheral_cues
from src.path import SAVE_MODEL_PATH, SRC_PATH
from src.utils import get_metrics, load_yaml, set_seed


def run_data_processing(dargs: dict, seed: int, fname: str, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the DataProcessor pipeline; returns in-memory train and test dfs (no on-disk split cache)."""
    return DataProcessor(
        fname=fname,
        test_size=dargs["test_size"],
        random_state=seed,
        device=device,
    ).run()


def build_loaders(args: dict, train_df: pd.DataFrame, test_df: pd.DataFrame,
                  seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Carve val out of train, standardize peripheral cues, and wrap each split in a torch DataLoader."""
    train_df, val_df = train_test_split(train_df, test_size=args["val_ratio"], random_state=seed)
    standardize_peripheral_cues(train_df, val_df, test_df)
    print(f"[Main] Train shape: {train_df.shape}")
    print(f"[Main] Val shape  : {val_df.shape}")
    print(f"[Main] Test shape : {test_df.shape}")
    return (
        get_data_loader(args, train_df, shuffle=True),
        get_data_loader(args, val_df,   shuffle=False),
        get_data_loader(args, test_df,  shuffle=False),
    )


def build_model(args: dict) -> MCHPM:
    """Instantiate the MCHPM model from config args."""
    return MCHPM(
        feature_dimension=args["feature_dimension"],
        num_heads=args["num_heads"],
        dropout=args["dropout"],
        dff=args["dff"],
    )


def resolve_device(requested: str) -> str:
    """Honor the configured device, falling back to CPU if CUDA was requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Main] CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested


def main() -> None:
    cfg = load_yaml(os.path.join(SRC_PATH, "config.yaml"))
    dargs = cfg["data"]
    args = cfg["args"]
    fname = dargs["fname"]

    seed = cfg["seed"]
    set_seed(seed)

    device = resolve_device(args["device"])
    print(f"[Main] Device: {device}")

    train_df, test_df = run_data_processing(dargs, seed, fname, device)
    train_loader, val_loader, test_loader = build_loaders(args, train_df, test_df, seed)

    print("[Main] Building PyTorch model...")
    model = build_model(args)

    save_dir = os.path.join(SAVE_MODEL_PATH, fname)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best.pth")
    model = train(
        args=args, model=model,
        train_loader=train_loader, val_loader=val_loader,
        best_model_path=best_model_path, device=device,
    )

    test_preds, test_trues = predict(model, test_loader, device=device)
    mae, mse, rmse, mape = get_metrics(test_preds, test_trues)
    print(f"[Test] MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  MAPE={mape:.3f}%")


if __name__ == "__main__":
    main()
