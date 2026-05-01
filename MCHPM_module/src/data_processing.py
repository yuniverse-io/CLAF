import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.image_cue_extractor import ImageCueExtractor
from src.path import PROCESSED_PATH, RAW_PATH
from src.review_image_downloader import ReviewImageDownloader
from src.text_cue_extractor import TextCueExtractor
from src.text_processing import clean_review_text, is_english
from src.utils import load_json_gz, load_parquet, save_parquet


# ---- DataProcessor (pipeline orchestrator) ------------------------------

@dataclass
class DataProcessor:
    """End-to-end data pipeline: raw JSONL → clean → cue extraction (cached); split runs in-memory."""

    fname: str
    test_size: float
    random_state: int
    device: str

    # Raw → canonical column renames applied in `_normalize` (after timestamp conversion).
    # Extend with `{your_raw_col: canonical_name}` entries for datasets with different schemas.
    COLUMN_ALIASES: ClassVar[dict] = {
        "title":  "review_title",
        "text":   "raw_review",
        "images": "review_images",
    }

    def __post_init__(self):
        self.raw_path      = os.path.join(RAW_PATH, f"{self.fname}.jsonl.gz")
        self.labeled_path  = os.path.join(PROCESSED_PATH, f"{self.fname}_labeled.parquet")
        self.cued_path     = os.path.join(PROCESSED_PATH, f"{self.fname}_cued.parquet")

    # ---- cache checks

    def _cued_exists(self) -> bool:
        """True iff the post-cue-extraction checkpoint parquet exists on disk."""
        return os.path.exists(self.cued_path)

    def _labeled_exists(self) -> bool:
        """True iff the post-`_build_label` checkpoint parquet exists on disk."""
        return os.path.exists(self.labeled_path)

    # ---- pipeline stages

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply early row filters and canonicalize column names (paper Sec 4.1 pre-split filters)."""
        # Identifier columns required downstream (image filenames, review_date) must be non-null.
        id_cols = ["user_id", "parent_asin", "timestamp"]
        missing = [c for c in id_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Required identifier columns missing: {missing}")
        before = len(df)
        df = df.dropna(subset=id_cols).reset_index(drop=True)
        print(f"[Stats] Dropped {before - len(df):,} rows with null user_id/parent_asin/timestamp; remaining {len(df):,}")

        # Verified_purchase filter.
        if "verified_purchase" in df.columns:
            before = len(df)
            df = df[df["verified_purchase"] == True].reset_index(drop=True)
            print(f"[Stats] Dropped {before - len(df):,} unverified reviews; remaining {len(df):,}")
        else:
            print("[DataProcessor] 'verified_purchase' column not found; skipping verified filter.")

        # Timestamp (ms) → `review_date` datetime column.
        df["review_date"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Rename raw → canonical columns.
        df = df.rename(columns=self.COLUMN_ALIASES).copy()

        # Keep only rows with at least one review image.
        if "review_images" not in df.columns:
            raise KeyError("Image column 'review_images' not found (check COLUMN_ALIASES).")
        before = len(df)
        has_image = df["review_images"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)
        df = df[has_image].reset_index(drop=True)
        print(f"[Stats] Dropped {before - len(df):,} rows without images; remaining {len(df):,}")

        # helpful_vote filter: drop rows with missing or zero votes.
        if "helpful_vote" not in df.columns:
            raise KeyError("Vote column 'helpful_vote' not found.")
        if df["helpful_vote"].dtype == object:
            df["helpful_vote"] = df["helpful_vote"].str.replace(",", "", regex=False)
        df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce")
        before = len(df)
        df = df[df["helpful_vote"] > 0].reset_index(drop=True)
        df["helpful_vote"] = df["helpful_vote"].astype(int)
        print(f"[Stats] Dropped {before - len(df):,} zero/missing helpful_vote rows; remaining {len(df):,}")

        # Non-English filter applied on raw review text.
        if "raw_review" in df.columns:
            before = len(df)
            df = df[df["raw_review"].apply(is_english)].reset_index(drop=True)
            print(f"[Stats] Dropped {before - len(df):,} non-English rows; remaining {len(df):,}")
        else:
            print("[DataProcessor] 'raw_review' column not found; skipping non-English filter.")

        return df

    def _preprocess_review_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimal URL/HTML cleanup; write cleaned result to `clean_review`."""
        if "raw_review" not in df.columns:
            raise KeyError("Raw review-text column 'raw_review' not found.")

        print("[DataProcessor] Cleaning 'raw_review' → 'clean_review'")
        tqdm.pandas(desc="Review text cleaning")
        df = df.copy()
        df["clean_review"] = df["raw_review"].progress_apply(clean_review_text)
        return df

    def _build_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log(x+1) transform to `helpful_vote` producing the regression `label` column (Sec 4.1)."""
        df = df.copy()
        df["label"] = np.log1p(df["helpful_vote"].values)
        print(f"[Stats] Label range: [{df['label'].min():.4f}, {df['label'].max():.4f}]")
        return df

    def _download_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Download every image per review; if the folder already has files, skip download and reconstruct paths. Rows without images are zero-padded downstream."""
        downloader = ReviewImageDownloader(save_dir_name=self.fname)
        id_cols = ["user_id", "parent_asin", "timestamp"]

        if downloader.has_existing_files():
            print(f"[DataProcessor] Image folder exists at {downloader.save_dir}; skipping download.")
            df = downloader.reconstruct_paths(df, id_cols=id_cols)
        else:
            df = downloader.run(df, url_col="review_images", id_cols=id_cols)

        empty = int(df["review_image_paths"].apply(lambda x: not isinstance(x, list) or len(x) == 0).sum())
        if empty:
            print(f"[Stats] {empty:,} rows have no review images; zero-padded downstream.")
        return df

    def _extract_cues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all four cue columns; per-column skip if present, lazy model loading. Order: text → image."""
        use_gpu = "cuda" in self.device
        df = TextCueExtractor(use_gpu=use_gpu).run(df, input_col="clean_review")
        df = ImageCueExtractor(use_gpu=use_gpu).run(df, input_col="review_image_paths")
        return df

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Shuffle-split into train / test (val is carved downstream in main)."""
        train, test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        print(f"[Stats] Split sizes: train={len(train):,}, test={len(test):,}")
        return train, test

    # ---- driver

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cache-resumable through cued; split runs in-memory each call."""
        print(f"\n{'=' * 10} Data Processing {'=' * 10}")

        if self._cued_exists():
            print(f"[DataProcessor] Resuming from cued checkpoint: {self.cued_path}")
            df = load_parquet(self.cued_path)
        else:
            if self._labeled_exists():
                print(f"[DataProcessor] Resuming from labeled checkpoint: {self.labeled_path}")
                df = load_parquet(self.labeled_path)
            else:
                df = load_json_gz(self.raw_path)                                 # 1. Load gz
                print(f"[Stats] Raw rows: {len(df):,}")

                df = self._normalize(df)                                         # 2-5 + vote / non-English
                df = self._preprocess_review_text(df)                            # 6. Review-text preprocessing

                # 7. Drop rows with null preprocessed review text (inline, per spec).
                before = len(df)
                df = df.dropna(subset=["clean_review"]).reset_index(drop=True)
                print(f"[Stats] Dropped {before - len(df):,} rows with null clean_review; remaining {len(df):,}")

                df = self._build_label(df)
                save_parquet(df, self.labeled_path)
                print(f"[DataProcessor] Saved labeled checkpoint: {self.labeled_path}")

            df = self._download_images(df)
            df = self._extract_cues(df)
            save_parquet(df, self.cued_path)
            print(f"[DataProcessor] Saved cued checkpoint: {self.cued_path}")

        train, test = self._split(df)
        print("[DataProcessor] Processing complete.")
        return train, test


# ---- Peripheral-cue post-processing -------------------------------------

def standardize_peripheral_cues(train_df: pd.DataFrame, *other_dfs: pd.DataFrame) -> None:
    """Fit StandardScaler on train peripheral columns and apply to all dfs in place (no data leakage)."""
    print("[Peripheral] Standardizing cues (fit on train)...")
    for col in ("review_text_peripheral", "review_image_peripheral"):
        scaler = StandardScaler().fit(np.stack(train_df[col].values))
        for df in (train_df,) + other_dfs:
            df[col] = list(scaler.transform(np.stack(df[col].values)))


# ---- Torch Dataset / DataLoader ------------------------------------------

class MultimodalDataset(Dataset):
    """Map-style dataset: per-row dict of central/peripheral feature tensors + scalar label."""

    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "review_text_central",
        "review_image_central",
        "review_text_peripheral",
        "review_image_peripheral",
    )

    def __init__(self, df: pd.DataFrame):
        self.features = {
            col: torch.tensor(np.stack(df[col].values), dtype=torch.float32)
            for col in self.FEATURE_COLUMNS
        }
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {col: tensors[idx] for col, tensors in self.features.items()}
        item["label"] = self.labels[idx]
        return item


def get_data_loader(args: dict, df: pd.DataFrame, shuffle: bool = True) -> DataLoader:
    """Wrap a DataFrame in `MultimodalDataset` and return a torch DataLoader."""
    dataset = MultimodalDataset(df)
    return DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
