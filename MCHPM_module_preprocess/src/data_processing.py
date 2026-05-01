import gzip
import html
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.image_cue_extractor import ImageCueExtractor
from src.path import PROCESSED_PATH, ROOT_DIR
from src.review_image_downloader import ReviewImageDownloader
from src.text_cue_extractor import TextCueExtractor
from src.utils import load_parquet, save_parquet


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
CTRL_RE = re.compile(r"[\u0000-\u001F\u007F]")
WS_RE = re.compile(r"\s+")


# ---- DataProcessor (pipeline orchestrator) ------------------------------

@dataclass
class DataProcessor:
    """End-to-end data pipeline: raw JSONL → clean → cue extraction (cached); split runs in-memory."""

    fname: str
    review_path: str
    test_size: float
    random_state: int
    device: str
    sample_size: Optional[int] = None

    # Raw → canonical column renames applied in `_normalize`.
    # Extend with `{your_raw_col: canonical_name}` entries for datasets with different schemas.
    COLUMN_ALIASES: ClassVar[dict] = {
        "title":  "review_title",
        "text":   "raw_review",
        "images": "review_images",
    }

    def __post_init__(self):
        self.raw_path      = self._resolve_path(self.review_path)
        self.labeled_path  = os.path.join(PROCESSED_PATH, f"{self.fname}_labeled.parquet")
        self.cued_path     = os.path.join(PROCESSED_PATH, f"{self.fname}_cued.parquet")

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve config paths relative to the module root, matching MFRHP config style."""
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(ROOT_DIR, path))

    # ---- cache checks

    def _cued_exists(self) -> bool:
        """True iff the post-cue-extraction checkpoint parquet exists on disk."""
        return os.path.exists(self.cued_path)

    def _labeled_exists(self) -> bool:
        """True iff the post-`_build_label` checkpoint parquet exists on disk."""
        return os.path.exists(self.labeled_path)

    # ---- pipeline stages

    def _load_data(self) -> pd.DataFrame:
        """Load MFRHP-style raw review columns from gzip JSONL."""
        print(f"[DataProcessor] Loading raw reviews: {self.raw_path} (sample_size={self.sample_size})")

        with gzip.open(self.raw_path, "rb") as f:
            df = pd.read_json(f, lines=True, nrows=self.sample_size)

        required_cols = ["text", "helpful_vote", "images"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Required MFRHP input columns missing: {missing}")

        optional_id_cols = [c for c in ("user_id", "parent_asin", "timestamp") if c in df.columns]
        df = df[required_cols + optional_id_cols]
        print(f"[Stats] Raw rows: {len(df):,}")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MFRHP-compatible row filters and canonicalize column names."""
        def has_images(x):
            if isinstance(x, list):
                return len(x) > 0
            if isinstance(x, dict):
                return len(x) > 0
            return False

        if df["helpful_vote"].dtype == object:
            df["helpful_vote"] = df["helpful_vote"].astype(str).str.replace(",", "", regex=False)
        df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce")

        before = len(df)
        mask = (
            (df["helpful_vote"] > 0) &
            (df["text"].notna()) &
            (df["text"].astype(str).str.strip().str.len() > 0) &
            (df["images"].apply(has_images))
        )
        df = df[mask].reset_index(drop=True)
        df["helpful_vote"] = df["helpful_vote"].astype(int)
        print(f"[Stats] Dropped {before - len(df):,} rows without helpful_vote/text/images; remaining {len(df):,}")

        # Rename raw → canonical columns.
        df = df.rename(columns=self.COLUMN_ALIASES).copy()
        return df

    def _preprocess_review_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MFRHP text cleanup; write cleaned result to `clean_review`."""
        def clean_text(text):
            if not text or not str(text).strip():
                return None
            text = str(text)
            text = html.unescape(text)
            text = BeautifulSoup(text, "html.parser").get_text()
            text = URL_RE.sub(" [URL] ", text)
            text = unicodedata.normalize("NFKC", text)
            text = CTRL_RE.sub(" ", text)
            text = WS_RE.sub(" ", text).strip().lower()
            return text if text else None

        if "raw_review" not in df.columns:
            raise KeyError("Raw review-text column 'raw_review' not found.")

        print("[DataProcessor] Cleaning 'raw_review' → 'clean_review' (MFRHP style)")
        tqdm.pandas(desc="Review text cleaning")
        df = df.copy()
        df["clean_review"] = df["raw_review"].progress_apply(clean_text)

        before = len(df)
        df = df[df["clean_review"].notna() & (df["clean_review"].str.len() >= 3)].reset_index(drop=True)
        print(f"[Stats] Dropped {before - len(df):,} rows with invalid clean_review; remaining {len(df):,}")
        return df

    @staticmethod
    def _first_image_url(images, key: str = "medium_image_url") -> Optional[str]:
        """Extract the first review image URL using the same convention as MFRHP."""
        if isinstance(images, list) and len(images) > 0:
            img = images[0]
            if isinstance(img, dict) and img.get(key):
                return img[key]
        elif isinstance(images, dict):
            if images.get(key):
                return images[key]
            for v in images.values():
                if isinstance(v, dict) and v.get(key):
                    return v[key]
        return None

    def _extract_image_url(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract MFRHP-style first image URL while preserving MCHPM list-based downstream input."""
        df = df.copy()
        df["image_url"] = df["review_images"].apply(self._first_image_url)

        before = len(df)
        df = df[df["image_url"].notna()].reset_index(drop=True)
        print(f"[Stats] Dropped {before - len(df):,} rows without first image_url; remaining {len(df):,}")
        return df

    def _build_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log(x+1) transform to `helpful_vote` producing the regression `label` column (Sec 4.1)."""
        df = df.copy()
        df["label"] = np.log1p(df["helpful_vote"].values)
        print(f"[Stats] Label range: [{df['label'].min():.4f}, {df['label'].max():.4f}]")
        return df

    def _download_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Download the MFRHP-selected first image per review; rows without readable images are zero-padded downstream."""
        downloader = ReviewImageDownloader(save_dir_name=self.fname)
        id_cols = [c for c in ("user_id", "parent_asin", "timestamp") if c in df.columns]
        id_cols = id_cols or None

        if downloader.has_existing_files() and id_cols is not None:
            print(f"[DataProcessor] Image folder exists at {downloader.save_dir}; skipping download.")
            df = downloader.reconstruct_paths(df, id_cols=id_cols)
        else:
            df = downloader.run(df, url_col="image_url", id_cols=id_cols)

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
                df = self._load_data()                                           # 1. Load gz
                df = self._normalize(df)                                         # 2. MFRHP row filters
                df = self._preprocess_review_text(df)                            # 3. MFRHP text preprocessing
                df = self._extract_image_url(df)                                  # 4. MFRHP first-image extraction
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
