import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from typing import ClassVar, Optional, Pattern, Sequence

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from src.path import REVIEW_IMAGES_PATH


@dataclass
class ReviewImageDownloader:
    """Parallel review-image downloader; saves JPEGs under `REVIEW_IMAGES_PATH/<save_dir_name>/`."""

    save_dir_name: str = "default"
    max_workers: int = 16
    max_retries: int = 2
    preferred_keys: tuple[str, ...] = ("medium_image_url", "large_image_url", "image_url")

    HEADERS: ClassVar[dict] = {"User-Agent": "Mozilla/5.0"}
    # Chars invalid in Windows filenames (and whitespace); replaced when building stable row ids.
    _PATH_UNSAFE_RE: ClassVar[Pattern] = re.compile(r'[<>:"/\\|?*\s]')

    def __post_init__(self):
        self.save_dir = os.path.join(REVIEW_IMAGES_PATH, self.save_dir_name)
        os.makedirs(self.save_dir, exist_ok=True)

    @classmethod
    def _safe_row_ids(cls, df: pd.DataFrame, id_cols: Sequence[str]) -> list[str]:
        """Build path-safe row ids: cast datetime cols to int ms (avoids ':' in filenames), join, then scrub remaining unsafe chars."""
        parts = {}
        for col in id_cols:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                # ns → ms; keeps row-id stable across raw-int vs parquet-datetime round-trips.
                parts[col] = (s.astype("int64") // 1_000_000).astype(str)
            else:
                parts[col] = s.astype(str)
        joined = pd.DataFrame(parts).agg("_".join, axis=1)
        return joined.map(lambda x: cls._PATH_UNSAFE_RE.sub("-", x)).tolist()

    # ---- folder-level shortcut

    def has_existing_files(self) -> bool:
        """True iff `save_dir` already contains at least one .jpg (treat as fully downloaded)."""
        return any(f.endswith(".jpg") for f in os.listdir(self.save_dir))

    def reconstruct_paths(self, df: pd.DataFrame,
                          id_cols: Sequence[str]) -> pd.DataFrame:
        """Attach `review_image_paths` by scanning `save_dir` for `{id_cols joined}_{img_idx}.jpg` files (no download)."""
        missing = [c for c in id_cols if c not in df.columns]
        if missing:
            raise KeyError(f"id_cols missing in DataFrame: {missing}")

        df = df.reset_index(drop=True).copy()
        row_ids = self._safe_row_ids(df, id_cols)

        files_by_prefix: dict = {}
        for fname in os.listdir(self.save_dir):
            if not fname.endswith(".jpg"):
                continue
            prefix, _, idx_str = fname[:-4].rpartition("_")
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            files_by_prefix.setdefault(prefix, []).append((idx, os.path.join(self.save_dir, fname)))

        for prefix in files_by_prefix:
            files_by_prefix[prefix].sort(key=lambda t: t[0])

        df["review_image_paths"] = [
            [path for _, path in files_by_prefix.get(rid, [])]
            for rid in row_ids
        ]
        return df

    # ---- URL extraction

    @staticmethod
    def _url_from_entry(entry, keys: tuple[str, ...]) -> Optional[str]:
        """Resolve a single entry (str or dict) to a URL string, trying `keys` in order."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            for k in keys:
                v = entry.get(k)
                if v:
                    return v
        return None

    def _extract_urls(self, item) -> list[str]:
        """Extract ALL URLs from a cell; returns a (possibly empty) list."""
        if item is None:
            return []
        if isinstance(item, str):
            return [item]
        if isinstance(item, (list, np.ndarray)):
            out = []
            for sub in item:
                u = self._url_from_entry(sub, self.preferred_keys)
                if u:
                    out.append(u)
            return out
        if isinstance(item, dict):
            u = self._url_from_entry(item, self.preferred_keys)
            return [u] if u else []
        return []

    # ---- file I/O

    def _download_one(self, url: str, save_path: str) -> Optional[str]:
        """Download `url` to `save_path` with up to `max_retries` retries; return path on success."""
        if not url or not isinstance(url, str):
            return None
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return save_path

        for _ in range(self.max_retries + 1):
            try:
                response = requests.get(url, headers=self.HEADERS, timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    image.convert("RGB").save(save_path, "JPEG")
                    return save_path
            except Exception:
                pass
        return None

    # ---- driver

    def run(self, df: pd.DataFrame, url_col: str,
            id_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Download every image per row, attach `review_image_paths`. Filenames are `{joined_ids}_{img_idx}.jpg` for cache stability."""
        print(f"[ReviewImageDownloader] Downloading images to {self.save_dir}...")
        df = df.reset_index(drop=True).copy()

        if id_cols is not None:
            missing = [c for c in id_cols if c not in df.columns]
            if missing:
                raise KeyError(f"id_cols missing in DataFrame: {missing}")
            row_ids = self._safe_row_ids(df, id_cols)
        else:
            row_ids = [str(i) for i in range(len(df))]

        url_lists = df[url_col].apply(self._extract_urls).tolist()

        tasks = []
        for row_idx, (rid, urls) in enumerate(zip(row_ids, url_lists)):
            for img_idx, url in enumerate(urls):
                save_path = os.path.join(self.save_dir, f"{rid}_{img_idx}.jpg")
                tasks.append((row_idx, url, save_path))

        print(f"[ReviewImageDownloader] {len(tasks):,} images to download across {len(df):,} reviews.")

        def _task(t):
            row_idx, url, save_path = t
            return (row_idx, self._download_one(url, save_path))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(_task, tasks),
                total=len(tasks), desc="Downloading",
            ))

        per_row: list[list[str]] = [[] for _ in range(len(df))]
        for row_idx, path in results:
            if path:
                per_row[row_idx].append(path)

        df["review_image_paths"] = per_row
        return df
