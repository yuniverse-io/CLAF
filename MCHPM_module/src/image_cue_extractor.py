import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


@dataclass
class ImageCueExtractor:
    """Multi-Cue Extraction Module — image side. Extracts review-image central (VGG-16) + peripheral (OpenCV) cues; per-column skip + lazy VGG load + single cv2 read per image."""

    use_gpu: bool = True

    def __post_init__(self):
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_vgg(self) -> None:
        """Idempotent VGG loader; called only when central cue extraction is needed."""
        if self.model is not None:
            return
        print(f"[ImageCueExtractor] Loading VGG16 on {self.device}...")
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Classifier[0..3] = Linear(25088,4096) -> ReLU -> Dropout -> Linear(4096,4096)
        classifier_head = nn.Sequential(*list(vgg.classifier.children())[:4])
        self.model = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            classifier_head,
        ).to(self.device).eval()

    def _process_one(self, review_image_path: str,
                     need_central: bool, need_peripheral: bool
                     ) -> Optional[tuple[Optional[list[float]], Optional[np.ndarray]]]:
        """Open one image once via cv2; derive only the requested cues. Returns None on failure."""
        if not review_image_path or not os.path.exists(review_image_path):
            return None
        try:
            img_bgr = cv2.imread(review_image_path)                         # ★ single disk read
            if img_bgr is None:
                return None

            peripheral = None
            central = None

            if need_peripheral:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray)) / 255.0
                contrast = float(gray.std()) / 255.0
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                saturation = float(np.mean(hsv[:, :, 1])) / 255.0
                edges = cv2.Canny(gray, 100, 200)
                edge_intensity = float(np.mean(edges)) / 255.0
                peripheral = [brightness, contrast, saturation, edge_intensity]

            if need_central:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
                tensor = self.transform(pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model(tensor)                               # Eq. 3: Vc = VGG16(I)
                central = feat.cpu().numpy().flatten().astype(np.float32)

            return peripheral, central
        except Exception:
            return None

    def _process_review(self, review_image_paths: list[str],
                        need_central: bool, need_peripheral: bool
                        ) -> tuple[Optional[list[float]], Optional[np.ndarray]]:
        """Process all images of one review; average per-image cues. Zero vectors if none readable."""
        zero_p = [0.0, 0.0, 0.0, 0.0]
        zero_c = np.zeros(4096, dtype=np.float32)
        empty = (
            zero_p if need_peripheral else None,
            zero_c if need_central    else None,
        )

        if not review_image_paths:
            return empty

        results = [
            r for r in (self._process_one(p, need_central, need_peripheral) for p in review_image_paths)
            if r is not None
        ]
        if not results:
            return empty

        avg_p, avg_c = None, None
        if need_peripheral:
            peripherals = [r[0] for r in results]
            avg_p = [float(v) for v in np.mean(peripherals, axis=0)]        # Paper Sec 4.1 step 5
        if need_central:
            centrals = [r[1] for r in results]
            avg_c = np.mean(centrals, axis=0).astype(np.float32)
        return avg_p, avg_c

    def run(self, df: pd.DataFrame, input_col: str) -> pd.DataFrame:
        """Attach image cue columns; skip per-column if present, lazy-load VGG only if central is needed."""
        if input_col not in df.columns:
            raise KeyError(f"{input_col} column not found in DataFrame.")

        need_central    = "review_image_central"    not in df.columns
        need_peripheral = "review_image_peripheral" not in df.columns
        if not need_central and not need_peripheral:
            print("[ImageCueExtractor] Both image cues already exist; skipping.")
            return df

        if need_central:
            self._load_vgg()

        df = df.copy()
        peripherals, centrals = [], []
        for paths in tqdm(df[input_col].tolist(), desc="Image encoding"):
            p, c = self._process_review(paths, need_central, need_peripheral)
            if need_peripheral:
                peripherals.append(p)
            if need_central:
                centrals.append(c)

        if need_peripheral:
            df["review_image_peripheral"] = peripherals
            print("[ImageCueExtractor] Peripheral cues added.")
        else:
            print("[ImageCueExtractor] review_image_peripheral exists; skipping.")
        if need_central:
            df["review_image_central"] = centrals
            print("[ImageCueExtractor] Central cues added.")
        else:
            print("[ImageCueExtractor] review_image_central exists; skipping.")

        if need_central and self.device.type == "cuda":
            torch.cuda.empty_cache()
        return df
