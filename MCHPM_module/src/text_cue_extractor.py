from dataclasses import dataclass

import numpy as np
import pandas as pd
import textstat
import torch
from textblob import TextBlob
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class TextCueExtractor:
    """Multi-Cue Extraction Module — text side. Extracts review-text central (BERT) + peripheral (TextBlob/textstat) cues; per-column skip + lazy BERT load."""

    model_ckpt: str = "bert-base-uncased"
    batch_size: int = 32
    use_gpu: bool = True

    def __post_init__(self):
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def _load_bert(self) -> None:
        """Idempotent BERT loader; called only when central cue extraction is needed."""
        if self.model is not None:
            return
        print(f"[TextCueExtractor] Loading {self.model_ckpt} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model = AutoModel.from_pretrained(self.model_ckpt).to(self.device).eval()

    @staticmethod
    def _peripheral(text: str) -> list[float]:
        """Polarity / subjectivity / readability / extremity (Table 1)."""
        if not isinstance(text, str) or not text.strip():
            return [0.0, 0.0, 0.0, 0.0]
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        try:
            readability = textstat.flesch_reading_ease(text)
        except Exception:
            readability = 0.0
        extremity = abs(polarity)
        return [polarity, subjectivity, readability, extremity]

    @torch.no_grad()
    def _central(self, texts: list[str]) -> list[np.ndarray]:
        """BERT [CLS] embeddings batched on GPU; returns per-item 768-dim numpy vectors."""
        vectors: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="BERT encoding"):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)
            outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]                        # Eq. 1: Tc = BERT(S)
            for v in cls.cpu().numpy():
                vectors.append(v.astype(np.float32))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return vectors

    def run(self, df: pd.DataFrame, input_col: str) -> pd.DataFrame:
        """Attach text cue columns; skip per-column if already present, lazy-load BERT only if central is needed."""
        if input_col not in df.columns:
            raise KeyError(f"{input_col} column not found in DataFrame.")

        need_central    = "review_text_central"    not in df.columns
        need_peripheral = "review_text_peripheral" not in df.columns
        if not need_central and not need_peripheral:
            print("[TextCueExtractor] Both text cues already exist; skipping.")
            return df

        df = df.copy()
        texts = df[input_col].tolist()

        if need_peripheral:
            df["review_text_peripheral"] = [self._peripheral(t) for t in texts]
            print("[TextCueExtractor] Peripheral cues added.")
        else:
            print("[TextCueExtractor] review_text_peripheral exists; skipping.")

        if need_central:
            self._load_bert()
            df["review_text_central"] = self._central(texts)
            print("[TextCueExtractor] Central cues added.")
        else:
            print("[TextCueExtractor] review_text_central exists; skipping.")

        return df
