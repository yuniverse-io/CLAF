import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")

import time

total_start_time = time.perf_counter()

import glob
import gzip
import hashlib
import html
import json
import random as python_random
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFRobertaModel, TFViTModel, RobertaTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


DATA_PATH = "/home/sw001/workspace/RHP/yoon_heo/data/Sports_and_Outdoors.jsonl.gz"
IMG_PATH = "/home/sw001/workspace/RHP/yoon_heo/data/img"
SAVE_DIR = "/home/sw001/workspace/RHP/yoon_heo/Modules/Save_weight/weight_time_check"
DATA_CACHE_DIR = "/home/sw001/workspace/RHP/yoon_heo/data/preprocess"

MAX_LENGTH = 256
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 1e-4
TUNE_FRAC = 0.1
TUNE_NUM_BINS = 10

tokenizer = None


def reset_random_seeds():
    tf.random.set_seed(42)
    np.random.seed(42)
    python_random.seed(42)


reset_random_seeds()


def configure_runtime():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"memory growth 설정 스킵: {exc}", flush=True)

    print(f"TensorFlow에서 인식한 GPU 수: {len(gpus)}", flush=True)
    for idx, gpu in enumerate(gpus):
        print(f"  GPU {idx}: {gpu}", flush=True)


def load_and_preprocess(DATA_PATH):
    # # 1) gzip 리뷰 데이터 로드
    df = pd.DataFrame([json.loads(l) for l in gzip.open(DATA_PATH, "rb")])
    print("원본:", df.shape, flush=True)

    # 2) 데이터 필터링
    def has_images(x):
        if isinstance(x, list):
            return len(x) > 0
        if isinstance(x, dict):
            return len(x) > 0
        return False

    mask = (
        (df["helpful_vote"] > 0) &
        (df["text"].notna()) &
        (df["text"].str.strip().str.len() > 0) &
        (df["images"].apply(has_images))
    )
    df = df[mask].reset_index(drop=True)
    print(f"필터링 후 (helpful_vote>0 & 텍스트/이미지 존재): {len(df)}개", flush=True)

    # 3) 텍스트 전처리
    URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
    CTRL_RE = re.compile(r"[\u0000-\u001F\u007F]")
    WS_RE = re.compile(r"\s+")

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

    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].notna() & (df["clean_text"].str.len() >= 3)].reset_index(drop=True)
    print(f"텍스트 전처리 후 (유효 텍스트 길이 >= 3): {len(df)}개", flush=True)

    # 4) 첫 번째 리뷰 이미지 URL 추출
    def first_image_url(images, key="medium_image_url"):
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

    df["image_url"] = df["images"].apply(first_image_url)
    df = df[df["image_url"].notna()].reset_index(drop=True)
    print(f"이미지 URL 추출 후: {len(df)}개", flush=True)

    df["log_helpful_vote"] = np.log(df["helpful_vote"].astype(np.float32) + 1)
    # 5) 필요한 컬럼만 남기기
    df = df[["clean_text", "image_url", "log_helpful_vote"]].copy()
    print(f"\n최종 데이터: {len(df)}개", flush=True)
    print(f"log_helpful_vote 분포:\n{df['log_helpful_vote'].describe()}", flush=True)

    return df


# 6) 리뷰 이미지 다운로드 (리뷰당 첫 번째 이미지 1장)
# 이미지는 이미 IMG_PATH에 있으므로 다운로드는 스킵하고 기존 파일명 규칙으로 매핑만 수행
def map_existing_images(df, image_dir):
    os.makedirs(image_dir, exist_ok=True)

    image_paths = []
    failed_indices = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="기존 이미지 매핑"):
        url = row["image_url"]
        h = hashlib.md5((url + str(i)).encode()).hexdigest()[:6]
        fname = f"{i}_{h}.jpg"
        fpath = os.path.join(image_dir, fname)

        if os.path.exists(fpath):
            image_paths.append(fpath)
        else:
            fallback_paths = sorted(glob.glob(os.path.join(image_dir, f"{i}_{h}.*")))
            if fallback_paths:
                image_paths.append(fallback_paths[0])
            else:
                failed_indices.append(i)

    # 실패한 행 drop
    df = df.drop(index=failed_indices).reset_index(drop=True)
    # 성공한 행에 image_path 매핑 (순서대로 append되었으므로 그대로 할당)
    df["image_path"] = image_paths

    success = len(image_paths)
    fail = len(failed_indices)
    print(f"\n이미지 매핑 완료: 성공 {success}, 실패 {fail} (실패 행 제거됨)", flush=True)
    print(f"최종 데이터: {len(df)}개", flush=True)
    return df


def load_or_create_tune_df(
    data_path,
    image_dir,
    cache_dir,
    frac=0.1,
    random_state=42,
    num_bins=10,
):
    # 튜닝용으로 원본 일부만 사용.
    # 이미 만들어둔 pkl이 있으면 전처리/이미지 매핑을 다시 하지 않고 바로 로드한다.
    os.makedirs(cache_dir, exist_ok=True)

    tune_df_path = os.path.join(cache_dir, f"Sports_tune_df_{int(frac * 100)}pct.pkl")

    if os.path.exists(tune_df_path):
        df_tune = pd.read_pickle(tune_df_path)
        print(f"저장된 튜닝 데이터 로드 완료: {tune_df_path}", flush=True)
        print(f"튜닝 데이터: {len(df_tune)}개", flush=True)
        print(df_tune["log_helpful_vote"].describe(), flush=True)
        return df_tune

    df = load_and_preprocess(data_path)
    df = map_existing_images(df, image_dir)

    df["_label_bin"] = pd.qcut(
        df["log_helpful_vote"],
        q=num_bins,
        duplicates="drop",
    )

    df_tune, _ = train_test_split(
        df,
        train_size=frac,
        random_state=random_state,
        stratify=df["_label_bin"],
    )

    df_tune = df_tune.drop(columns=["_label_bin"]).reset_index(drop=True)
    df_tune.to_pickle(tune_df_path)

    print(f"튜닝 데이터 새로 생성 및 저장 완료: {tune_df_path}", flush=True)
    print(f"원본 유효 데이터: {len(df)}개", flush=True)
    print(f"튜닝 데이터: {len(df_tune)}개", flush=True)
    print(df_tune["log_helpful_vote"].describe(), flush=True)

    return df_tune


# 샘플 하나를 모델 입력 형태로 변환
def preprocess_image(inputs, label):
    raw = tf.io.read_file(inputs["image_path"])
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224]) / 255.0
    img = tf.transpose((img - 0.5) / 0.5, [2, 0, 1])
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": img,
    }, label


def build_dataset(texts, image_paths, labels, max_length=256, batch_size=512, shuffle=False):
    # 텍스트 토큰화
    enc = tokenizer(texts, padding="max_length", max_length=max_length,
                    truncation=True, return_tensors="tf")

    # 입력, 딕셔너리, 라벨 쌍으로 샘플 단위로 슬라이싱
    dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": enc["input_ids"],
         "attention_mask": enc["attention_mask"],
         "image_path": image_paths},
        labels,
    ))
    # 학습 데이터일 경우 매 epoch마다 순서를 섞음
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))
    # 각 샘플에 process 전처리 적용 num_parallel_calls=AUTOTUNE로 병렬 처리
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    # 개별 샘플들을 배치 크기로 묶음
    dataset = dataset.batch(batch_size)
    # GPU가 현재 배치 학습하는 동안 CPU가 다음 배치를 준비
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ## 양방향 Cross-Attention
# ### BidirectionalCrossAttentionFusion
# 
# T_merged와 I_merged 사이에 양방향 cross-attention을 수행하는 모듈.
# 각 모달리티가 상대 모달리티 전체를 attention으로 흡수한 결과 (t2i, i2t)를 시퀀스 그대로 반환.
# 이후 외부에서 Top-K 토큰 선택 → Dense projection → flatten → MLP로 처리됨.


class BidirectionalCrossAttentionFusion(layers.Layer):
    """
    T_merged와 I_merged를 받아 양방향 cross-attention을 수행.
    각 모달리티의 cross-attended 시퀀스를 그대로 반환 (압축 / 융합 없음).

      1) t2i -> Text-to-Image Cross-Attention (Q=T, K=V=I) + FFN + Residual + LN
      2) i2t -> Image-to-Text Cross-Attention (Q=I, K=V=T) + FFN + Residual + LN

    Cross-Attention + FFN 단계별 shape:
        입력:
            text_repr  (T_merged): (B, 256, 768)
            image_repr (I_merged): (B, 197, 768)

        1단계: Text Branch
            t2i_attn = text_to_image_attn(Q=text, K=image, V=image): (B, 256, 768)
            t2i = LN(text_repr + t2i_attn): (B, 256, 768)
            t2i = LN(t2i + FFN(t2i)): (B, 256, 768)

        2단계: Image Branch
            i2t_attn = image_to_text_attn(Q=image, K=text, V=text): (B, 197, 768)
            i2t = LN(image_repr + i2t_attn): (B, 197, 768)
            i2t = LN(i2t + FFN(i2t)): (B, 197, 768)

    출력: (t2i, i2t) 튜플 — 시퀀스 차원 그대로 보존
        t2i: (B, 256, 768)  ← 텍스트 토큰 + 이미지 cross-modal 문맥
        i2t: (B, 197, 768)  ← 이미지 패치 + 텍스트 cross-modal 문맥

    설계 노트:
        본 클래스는 양방향 cross-attention + FFN까지만 담당.
        이후 단계 (Top-K 선택, Dense projection, flatten, MLP)는
        외부 모듈로 분리되어 모델 클래스에서 호출됨.
    """

    def __init__(self, d_model=768, num_heads=12, d_ff=3072, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        # Cross-Attention (양방향, 각 1개씩)
        self.text_to_image_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.image_to_text_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )

        # LayerNorms (Attention 뒤 + FFN 뒤, 각 브랜치에 2개씩)
        self.ln1_t = layers.LayerNormalization()
        self.ln2_t = layers.LayerNormalization()
        self.ln1_i = layers.LayerNormalization()
        self.ln2_i = layers.LayerNormalization()

        # FFN (텍스트 브랜치)
        self.ffn_t = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])

        # FFN (이미지 브랜치)
        self.ffn_i = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])

    def call(self, text_repr, image_repr, text_mask, training=False):
        # i2t cross-attention 시 텍스트 padding 차단을 위한 mask 생성
        # text_mask (B, 256) → (B, 1, 256) — i2t의 attention_mask 인자에 전달
        text_mask_cross = tf.expand_dims(text_mask, axis=1)

        # 1) Text branch: 텍스트가 이미지를 본 결과 (Q=text, K=V=image)
        t2i_attn = self.text_to_image_attn(
            query=text_repr, key=image_repr, value=image_repr, training=training
        )
        t2i = self.ln1_t(text_repr + t2i_attn)
        t2i = self.ln2_t(t2i + self.ffn_t(t2i, training=training))

        # 2) Image branch: 이미지가 텍스트를 본 결과 (Q=image, K=V=text, text padding mask 적용)
        i2t_attn = self.image_to_text_attn(
            query=image_repr, key=text_repr, value=text_repr,
            attention_mask=text_mask_cross, training=training
        )
        i2t = self.ln1_i(image_repr + i2t_attn)
        i2t = self.ln2_i(i2t + self.ffn_i(i2t, training=training))

        return t2i, i2t


# ## Top-K 토큰 선택 (시퀀스 보존)
# ### TopKTokenSelector
# 
# Cross-attention 출력에서 **점수 기반 상위 k개 토큰만 선택**, 나머지는 버리는 압축 레이어.
# 
# **일반 AttentionPool과의 차이**:
# - 일반 풀링: `(B, N, D) → (B, D)` — 모든 토큰을 가중합으로 단일 벡터로 뭉갬 (정보 손실 큼, N:1 압축)
# - TopKTokenSelector: `(B, N, D) → (B, k, D)` — 선택된 k개 토큰을 **시퀀스로 그대로 보존**
# 
# **왜 시퀀스로 보존하나**:
# - Cross-attention이 만든 토큰별 cross-modal 문맥을 살리기 위함
# - 단일 벡터로 압축하면 토큰 단위로 정렬된 멀티모달 정보가 평균에 묻혀 사라짐
# - k개를 시퀀스 그대로 유지하면 압축률 N:k 정도로 완화
# 
# **동작 단계**:
# 1. `Dense(1)`로 각 토큰의 중요도 스칼라 점수 계산
# 2. padding 위치는 `-1e9`로 마스킹 → top-k에서 자동 배제
# 3. `tf.math.top_k`로 상위 k개 인덱스 추출
# 4. `tf.gather`로 원본 토큰 시퀀스에서 선택된 k개 토큰을 추출
# 
# **k 자동 계산 (`k_ratio` 기반)**:
# - 입력 시퀀스 길이에 비례하여 build() 단계에서 자동 결정
# - 예: `k_ratio=0.2`
#   - `(B, 256, 768)` → k = 51 (텍스트)
#   - `(B, 197, 768)` → k = 39 (이미지)
# 
# **미분 가능성**:
# - `tf.math.top_k` 자체는 미분 불가하지만, `tf.gather`로 가져온 선택된 토큰에는 gradient 정상 전달
# - `attention_fc` (Dense(1))도 선택된 토큰들의 학습 신호로 업데이트됨


class TopKTokenSelector(layers.Layer):
    """
    시퀀스 보존형 Top-K 토큰 선택 레이어
    (B, N, D) → (B, k, D)

    K값을 입력 시퀀스 길이의 비율(k_ratio)로 자동 계산.
    예: k_ratio=0.2, 입력 (B, 256, D) → k = 51
        k_ratio=0.2, 입력 (B, 197, D) → k = 39

    1) Dense(1)로 각 토큰의 중요도 스칼라 점수 계산
    2) padding mask 적용 (-1e9로 점수 매우 작게 → top-k에서 자동 배제)
    3) Top-K 인덱스 추출 (tf.math.top_k)
    4) tf.gather로 선택된 k개 토큰을 시퀀스로 보존 (가중합 없음)
    """
    def __init__(self, k_ratio=0.2, **kwargs):
        super().__init__(**kwargs)
        self.attention_fc = layers.Dense(1)
        self.k_ratio = k_ratio
        self.k = None   # build()에서 입력 시퀀스 길이로 자동 계산

    def build(self, input_shape):
        # input_shape: (B, N, D) - 입력 시퀀스 길이 N으로 K 자동 계산
        seq_len = input_shape[1]
        self.k = max(int(seq_len * self.k_ratio), 1)   # 최소 1 보장
        print(f"[TopKTokenSelector] seq_len={seq_len}, k_ratio={self.k_ratio} → k={self.k}")
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (B, N, D), mask: (B, N) or None

        # 1) 각 토큰의 점수 계산
        # 학습된 Dense(1)로 D차원 토큰 → 스칼라 점수 1개로 압축
        scores = self.attention_fc(x)              # (B, N, 1)
        scores = tf.squeeze(scores, axis=-1)       # (B, N)

        # 2) padding 위치 점수를 매우 작게
        # padding은 -1e9로 만들어 top-k에서 자동 배제
        if mask is not None:
            mask_f = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask_f) * -1e9

        # 3) Top-K 인덱스 추출 (build()에서 자동 계산된 self.k 사용)
        # 점수 기준 상위 K개의 인덱스 (점수 내림차순 정렬)
        _, top_k_idx = tf.math.top_k(scores, k=self.k)   # (B, k)

        # 4) 선택된 토큰 추출 — 가중합 없이 시퀀스 그대로 보존
        # 인덱스로 원본 토큰 벡터(D차원)를 그대로 모음
        top_k_tokens = tf.gather(x, top_k_idx, batch_dims=1)   # (B, k, D)

        return top_k_tokens


# ## 전체 모델


class MultimodalReviewHelpfulnessModel(keras.Model):
    """
Multimodal Review Helpfulness Prediction Model (Top-K + Dense projection + flatten + MLP)

전체 shape 흐름:

1단계: 인코더 (Frozen)
    input_ids: (B, 256)
    attention_mask: (B, 256)
    pixel_values: (B, 3, 224, 224)
    
    text_hidden: 13개 × (B, 256, 768)   ← RoBERTa 각 층 출력
    image_hidden: 13개 × (B, 197, 768)  ← ViT 각 층 출력

2단계: 층 집계 (Mean Pooling, L4/L8/L12 → 1)
    T_merged = mean(text_layers,  axis=0): (B, 256, 768)
    I_merged = mean(image_layers, axis=0): (B, 197, 768)
    → 시퀀스 차원은 보존, 층 차원만 3 → 1로 압축 (3:1, 정보 손실 작음)

3단계: BidirectionalCrossAttentionFusion (1번)
    t2i: (B, 256, 768)  ← 텍스트 토큰 + 이미지 cross-modal 문맥
    i2t: (B, 197, 768)  ← 이미지 패치 + 텍스트 cross-modal 문맥

4단계: Top-K 토큰 선택 (시퀀스 보존, k_ratio=0.2)
    t2i_topk = TopKTokenSelector(t2i, mask=text_mask): (B, 51, 768)
    i2t_topk = TopKTokenSelector(i2t, mask=None):       (B, 39, 768)

5단계: Concat (모달리티 결합)
    combined = concat([t2i_topk, i2t_topk], axis=1): (B, 90, 768)

6단계: Per-token Dense projection (768 → 64)
    projected = Dense(64, gelu)(combined): (B, 90, 64)
    LayerNorm 적용

7단계: Flatten + MLP Prediction Head
    flat = Flatten(projected): (B, 5760)   ← 90 × 64
    Dense(mlp_hidden, relu) + Dropout
    Dense(mlp_hidden//2, relu) + Dropout
    Dense(1): (B, 1)  ← 최종 helpfulness 점수
"""

    EXTRACT_LAYERS = [4, 8, 12]

    def __init__(
        self,
        roberta_name="roberta-base",
        vit_name="google/vit-base-patch16-224",
        d_model=768,
        num_heads=12,
        d_ff=3072, 
        dropout=0.1,
        mlp_hidden=256,
        k_ratio_text=0.2,
        k_ratio_image=0.2,
        proj_dim=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 1) encoder (frozen)
        self.text_encoder = TFRobertaModel.from_pretrained(
            roberta_name, output_hidden_states=True
        )
        self.image_encoder = TFViTModel.from_pretrained(
            vit_name, output_hidden_states=True
        )
        self.text_encoder.trainable = False
        self.image_encoder.trainable = False

        # 2) cross-attention (1 module, 층 집계 후 한 번만)
        self.cross_attn = BidirectionalCrossAttentionFusion(
            d_model, num_heads, d_ff, dropout, name="cross_attn"
        )

        # 3) Top-K token selector (모달리티별, 시퀀스 보존)
        self.text_selector = TopKTokenSelector(k_ratio=k_ratio_text, name="text_topk")
        self.image_selector = TopKTokenSelector(k_ratio=k_ratio_image, name="image_topk")

        # 4) Per-token Dense projection (차원 축소 768 → proj_dim)
        self.proj = layers.Dense(proj_dim, activation="gelu", name="proj")
        self.proj_ln = layers.LayerNormalization(name="proj_ln")

        # 5) Flatten + MLP head
        self.flatten = layers.Flatten(name="flatten")
        self.mlp = keras.Sequential([
            layers.Dense(mlp_hidden, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(mlp_hidden // 2, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),
        ], name="mlp_head")

    def call(self, inputs, training=False):
        # 1단계 텍스트(RoBERTa), 이미지(ViT) 인코더
        # backbone은 frozen이므로 dropout/normalization 비활성화 위해 training=False 고정
        text_out = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            training=False,
        )
        image_out = self.image_encoder(
            pixel_values=inputs["pixel_values"],
            training=False,
        )
        text_hidden = text_out.hidden_states     # 13 × (B, 256, 768)
        image_hidden = image_out.hidden_states   # 13 × (B, 197, 768)

        # 2단계 L4, L8, L12 mean-pooling으로 텍스트끼리/이미지끼리 1개로 집계
        # 시퀀스 차원은 보존, 층 차원만 평균 (3 → 1)
        text_layers = tf.stack(
            [text_hidden[idx] for idx in self.EXTRACT_LAYERS], axis=0
        )  # (3, B, 256, 768)
        image_layers = tf.stack(
            [image_hidden[idx] for idx in self.EXTRACT_LAYERS], axis=0
        )  # (3, B, 197, 768)

        T_merged = tf.reduce_mean(text_layers,  axis=0)  # (B, 256, 768)
        I_merged = tf.reduce_mean(image_layers, axis=0)  # (B, 197, 768)

        # 3단계 양방향 Cross-Attention (1번)
        # 출력: t2i (B, 256, 768), i2t (B, 197, 768) — 시퀀스 보존
        t2i, i2t = self.cross_attn(
            T_merged, I_merged, inputs["attention_mask"], training=training
        )

        # 4단계 Top-K 토큰 선택 (시퀀스 보존, 가중합 없이 토큰 그대로 추출)
        # 텍스트는 padding mask 적용, 이미지는 mask=None
        t2i_topk = self.text_selector(t2i, mask=inputs["attention_mask"])  # (B, 51, 768)
        i2t_topk = self.image_selector(i2t, mask=None)                     # (B, 39, 768)

        # 5단계 Concat (모달리티 결합) — 토큰 단위로 그대로 쌓음
        combined = tf.concat([t2i_topk, i2t_topk], axis=1)  # (B, 90, 768)

        # 6단계 Per-token Dense projection — 768 → proj_dim 차원 축소 + LayerNorm
        projected = self.proj(combined)               # (B, 90, proj_dim=64)
        projected = self.proj_ln(projected)

        # 7단계 Flatten + MLP → helpfulness 점수
        flat = self.flatten(projected)                # (B, 90·64 = 5760)
        score = self.mlp(flat, training=training)     # (B, 1)
        return score

configure_runtime()
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"DATA_PATH={DATA_PATH}", flush=True)
print(f"IMG_PATH={IMG_PATH}", flush=True)
print(f"SAVE_DIR={SAVE_DIR}", flush=True)
print(f"DATA_CACHE_DIR={DATA_CACHE_DIR}", flush=True)
print(f"BATCH_SIZE={BATCH_SIZE}", flush=True)
print(f"TUNE_FRAC={TUNE_FRAC}", flush=True)

# 모델 생성
model = MultimodalReviewHelpfulnessModel()

# 더미 입력으로 모델 빌드 & 확인 (실제 학습과 동일한 max_length=256)
dummy_inputs = {
    "input_ids": tf.random.uniform((2, MAX_LENGTH), maxval=50265, dtype=tf.int32),
    "attention_mask": tf.ones((2, MAX_LENGTH), dtype=tf.int32),
    "pixel_values": tf.random.normal((2, 3, 224, 224)),
}

output = model(dummy_inputs)
model.summary(print_fn=lambda line: print(line, flush=True))
print(f"Output shape: {output.shape}", flush=True)  # (2, 1)

df = load_or_create_tune_df(
    DATA_PATH,
    IMG_PATH,
    DATA_CACHE_DIR,
    frac=TUNE_FRAC,
    random_state=42,
    num_bins=TUNE_NUM_BINS,
)
print(df.head(), flush=True)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 데이터 준비
texts = df["clean_text"].tolist()
image_paths = df["image_path"].tolist()
labels = df["log_helpful_vote"].values.astype(np.float32)

# 1차 분할: train / test (80 / 20)
train_texts, test_texts, train_imgs, test_imgs, train_labels, test_labels = \
    train_test_split(texts, image_paths, labels, test_size=0.2, random_state=42)

# 2차 분할: train_inner / val (전체 기준 70 / 10 / 20)
train_texts, val_texts, train_imgs, val_imgs, train_labels, val_labels = \
    train_test_split(train_texts, train_imgs, train_labels, test_size=0.125, random_state=42)

# Dataset 생성
train_dataset = build_dataset(train_texts, train_imgs, train_labels, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = build_dataset(val_texts, val_imgs, val_labels, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
test_dataset = build_dataset(test_texts, test_imgs, test_labels, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)

print(f"Train: {len(train_labels)}개, Val: {len(val_labels)}개, Test: {len(test_labels)}개", flush=True)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="mean_squared_error", metrics=["mean_absolute_error", "mean_squared_error"])

early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True, mode="min")
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(SAVE_DIR, "CLAFRHP_7_epoch_{epoch:02d}.weights.h5"),
    monitor="val_loss",
    save_best_only=False,
    save_weights_only=True,
    verbose=1,
)

print("모델 학습 시작...", flush=True)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, early_stopping],
    verbose=1,
)
print("모델 학습 완료!", flush=True)

predicted_ratings = model.predict(test_dataset, verbose=1).flatten()
test_y = np.concatenate([y.numpy() for _, y in test_dataset])

mae = mean_absolute_error(test_y, predicted_ratings)
mse = mean_squared_error(test_y, predicted_ratings)
rmse = np.sqrt(mse)
mape = 100 * mean_absolute_percentage_error(test_y, predicted_ratings)

print(f"MAE: {mae:.4f}", flush=True)
print(f"MSE: {mse:.4f}", flush=True)
print(f"RMSE: {rmse:.4f}", flush=True)
print(f"MAPE: {mape:.4f}", flush=True)

total_elapsed = time.perf_counter() - total_start_time
hours, remainder = divmod(total_elapsed, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"\n총 실행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초", flush=True)
print(f"총 실행 시간(초): {total_elapsed:.2f}초", flush=True)
