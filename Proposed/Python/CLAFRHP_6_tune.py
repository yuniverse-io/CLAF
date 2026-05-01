import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # GPU 메모리 단편화 완화용
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
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 1e-4

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

class AttentionPool(layers.Layer):
    """
    시퀀스 압축용 기본 Attention Pooling
    (B, N, D) → (B, D)

    1) Dense(1)로 각 토큰의 중요도 스칼라 점수 계산
    2) padding mask 적용 (-1e9로 점수 매우 작게)
    3) softmax로 가중치 정규화
    4) 전체 토큰의 가중합
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_fc = layers.Dense(1)

    def call(self, x, mask=None):
        # x: (B, N, D), mask: (B, N) or None

        # 1) 각 토큰의 점수 계산 (학습된 Dense(1)로 D차원 → 스칼라)
        scores = self.attention_fc(x)              # (B, N, 1)
        scores = tf.squeeze(scores, axis=-1)       # (B, N)

        # 2) padding 위치 점수를 매우 작게
        if mask is not None:
            mask_f = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask_f) * -1e9

        # 3) softmax로 가중치 정규화 (합=1)
        weights = tf.nn.softmax(scores, axis=-1)   # (B, N)

        # 4) 가중합 (스칼라 가중치 × D차원 토큰 벡터)
        pooled = tf.reduce_sum(
            tf.expand_dims(weights, axis=-1) * x,
            axis=1
        )                                           # (B, D)

        return pooled


class MultiSlotAttentionPool(layers.Layer):
    """
    Multi-slot Attention Pooling
    (B, N, D) → (B, S, D)

    기존 AttentionPool은 시퀀스 전체를 1개의 벡터로 압축한다.
        (B, 256, 768) → (B, 768)

    이 레이어는 같은 시퀀스를 S개의 서로 다른 attention 분포로 요약한다.
        (B, 256, 768) → (B, 4, 768)

    핵심 아이디어:
        1) Dense(S)로 각 토큰마다 S개의 slot score를 계산
        2) slot별로 토큰 방향 softmax를 적용
        3) 각 slot이 전체 토큰/패치 시퀀스를 서로 다른 가중치로 요약

    예시:
        slot 1 = 전체 256개 토큰을 보고 만든 요약 벡터
        slot 2 = 전체 256개 토큰을 또 다른 가중치로 보고 만든 요약 벡터
        slot 3 = 전체 256개 토큰을 또 다른 가중치로 보고 만든 요약 벡터
        slot 4 = 전체 256개 토큰을 또 다른 가중치로 보고 만든 요약 벡터

    즉 256개를 4등분하는 것이 아니라,
    4개의 학습 가능한 scoring vector가 같은 256개 전체를 각각 다르게 본다.
    """
    def __init__(self, num_slots=4, **kwargs):
        super().__init__(**kwargs)
        self.num_slots = num_slots

        # Dense(num_slots)는 slot마다 다른 scoring vector를 가진다.
        # 기존 Dense(1)이 attention 분포 1개를 만들었다면,
        # Dense(4)는 attention 분포 4개를 동시에 만든다.
        self.slot_score_fc = layers.Dense(num_slots)

    def call(self, x, mask=None):
        # x: (B, N, D)
        #   텍스트라면 N=256, 이미지는 N=197
        # mask: (B, N) or None

        # 1) 각 토큰/패치마다 slot별 점수 계산
        #    scores[b, n, s] = n번째 토큰이 s번째 slot에 얼마나 중요한지
        scores = self.slot_score_fc(x)              # (B, N, S)

        # 2) slot을 앞으로 옮겨서 slot별 token attention을 계산하기 쉽게 만든다.
        #    (B, N, S) → (B, S, N)
        scores = tf.transpose(scores, [0, 2, 1])    # (B, S, N)

        # 3) 텍스트 padding 위치는 attention을 받지 못하도록 매우 작은 점수를 더한다.
        #    이미지 patch에는 padding이 없으므로 mask=None으로 사용한다.
        if mask is not None:
            mask_f = tf.cast(tf.expand_dims(mask, axis=1), scores.dtype)  # (B, 1, N)
            scores = scores + (1.0 - mask_f) * -1e9

        # 4) 각 slot마다 N개 토큰/패치에 대한 attention 분포를 만든다.
        #    weights[b, s, :]의 합은 1이다.
        weights = tf.nn.softmax(scores, axis=-1)    # (B, S, N)

        # 5) slot별 가중합.
        #    weights: (B, S, N), x: (B, N, D)
        #    결과: 각 slot이 만든 D차원 요약 벡터 S개
        pooled = tf.matmul(weights, x)              # (B, S, D)

        return pooled


class MFB(layers.Layer):
    """
    Multi-modal Factorized Bilinear Pooling (Yu et al., ICCV 2017) — 표준 구현

    두 모달리티 벡터를 bilinear pooling으로 융합.

    Bilinear weight를 W ≈ U·V^T 로 분해 (factorize)하여 파라미터 폭발 방지:
        T (B, D_T) → proj_t → (B, k_factors · mfb_dim)
        I (B, D_I) → proj_i → (B, k_factors · mfb_dim)
        hadamard product → (B, k_factors · mfb_dim)
        reshape → (B, mfb_dim, k_factors)
        sum over k_factors → (B, mfb_dim)
        power norm + L2 norm → (B, mfb_dim)

    출력: (B, mfb_dim)  ← 추가 projection 없음 (표준 MFB)

    Hyperparameters:
        mfb_dim: 최종 출력 차원 (= bilinear interaction 패턴 개수)
        k_factors: 각 패턴의 rank (factorization 정도)
    """
    def __init__(self, mfb_dim=1000, k_factors=5, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mfb_dim = mfb_dim
        self.k_factors = k_factors

        # Expand stage: T, I → k_factors × mfb_dim 차원으로 projection
        # use_bias=False — 표준 MFB 논문 / Ren et al. 2024 따름
        self.proj_t = layers.Dense(k_factors * mfb_dim, use_bias=False)
        self.proj_i = layers.Dense(k_factors * mfb_dim, use_bias=False)

        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        # inputs: [text_vec, image_vec], 각 (B, D)
        T, I = inputs

        # 1) Expand stage — k·mfb_dim 차원 factor space로 projection
        proj_t = self.proj_t(T)   # (B, k_factors × mfb_dim)
        proj_i = self.proj_i(I)   # (B, k_factors × mfb_dim)

        # 2) Hadamard product (element-wise multiply, bilinear interaction)
        hadamard = tf.multiply(proj_t, proj_i)               # (B, k_factors × mfb_dim)
        hadamard = self.dropout(hadamard, training=training)

        # 3) Squeeze stage — Sum pool: k개 factor를 합산
        # (B, k·mfb_dim) → (B, mfb_dim, k_factors) → sum axis=-1 → (B, mfb_dim)
        sum_pooled = tf.reduce_sum(
            tf.reshape(hadamard, (-1, self.mfb_dim, self.k_factors)),
            axis=-1,
        )                                                     # (B, mfb_dim)

        # 4) Power normalization: z ← sign(z) · sqrt(|z|)
        sum_pooled = tf.sign(sum_pooled) * tf.sqrt(tf.abs(sum_pooled) + 1e-8)

        # 5) L2 normalization: z ← z / ‖z‖
        sum_pooled = tf.math.l2_normalize(sum_pooled, axis=-1)

        return sum_pooled                                     # (B, mfb_dim)


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
        이후 단계인 시퀀스 → 단일 벡터 압축 (AttentionPool)과
        모달리티 간 융합 (MFB)은 외부 모듈로 분리되어 모델 클래스에서 호출됨
        (모듈별 책임 분리, 각 컴포넌트 교체 용이).
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


class MultimodalReviewHelpfulnessModel(keras.Model):
    """
Multimodal Review Helpfulness Prediction Model
(Mean-layer Cross-Attention + Multi-slot Pooling + Slot-wise MFB)

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
    → 시퀀스 차원은 보존, 층 차원만 3 → 1로 압축

3단계: BidirectionalCrossAttentionFusion (1번)
    t2i: (B, 256, 768)  ← 텍스트 토큰 + 이미지 cross-modal 문맥
    i2t: (B, 197, 768)  ← 이미지 패치 + 텍스트 cross-modal 문맥

4단계: Multi-slot Attention Pooling
    기존 모델:
        t2i: (B, 256, 768) → (B, 768)
        i2t: (B, 197, 768) → (B, 768)

    6 모델:
        text_slots  = MultiSlotAttentionPool(t2i): (B, 4, 768)
        image_slots = MultiSlotAttentionPool(i2t): (B, 4, 768)

    의미:
        리뷰/이미지를 1개의 전역 벡터로 바로 뭉개지 않고,
        4개의 학습 가능한 attention slot으로 서로 다른 요약 관점을 유지한다.

5단계: Slot-wise Shared MFB Fusion
    text_slots:  (B, 4, 768)
    image_slots: (B, 4, 768)

    같은 slot 번호끼리 MFB 적용:
        fused_slot_1 = MFB(text_slot_1, image_slot_1)
        fused_slot_2 = MFB(text_slot_2, image_slot_2)
        fused_slot_3 = MFB(text_slot_3, image_slot_3)
        fused_slot_4 = MFB(text_slot_4, image_slot_4)

    구현은 B와 slot을 합쳐 shared MFB 1개를 재사용:
        (B, 4, 768) → (B*4, 768)
        MFB: (B*4, 768), (B*4, 768) → (B*4, mfb_dim)
        reshape → (B, 4, mfb_dim)

6단계: Slot Attention Pooling
    fused_slots: (B, 4, mfb_dim)
    final_fused = AttentionPool(fused_slots): (B, mfb_dim)

    4개의 멀티모달 관점 중 helpfulness 예측에 중요한 관점을 다시 가중합한다.

7단계: MLP Prediction Head
    Dense(mlp_hidden, relu): (B, mlp_hidden)
    Dropout(dropout)
    Dense(mlp_hidden//2, relu)
    Dropout(dropout)
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
        mfb_dim=1000,
        k_factors=5,
        num_slots=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.mfb_dim = mfb_dim
        self.num_slots = num_slots

        # 1) encoder (frozen)
        self.text_encoder = TFRobertaModel.from_pretrained(
            roberta_name, output_hidden_states=True
        )
        self.image_encoder = TFViTModel.from_pretrained(
            vit_name, output_hidden_states=True
        )
        self.text_encoder.trainable = False
        self.image_encoder.trainable = False

        # 2) cross-attention (5.py와 동일하게 층 집계 후 한 번만 수행)
        self.cross_attn = BidirectionalCrossAttentionFusion(
            d_model, num_heads, d_ff, dropout, name="cross_attn"
        )

        # 3) Multi-slot pooling
        #    텍스트/이미지 시퀀스를 각각 1개 벡터가 아니라 num_slots개 벡터로 압축한다.
        #    각 slot은 전체 토큰/패치를 보지만 서로 다른 학습 score로 가중합한다.
        self.text_slot_pool = MultiSlotAttentionPool(num_slots=num_slots, name="text_slot_pool")
        self.image_slot_pool = MultiSlotAttentionPool(num_slots=num_slots, name="image_slot_pool")

        # 4) shared slot-wise MFB
        #    slot 4개마다 MFB 레이어를 따로 만들지 않고 같은 MFB를 공유한다.
        #    그래서 파라미터는 1개 MFB만큼만 증가하고, 연산만 slot 수만큼 반복된다.
        self.mfb = MFB(
            mfb_dim=mfb_dim,
            k_factors=k_factors,
            dropout=dropout,
            name="shared_slot_mfb",
        )

        # 5) fused slot pooling
        #    slot-wise MFB 결과 (B, 4, mfb_dim)를 최종 1개 벡터로 합친다.
        #    여기서는 기존 AttentionPool을 재사용한다.
        self.slot_pool = AttentionPool(name="slot_pool")

        # 6) MLP head
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

        T_merged = tf.reduce_mean(text_layers, axis=0)   # (B, 256, 768)
        I_merged = tf.reduce_mean(image_layers, axis=0)  # (B, 197, 768)

        # 3단계 양방향 Cross-Attention (1번)
        # 출력: t2i (B, 256, 768), i2t (B, 197, 768) — 시퀀스 보존
        t2i, i2t = self.cross_attn(
            T_merged, I_merged, inputs["attention_mask"], training=training
        )

        # 4단계 Multi-slot Attention Pooling
        #    5.py는 여기서 (B, 256, 768) → (B, 768)로 바로 줄였다.
        #    6.py는 (B, 256, 768) → (B, 4, 768)로 줄여서 4개의 요약 관점을 남긴다.
        text_slots = self.text_slot_pool(t2i, mask=inputs["attention_mask"])  # (B, S, 768)
        image_slots = self.image_slot_pool(i2t, mask=None)                    # (B, S, 768)

        # 5단계 Slot-wise Shared MFB
        #    같은 slot 번호끼리 MFB를 적용하기 위해 batch 축과 slot 축을 잠시 합친다.
        #    text_slots[:, 0, :] ↔ image_slots[:, 0, :]
        #    text_slots[:, 1, :] ↔ image_slots[:, 1, :]
        #    ...
        #    이렇게 B*S개의 쌍을 shared MFB 1개에 한 번에 넣는다.
        batch_size = tf.shape(text_slots)[0]

        text_flat = tf.reshape(text_slots, [-1, self.d_model])    # (B*S, 768)
        image_flat = tf.reshape(image_slots, [-1, self.d_model])  # (B*S, 768)

        # shared MFB 1개를 모든 slot 쌍에 적용
        fused_flat = self.mfb([text_flat, image_flat], training=training)  # (B*S, mfb_dim)

        # 다시 slot 구조로 복원
        fused_slots = tf.reshape(
            fused_flat,
            [batch_size, self.num_slots, self.mfb_dim],
        )  # (B, S, mfb_dim)

        # 6단계 Slot Attention Pooling
        #    4개의 멀티모달 slot 중 helpfulness 예측에 중요한 slot에 더 큰 가중치를 준다.
        final_fused = self.slot_pool(fused_slots, mask=None)  # (B, mfb_dim)

        # 7단계 MLP → helpfulness 점수
        score = self.mlp(final_fused, training=training)      # (B, 1)
        return score


configure_runtime()
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"DATA_PATH={DATA_PATH}", flush=True)
print(f"IMG_PATH={IMG_PATH}", flush=True)
print(f"SAVE_DIR={SAVE_DIR}", flush=True)
print(f"DATA_CACHE_DIR={DATA_CACHE_DIR}", flush=True)
print(f"BATCH_SIZE={BATCH_SIZE}", flush=True)

# 모델 생성
model = MultimodalReviewHelpfulnessModel()

# 더미 입력으로 모델 빌드 & 확인 (실제 학습과 동일한 max_length=256)
dummy_inputs = {
    "input_ids": tf.random.uniform((2, MAX_LENGTH), maxval=50265, dtype=tf.int32),
    "attention_mask": tf.ones((2, MAX_LENGTH), dtype=tf.int32),
    "pixel_values": tf.random.normal((2, 3, 224, 224)),
}


def load_or_create_tune_df(
    data_path,
    image_dir,
    cache_dir,
    frac=0.1,
    random_state=42,
    num_bins=10,
):
    """
    전체 데이터를 매번 전처리/매핑하지 않기 위한 튜닝용 캐시 로더.

    1) cache_dir에 10% 튜닝 df pickle이 있으면 바로 로드
    2) 없으면 전체 데이터 전처리 + 기존 이미지 매핑 수행
    3) log_helpful_vote 분포가 너무 한쪽으로 쏠리지 않도록 qcut bin 기준 stratify sampling
    4) 생성한 튜닝 df를 pickle로 저장

    모델 구조는 CLAFRHP_6 그대로 두고, 데이터 양만 줄여 빠르게 시간/성능을 확인하는 용도다.
    """
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

    # helpfulness 값의 분포를 최대한 유지하기 위해 label을 분위수 bin으로 나눈다.
    # duplicates='drop'은 동일 값이 많아 bin 경계가 겹칠 때 자동으로 bin 수를 줄인다.
    df["_label_bin"] = pd.qcut(
        df["log_helpful_vote"],
        q=num_bins,
        duplicates="drop",
    )

    try:
        df_tune, _ = train_test_split(
            df,
            train_size=frac,
            random_state=random_state,
            stratify=df["_label_bin"],
        )
    except ValueError as exc:
        # bin별 샘플 수가 너무 작으면 stratify가 실패할 수 있다.
        # 그때는 전체 분포를 완벽히 보장하진 못하지만, 작은 실험을 계속하기 위해 랜덤 샘플링으로 fallback.
        print(f"stratify sampling 실패, 랜덤 샘플링으로 대체: {exc}", flush=True)
        df_tune = df.sample(frac=frac, random_state=random_state)

    df_tune = df_tune.drop(columns=["_label_bin"]).reset_index(drop=True)
    df_tune.to_pickle(tune_df_path)

    print(f"튜닝 데이터 새로 생성 및 저장 완료: {tune_df_path}", flush=True)
    print(f"원본 유효 데이터: {len(df)}개", flush=True)
    print(f"튜닝 데이터: {len(df_tune)}개", flush=True)
    print(df_tune["log_helpful_vote"].describe(), flush=True)

    return df_tune


output = model(dummy_inputs)
model.summary(print_fn=lambda line: print(line, flush=True))
print(f"Output shape: {output.shape}", flush=True)  # (2, 1)

df = load_or_create_tune_df(
    DATA_PATH,
    IMG_PATH,
    DATA_CACHE_DIR,
    frac=0.1,
    random_state=42,
    num_bins=10,
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
    filepath=os.path.join(SAVE_DIR, "CLAFRHP_6_tune_epoch_{epoch:02d}.weights.h5"),
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
