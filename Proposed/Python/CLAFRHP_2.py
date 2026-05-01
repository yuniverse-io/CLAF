import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")

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


DATA_PATH = "Text/Bi-MPM_module/data/reviews_sample.jsonl.gz"
IMG_PATH = "Text/Bi-MPM_module/data/img"
SAVE_DIR = "./save_dir"

MAX_LENGTH = 256
BATCH_SIZE = 512
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


def build_dataset(texts, image_paths, labels, max_length=512, batch_size=256, shuffle=False):
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
    2) padding mask 적용
    3) softmax로 가중치 정규화
    4) 전체 토큰의 가중합
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_fc = layers.Dense(1)

    def call(self, x, mask=None):
        # x: (B, N, D), mask: (B, N) or None

        # 1) 각 토큰의 점수 계산
        scores = self.attention_fc(x)              # (B, N, 1)
        scores = tf.squeeze(scores, axis=-1)       # (B, N)

        # 2) padding 위치 점수를 매우 작게
        if mask is not None:
            mask_f = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask_f) * -1e9

        # 3) softmax로 가중치
        weights = tf.nn.softmax(scores, axis=-1)   # (B, N)

        # 4) 가중합
        pooled = tf.reduce_sum(
            tf.expand_dims(weights, axis=-1) * x,
            axis=1
        )                                           # (B, D)

        return pooled

class BidirectionalCrossAttentionFusion(layers.Layer):
    """
    같은 층의 T_l과 I_l을 받아서:
      1) t2i -> Text-to-Image Cross-Attention (Q=T, K=V=I)
      2) i2t -> Image-to-Text Cross-Attention (Q=I, K=V=T)
      3) 두 결과를 Attention Pooling -> Concat → Normalize -> H_l

    Cross-Attention + FFN 단계별 shape 정리
        입력:

        text_repr: (B, 512, 768)
        image_repr: (B, 197, 768)
        1단계: Text Branch

        t2i_attn = text_to_image_attn(Q=text, K=image, V=image): (B, 512, 768)
        t2i = LN(text_repr + t2i_attn): (B, 512, 768) (Residual + LN)
        t2i = LN(t2i + FFN(t2i)): (B, 512, 768) (FFN + Residual + LN)
        2단계: Image Branch

        i2t_attn = image_to_text_attn(Q=image, K=text, V=text): (B, 197, 768)
        i2t = LN(I_l + i2t_attn): (B, 197, 768) (Residual + LN)
        i2t = LN(i2t + FFN(i2t)): (B, 197, 768) (FFN + Residual + LN)
        3단계: Attention Pooling
        t2i_pooled = AttentionPool(t2i, mask=text_mask)
        i2t_pooled = AttentionPool(i2t)

        t2i_pooled = TopKAttentionPool(t2i, mask=text_mask): (B, 768)
        i2t_pooled = TopKAttentionPool(i2t): (B, 768)
        4단계: Concat (양방향 정보 결합)

        fused = concat([t2i_pooled, i2t_pooled]): (B, 1536)
        5단계: Final LayerNorm

        fused = fusion_ln(fused): (B, 1536)
        최종 출력: (B, 1536) = H_l 하나
    """

    def __init__(self, d_model=768, num_heads=12, d_ff=3072, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        # Cross-Attention
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

        # AttentionPool
        self.text_pool = AttentionPool()
        self.image_pool = AttentionPool()


        # 최종 fusion 후 LayerNorm
        self.fusion_ln = layers.LayerNormalization()

    def call(self, text_repr, image_repr, text_mask, training=False):
        # attention_mask 생성 
        # text_mask_cross(B, 1, 512)을 통해서 i2t 벡터를 만들때 text-padding 정보가 연산되는 것을 막음
        text_mask_cross = tf.expand_dims(text_mask, axis = 1)
        # 1) Text branch
        t2i_attn = self.text_to_image_attn(
            query=text_repr, key=image_repr, value=image_repr, training=training
        )
        t2i = self.ln1_t(text_repr + t2i_attn)
        t2i = self.ln2_t(t2i + self.ffn_t(t2i, training=training))

        # 2) Image branch
        i2t_attn = self.image_to_text_attn(
            query=image_repr, key=text_repr, value=text_repr, attention_mask = text_mask_cross, training=training
        )
        i2t = self.ln1_i(image_repr + i2t_attn)
        i2t = self.ln2_i(i2t + self.ffn_i(i2t, training=training))

        # 3) Attention Pooling + Concat
        t2i_pooled = self.text_pool(t2i, mask=text_mask)   # (B, 768)
        i2t_pooled = self.image_pool(i2t, mask=None)        # (B, 768)
        fused = tf.concat([t2i_pooled, i2t_pooled], axis=-1)  # (B, 1536)
        fused = self.fusion_ln(fused)

        return fused


class AttentivePooling(layers.Layer):
    """
__init__ -> layer 초기화
    self.gate_fc = layers.Dense(gate_dims, activation="sigmoid")
    -> 각 층의 1536차원 벡터에 대해 차원별 gate 값(0~1)을 생성하는 FC 층
    self.attention_fc = layers.Dense(1)
    -> gate가 적용된 벡터를 스칼라 점수 하나로 변환하는 학습 가능한 FC 층

call 과정:
    1) stack: 3개 층의 벡터(H4, H8, H12)를 층 차원으로 쌓음
        (B, 1536) × 3 -> (B, 3, 1536)

    2) Gate: Dense(1536, sigmoid)로 차원별 정보 흐름 제어
        (B, 3, 1536) -> (B, 3, 1536)
        -> 각 차원에서 0~1 사이 값으로 불필요한 정보를 억제

    3) gated: 원본 벡터에 gate 적용
        (B, 3, 1536) * (B, 3, 1536) -> (B, 3, 1536)

    4) Dense(1): gate된 벡터를 스칼라 점수로 변환
        (B, 3, 1536) -> (B, 3, 1)
        -> 차원이 정리된 상태에서 "어떤 층이 유용한지" 판단

    5) softmax(axis=1): 3개 층의 점수를 합이 1인 확률로 정규화
        (B, 3, 1) -> (B, 3, 1)
        -> 층별 중요도 α = [α_H4, α_H8, α_H12]

    6) weighted sum: 가중치를 gated 벡터에 곱하고 층 차원을 합산
        (B, 3, 1) * (B, 3, 1536) -> (B, 3, 1536) -> sum -> (B, 1536)

결과: sigmoid gate로 차원별 불필요한 정보를 먼저 억제한 뒤,
     softmax attention으로 층별 중요도를 판단하여
     3개 층의 정보를 하나의 1536차원 벡터로 압축
"""

    def __init__(self, gate_dims=1536, **kwargs):
        super().__init__(**kwargs)
        self.attention_fc = layers.Dense(1)
        self.gate_fc = layers.Dense(gate_dims, activation="sigmoid")

    def call(self, layer_representations):
        stacked = tf.stack(layer_representations, axis=1)
        gates = self.gate_fc(stacked)
        gated = gates * stacked
        scores = self.attention_fc(gated)
        weights = tf.nn.softmax(scores, axis=1)
        h_final = tf.reduce_sum(weights * gated, axis=1)
        return h_final


class MultimodalReviewHelpfulnessModel(keras.Model):
    """
Multimodal Review Helpfulness Prediction Model

전체 shape 흐름:

1단계: 인코더 (Frozen)
    input_ids: (B, 512)
    attention_mask: (B, 512)
    pixel_values: (B, 3, 224, 224)

    text_hidden: 13개 × (B, 512, 768)   ← RoBERTa 각 층 출력
    image_hidden: 13개 × (B, 197, 768)  ← ViT 각 층 출력

2단계: BidirectionalCrossAttentionFusion (×3회, L4/L8/L12)
    T_l: (B, 512, 768)  ← 텍스트 hidden state
    I_l: (B, 197, 768)  ← 이미지 hidden state

    [Text Branch]
        Cross-Attention + Residual + LN:
            t2i_attn = CrossAttn(Q=T, K=I, V=I): (B, 512, 768)
            t2i = LN(T_l + t2i_attn): (B, 512, 768)
        FFN + Residual + LN:
            t2i = LN(t2i + FFN(t2i)): (B, 512, 768)
            (FFN: Dense(3072, gelu) → Dropout → Dense(768))

    [Image Branch]
        Cross-Attention + Residual + LN:
            i2t_attn = CrossAttn(Q=I, K=T, V=T): (B, 197, 768)
            i2t = LN(I_l + i2t_attn): (B, 197, 768)
        FFN + Residual + LN:
            i2t = LN(i2t + FFN(i2t)): (B, 197, 768)

    [Fusion]
        Mean Pooling:
            t2i_pooled = mean(t2i, axis=1): (B, 768)
            i2t_pooled = mean(i2t, axis=1): (B, 768)
        Concat + LayerNorm:
            H_l = fusion_LN(concat([t2i_pooled, i2t_pooled])): (B, 1536)

3단계: Attentive Pooling
    stacked = stack([H4, H8, H12]): (B, 3, 1536)
    scores = Dense(1)(stacked): (B, 3, 1)
    weights = softmax(scores, axis=1): (B, 3, 1)
    h_final = sum(weights * stacked, axis=1): (B, 1536)

4단계: MLP Prediction Head
    Dense(256, relu): (B, 256)
    Dropout(0.1)
    Dense(128, relu): (B, 128)
    Dropout(0.1)
    Dense(1): (B, 1)  ← 최종 helpfulness 점수
"""

    EXTRACT_LAYERS = [4, 8, 12]

    def __init__(
        # 모델 초기화
        self,
        roberta_name="roberta-base",
        vit_name="google/vit-base-patch16-224",
        d_model=768,
        num_heads=12,
        d_ff=3072,
        dropout=0.1,
        mlp_hidden=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 1) encoder
        self.text_encoder = TFRobertaModel.from_pretrained(
            roberta_name, output_hidden_states=True
        )
        self.image_encoder = TFViTModel.from_pretrained(
            vit_name, output_hidden_states=True
        )
        self.text_encoder.trainable = False
        self.image_encoder.trainable = False

        # 2) cross-attention
        self.cross_attn_layers = []
        for l in self.EXTRACT_LAYERS:
            self.cross_attn_layers.append(
                BidirectionalCrossAttentionFusion(d_model, num_heads, d_ff, dropout, name=f"cross_attn_L{l}")
            )

        # 3) attentive pooling
        self.attentive_pooling = AttentivePooling()

        # 4) MLP
        self.mlp = keras.Sequential([
            layers.Dense(mlp_hidden, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(mlp_hidden // 2, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),
        ], name="mlp_head")

    def call(self, inputs, training=False):
        # 1단계 텍스트(RoBERTa), 이미지(VIT) 인코더로 각 layer의 임베딩 벡터 출력
        text_out = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            training=training,
        )
        image_out = self.image_encoder(
            pixel_values=inputs["pixel_values"],
            training=training,
        )
        # 각 층의 임베딩 백터를 text_hidden, image_hidden에 저장
        text_hidden = text_out.hidden_states
        image_hidden = image_out.hidden_states

        # 2단계 4, 8, 12 layer를 추출하고 BidirectionalCrossAttentionFusion 수행 후 multimodal_reprs에 저장
        multimodal_reprs = []
        # text_hidden, image_hidden의 4,8,12를 self.cross_attn_layers 0, 1, 2으로 cross-attention을 진행한 후 multimodal_reprs에 저장
        for i, idx in enumerate(self.EXTRACT_LAYERS):
            T_l = text_hidden[idx]
            I_l = image_hidden[idx]
            H_l = self.cross_attn_layers[i](T_l, I_l, inputs["attention_mask"], training=training)
            multimodal_reprs.append(H_l)

        # 3단계 multimodal_reprs(B, 3, 1536) -> (B, 1536)을 Attentive Pooling으로 적용
        h_final = self.attentive_pooling(multimodal_reprs)

        # 4단계 MLP층에 넣어서 최종 score 출력
        score = self.mlp(h_final, training=training)
        return score


configure_runtime()
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"DATA_PATH={DATA_PATH}", flush=True)
print(f"IMG_PATH={IMG_PATH}", flush=True)
print(f"SAVE_DIR={SAVE_DIR}", flush=True)
print(f"BATCH_SIZE={BATCH_SIZE}", flush=True)

# 모델 생성
model = MultimodalReviewHelpfulnessModel()

# 더미 입력으로 모델 빌드 & 확인
dummy_inputs = {
    "input_ids": tf.random.uniform((2, 128), maxval=50265, dtype=tf.int32),
    "attention_mask": tf.ones((2, 128), dtype=tf.int32),
    "pixel_values": tf.random.normal((2, 3, 224, 224)),
}

output = model(dummy_inputs)
model.summary(print_fn=lambda line: print(line, flush=True))
print(f"Output shape: {output.shape}", flush=True)  # (2, 1)

df = load_and_preprocess(DATA_PATH)
df = map_existing_images(df, IMG_PATH)
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
    filepath=os.path.join(SAVE_DIR, "CLAFRHP_epoch_{epoch:02d}.weights.h5"),
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