import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import (
    load_config,
    seed_everything,
    get_early_stopping,
    calculate_metrics,
    print_metrics,
    configure_gpu_memory_growth,
    hold_gpu_memory,
)
from model.model import build_model
from src.preprocess import Preprocessor
from src.bert_encoder import build_bert_embeddings
from src.vgg16_encoder import build_vgg16_embeddings


def normalize_feature(train_vals, val_vals, test_vals):

    """train으로 fit한 StandardScaler를 train/val/test에 각각 적용"""

    scaler      = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).reshape(-1)
    val_scaled   = scaler.transform(val_vals.reshape(-1, 1)).reshape(-1)
    test_scaled  = scaler.transform(test_vals.reshape(-1, 1)).reshape(-1)

    return train_scaled, val_scaled, test_scaled


# 설정 로드 + GPU에 작게 메모리 올려서 충돌 막기 + 재현성 고정
config = load_config("config.yaml")

configure_gpu_memory_growth()
hold_gpu_memory(size=4096)

seed_everything(config['training']['random_seed'])

# 1. 데이터 전처리 (텍스트 + 이미지 URL + 수작업 피처)
preprocessor = Preprocessor(config)
df_train, df_val, df_test = preprocessor.run()

print("전처리 후 컬럼:", df_train.columns.tolist())
print("Train 샘플 수:", len(df_train), "Val 샘플 수:", len(df_val), "Test 샘플 수:", len(df_test))

# 2. BERT 인코딩 (리뷰 텍스트 → 768-dim CLS 벡터)
bert_train, bert_val, bert_test = build_bert_embeddings(
    df_train['clean_text'].tolist(),
    df_val['clean_text'].tolist(),
    df_test['clean_text'].tolist(),
    config
)

# 3. VGG16 인코딩 + 이미지 수작업 피처
(vgg_train,    vgg_val,    vgg_test,
 pixels_train, pixels_val, pixels_test,
 bright_train, bright_val, bright_test) = build_vgg16_embeddings(
    df_train.index.tolist(), df_train['image_url'].tolist(),
    df_val.index.tolist(),   df_val['image_url'].tolist(),
    df_test.index.tolist(),  df_test['image_url'].tolist(),
    config
)

# 4. 수작업 피처 정규화 (train 기준 fit → val/test에 적용)
rev_len_train, rev_len_val, rev_len_test = normalize_feature(
    df_train['review_length'].values.astype(float),
    df_val['review_length'].values.astype(float),
    df_test['review_length'].values.astype(float)
)
fog_train, fog_val, fog_test = normalize_feature(
    df_train['gunning_fog_index'].values.astype(float),
    df_val['gunning_fog_index'].values.astype(float),
    df_test['gunning_fog_index'].values.astype(float)
)
pixels_train, pixels_val, pixels_test = normalize_feature(
    pixels_train, pixels_val, pixels_test
)
bright_train, bright_val, bright_test = normalize_feature(
    bright_train, bright_val, bright_test
)

train_label = df_train['log_vote'].values
val_label   = df_val['log_vote'].values
test_label  = df_test['log_vote'].values

# 5. 모델
model = build_model(config)
model.summary()

# 6. 학습
print("모델 학습 시작...")
model.fit(
    [rev_len_train, fog_train, bert_train, pixels_train, bright_train, vgg_train],
    train_label,
    validation_data=(
        [rev_len_val, fog_val, bert_val, pixels_val, bright_val, vgg_val],
        val_label
    ),
    epochs=config['training']['epochs'],
    batch_size=config['training']['batch_size'],
    callbacks=[get_early_stopping()]
)
print("모델 학습 완료!")

# 7. 평가
predicted = model.predict(
    [rev_len_test, fog_test, bert_test, pixels_test, bright_test, vgg_test]
)
print_metrics(calculate_metrics(test_label, predicted))
