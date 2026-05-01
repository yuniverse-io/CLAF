import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")

import time
total_start_time = time.perf_counter() # 시작 시간 체크

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
from src.end_to_end_inputs import (
    build_text_inputs,
    build_image_inputs,
    build_dataset,
)


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

# 2. BERT 입력 생성 (리뷰 텍스트 → input_ids / attention_mask)
text_train, text_val, text_test = build_text_inputs(
    df_train['clean_text'].tolist(),
    df_val['clean_text'].tolist(),
    df_test['clean_text'].tolist(),
    config
)

# 3. 이미지 입력 준비 (이미지 path + 수작업 피처)
(img_train, pixels_train, bright_train, valid_train) = build_image_inputs(
    df_train.index.tolist(), df_train['image_url'].tolist(), config, split_name='Train'
)
(img_val, pixels_val, bright_val, valid_val) = build_image_inputs(
    df_val.index.tolist(), df_val['image_url'].tolist(), config, split_name='Val'
)
(img_test, pixels_test, bright_test, valid_test) = build_image_inputs(
    df_test.index.tolist(), df_test['image_url'].tolist(), config, split_name='Test'
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

# 5. tf.data.Dataset 구성
train_dataset = build_dataset(
    rev_len_train, fog_train, text_train,
    pixels_train, bright_train, img_train, valid_train,
    train_label, config, shuffle=True
)
val_dataset = build_dataset(
    rev_len_val, fog_val, text_val,
    pixels_val, bright_val, img_val, valid_val,
    val_label, config
)
test_dataset = build_dataset(
    rev_len_test, fog_test, text_test,
    pixels_test, bright_test, img_test, valid_test,
    test_label, config
)

# 6. 모델
model = build_model(config)
model.summary()

# 7. 학습
print("모델 학습 시작...")
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config['training']['epochs'],
    callbacks=[get_early_stopping()]
)
print("모델 학습 완료!")

# 8. 평가
predicted = model.predict(test_dataset)
print_metrics(calculate_metrics(test_label, predicted))

# 총 실행시간 출력
total_elapsed = time.perf_counter() - total_start_time
hours, remainder = divmod(total_elapsed, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"\n총 실행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초")
print(f"총 실행 시간(초): {total_elapsed:.2f}초")
