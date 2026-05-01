import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")

import time
total_start_time = time.perf_counter() # 시작 시간 체크

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


# 설정 로드 + GPU에 작게 메모리 올려서 충돌 막기 + 재현성 고정
config = load_config("config.yaml")

configure_gpu_memory_growth()
hold_gpu_memory(size=4096)

seed_everything(config['training']['random_seed'])

# 1. 데이터 전처리
preprocessor = Preprocessor(config)
df_train, df_val, df_test = preprocessor.run()

# 2. BERT 임베딩
# 리뷰 텍스트 -> 768차원 BERT CLS 벡터
x_train, x_val, x_test = build_bert_embeddings(
    df_train['clean_text'].tolist(),
    df_val['clean_text'].tolist(),
    df_test['clean_text'].tolist(),
    config
)
train_label = df_train['log_vote'].values
val_label   = df_val['log_vote'].values
test_label  = df_test['log_vote'].values

# 3. 모델
model = build_model(config)
model.summary()

# 4. 학습
print("모델 학습 시작...")
model.fit(
    x_train, train_label,
    validation_data=(x_val, val_label),
    epochs=config['training']['epochs'],
    batch_size=config['training']['batch_size'],
    callbacks=[get_early_stopping()]
)
print("모델 학습 완료!")

# 5. 평가
predicted = model.predict(x_test)
print_metrics(calculate_metrics(test_label, predicted))

# 총 실행시간 출력
total_elapsed = time.perf_counter() - total_start_time
hours, remainder = divmod(total_elapsed, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"\n총 실행 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초")
print(f"총 실행 시간(초): {total_elapsed:.2f}초")
