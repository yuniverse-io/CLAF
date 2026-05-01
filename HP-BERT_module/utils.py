# 잡다한 함수들 모음

import random
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

GPU_HOLDER = None

def configure_gpu_memory_growth():
    """ GPU 메모리를 필요한 만큼만 잡도록 설정하는 거 """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU 인식 실패!", flush=True)
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"memory growth 설정 스킵: {exc}", flush=True)

    print(f"TensorFlow에서 인식한 GPU 수: {len(gpus)}", flush=True)


def hold_gpu_memory(size=4096):
    """ 데이터 로딩 전에 작은 텐서를 GPU에 올려 nvidia-smi에 표시되게 하는용 약 64메가 정도됨 큰 영향 없음"""
    global GPU_HOLDER

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU 인식 실패!", flush=True)
        return

    with tf.device("/GPU:0"):
        GPU_HOLDER = tf.Variable(
            tf.zeros([size, size], dtype=tf.float32),
            trainable=False,
            name="gpu_holder",
        )
        _ = tf.reduce_sum(GPU_HOLDER).numpy()

    print(f"GPU holder 활성화 완료: size={size}", flush=True)


def load_config(path="config.yaml"):

    """YAML 설정 파일을 읽어 dict로 반환"""

    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed):

    """재현성 위해서 시드 고정하기"""

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_early_stopping():

    """EarlyStopping 반환"""

    return EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )


def calculate_metrics(test_labels, predicted):

    """MAE, MSE, RMSE, MAPE 계산 후 dict로 반환"""

    mae  = mean_absolute_error(test_labels, predicted)
    mse  = mean_squared_error(test_labels, predicted)
    rmse = np.sqrt(mse)
    mape = 100 * mean_absolute_percentage_error(test_labels, predicted)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def print_metrics(metrics: dict):

    """계산된 지표를 터미널에 출력"""

    print("\n===== 평가 결과 =====")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("====================")
