# frozen end-to-end 학습용 입력을 만드는 파일
# BERT/VGG16 피처를 미리 뽑지 않고, 토큰과 이미지 텐서를 모델로 넘긴다.

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import glob
import hashlib
import io
import os

import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from transformers import BertTokenizer


def build_text_inputs(train_review, val_review, test_review, config):

    """리뷰 텍스트를 BERT input_ids / attention_mask로 변환"""

    print(f"BERT 토크나이저 로딩 중: {config['model']['bert_model_name']}")
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])

    def encode(texts, split_name):
        print(f"{split_name} 토큰화 중... (총 {len(texts)}개 문장)")
        encoded = tokenizer(
            list(texts),
            padding='max_length',
            truncation=True,
            max_length=config['model']['bert_max_len'],
            return_tensors='np'
        )
        return {
            'input_ids': encoded['input_ids'].astype(np.int32),
            'attention_mask': encoded['attention_mask'].astype(np.int32),
        }

    return (
        encode(train_review, 'Train'),
        encode(val_review,   'Val'),
        encode(test_review,  'Test'),
    )


def _cache_path(row_id, url, image_dir):

    """기존 CLAFRHP/MFRHP 이미지 캐시 파일명 규칙 사용"""

    h = hashlib.md5((url + str(row_id)).encode()).hexdigest()[:6]
    exact_jpg = os.path.join(image_dir, f"{row_id}_{h}.jpg")

    if os.path.exists(exact_jpg):
        return exact_jpg

    candidates = sorted(glob.glob(os.path.join(image_dir, f"{row_id}_{h}.*")))
    if candidates:
        return candidates[0]

    return exact_jpg


def _load_image_from_cache_or_url(row_id, url, image_dir):

    """
    1) 로컬 캐시가 있으면 그 이미지 사용
    2) 없으면 URL에서 다운로드
    3) 다운로드 성공 시 캐시에 저장
    """

    save_path = _cache_path(row_id, url, image_dir)

    if os.path.exists(save_path):
        img = Image.open(save_path).convert('RGB')
        return img, save_path

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content)).convert('RGB')

    os.makedirs(image_dir, exist_ok=True)
    img.save(save_path, format='JPEG')

    return img, save_path


def build_image_inputs(row_ids, urls, config, split_name='Data'):

    """이미지 경로 + pixels/brightness 수작업 피처 생성"""

    image_dir = config['data']['image_cache_dir']
    image_paths = []
    pixels_list = []
    brightness_list = []
    valid_list = []

    print(f"{split_name} 이미지 준비 중... (총 {len(urls)}개 이미지)")

    for row_id, url in zip(row_ids, urls):
        try:
            img, path = _load_image_from_cache_or_url(row_id, url, image_dir)
            pixels = float(img.width * img.height)
            brightness = float(np.mean(np.array(img.convert('L'))))
            valid = 1.0
        except Exception:
            path = ''
            pixels = 0.0
            brightness = 0.0
            valid = 0.0

        image_paths.append(path)
        pixels_list.append(pixels)
        brightness_list.append(brightness)
        valid_list.append(valid)

    print(f"{split_name} 이미지 준비 완료. 성공 {int(sum(valid_list))}, 실패 {len(valid_list) - int(sum(valid_list))}")

    return (
        np.array(image_paths, dtype=str),
        np.array(pixels_list, dtype=np.float32),
        np.array(brightness_list, dtype=np.float32),
        np.array(valid_list, dtype=np.float32),
    )


def _load_image_tensor(path, image_size):

    """tf.data 안에서 이미지 path를 VGG16 입력 텐서로 변환"""

    def read_image():
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32)
        return preprocess_input(image)

    def zero_image():
        return tf.zeros([image_size, image_size, 3], dtype=tf.float32)

    return tf.cond(tf.strings.length(path) > 0, read_image, zero_image)


def build_dataset(review_length, readability, text_inputs,
                  pixels, brightness, image_paths, image_valid,
                  labels, config, shuffle=False):

    """MFRHP fine 모델 입력용 tf.data.Dataset 생성"""

    image_size = config['model']['vgg16_image_size']
    batch_size = config['training']['batch_size']

    inputs = {
        'ReviewLengthInput': review_length.astype(np.float32).reshape(-1, 1),
        'ReadabilityInput': readability.astype(np.float32).reshape(-1, 1),
        'BERTInputIds': text_inputs['input_ids'],
        'BERTAttentionMask': text_inputs['attention_mask'],
        'PixelInput': pixels.astype(np.float32).reshape(-1, 1),
        'BrightnessInput': brightness.astype(np.float32).reshape(-1, 1),
        'ImagePathInput': image_paths,
        'ImageValidInput': image_valid.astype(np.float32).reshape(-1, 1),
    }

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels.astype(np.float32)))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels), seed=config['training']['random_seed'])

    def preprocess(inputs, label):
        image_path = inputs.pop('ImagePathInput')
        inputs['ImageInput'] = _load_image_tensor(image_path, image_size)
        return inputs, label

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
