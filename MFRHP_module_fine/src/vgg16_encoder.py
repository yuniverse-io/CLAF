# 리뷰 이미지 URL을 VGG16 피처(4096차원)와 이미지 수작업 피처(pixels, brightness)로 변환하는 파일

import glob
import hashlib
import io
import os

import numpy as np
import requests
from PIL import Image

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model


def _build_feature_extractor(config):

    """VGG16 fc1 레이어 출력(4096-dim)을 반환하는 피처 추출기 생성"""

    image_size = config['model']['vgg16_image_size']

    base = VGG16(
        weights='imagenet',
        include_top=True,
        input_shape=(image_size, image_size, 3)
    )
    model = Model(
        inputs=base.input,
        outputs=base.get_layer('fc1').output
    )

    return model


def _cache_path(row_id, url, image_dir):
    """CLAFRHP와 동일한 파일명 규칙 사용"""

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
        return img

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content)).convert('RGB')

    os.makedirs(image_dir, exist_ok=True)
    img.save(save_path, format='JPEG')

    return img


def _path_or_url_to_features(row_id, url, image_dir, image_size):
    """
    이미지 로딩 후 3가지 정보 반환
      - arr       : VGG16용 resize(224,224) 배열
      - pixels    : 원본 이미지의 가로×세로 픽셀 수
      - brightness: 원본 이미지의 평균 밝기 (grayscale 기준)
    """

    img = _load_image_from_cache_or_url(row_id, url, image_dir)

    pixels = float(img.width * img.height)
    brightness = float(np.mean(np.array(img.convert('L'))))

    img = img.resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32)

    return arr, pixels, brightness


def encode_images(row_ids, urls, config):

    """
    row_id + image_url을 이용해
    캐시 이미지가 있으면 재사용, 없으면 다운로드 후 VGG16 피처 추출
    """

    image_size = config['model']['vgg16_image_size']
    batch_size = 32
    image_dir = config['data']['image_cache_dir']
    extractor = _build_feature_extractor(config)

    features = []
    pixels_list = []
    brightness_list = []

    print(f"VGG16 인코딩 시작... (총 {len(urls)}개 이미지)")

    for i in range(0, len(urls), batch_size):
        batch_ids = row_ids[i:i + batch_size]
        batch_urls = urls[i:i + batch_size]
        batch_imgs = []

        for row_id, url in zip(batch_ids, batch_urls):
            try:
                arr, px, br = _path_or_url_to_features(row_id, url, image_dir, image_size)
            except Exception:
                arr = np.zeros((image_size, image_size, 3), dtype=np.float32)
                px = 0.0
                br = 0.0

            batch_imgs.append(arr)
            pixels_list.append(px)
            brightness_list.append(br)

        batch_arr = preprocess_input(np.stack(batch_imgs))
        feat = extractor.predict(batch_arr, verbose=0)
        features.append(feat)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  진행 중: {min(i + batch_size, len(urls))}/{len(urls)}")

    features = np.vstack(features)
    pixels = np.array(pixels_list, dtype=np.float32)
    brightness = np.array(brightness_list, dtype=np.float32)

    print(f"VGG16 인코딩 완료. shape: {features.shape}")

    return features, pixels, brightness


def build_vgg16_embeddings(
    train_ids, train_urls,
    val_ids, val_urls,
    test_ids, test_urls,
    config
):

    """train/val/test 이미지 URL을 VGG16 피처 + 수작업 피처로 변환"""

    print("Train 이미지 인코딩 중...")
    vgg_train, pixels_train, bright_train = encode_images(train_ids, train_urls, config)

    print("Val 이미지 인코딩 중...")
    vgg_val, pixels_val, bright_val = encode_images(val_ids, val_urls, config)

    print("Test 이미지 인코딩 중...")
    vgg_test, pixels_test, bright_test = encode_images(test_ids, test_urls, config)

    return (
        vgg_train, vgg_val, vgg_test,
        pixels_train, pixels_val, pixels_test,
        bright_train, bright_val, bright_test
    )
