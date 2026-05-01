# 데이터를 모델에 맞게 다듬는 곳

import os
import glob
import gzip
import hashlib
import html
import re
import unicodedata

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 제안모델에서 만든 텍스트 정제 정규식임
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
CTRL_RE = re.compile(r"[\u0000-\u001F\u007F]")
WS_RE = re.compile(r"\s+")

class Preprocessor:

    def __init__(self, config):
        self.config = config

    def load_data(self):

        """
        gzip에서 리뷰 데이터를 읽어 반환
        
        HP-BERT 이라서 필요한 text랑 helpful_vote, images만 가져오기

        """

        cfg = self.config
        nrows = cfg['data']['sample_size']

        print(f"데이터 로딩 시작... sample_size={nrows}")

        with gzip.open(cfg['data']['review_path'], 'rb') as f:
            reviews = pd.read_json(f, lines=True, nrows=nrows)

        reviews = reviews[['text', 'helpful_vote', 'images']]
        print(f"로딩된 크기: {reviews.shape}")

        return reviews
    
    def filter_valid_rows(self, df):

        """
        전처리 과정 필요한 행만 남기기

        제안 모델에서 한 방식
        1. helpful_vote > 0
        2. text 있어야함
        3. 이미지 있어야함

        """

        def has_images(x):
            if isinstance(x, list):
                return len(x) > 0
            if isinstance(x, dict):
                return len(x) > 0
            return False

        mask = (
            (df['helpful_vote'] > 0) &
            (df['text'].notna()) &
            (df['text'].astype(str).str.strip().str.len() > 0) &
            (df['images'].apply(has_images))
        )
        df = df[mask].reset_index(drop=True)
        print(f"필터링 후 (helpful_vote>0 와 텍스트/이미지 존재): {len(df)}개")

        return df
    
    def preprocess_text_and_extract_image(self, df):

        """
        제안 모델 처럼 clean_text 생성 + 첫 이미지 URL 추출

        사실 이미지 안 쓰긴 하는데 동일하게 맞추려면 해야함

        """

        def clean_text(text):

            if not text or not str(text).strip():
                return None
            
            text = str(text)
            # HTML entity 복원임
            text = html.unescape(text)
            # 이게 HTML 태그 제거
            text = BeautifulSoup(text, "html.parser").get_text()
            text = URL_RE.sub(" [URL] ", text)
            text = unicodedata.normalize("NFKC", text)
            text = CTRL_RE.sub(" ", text)
            text = WS_RE.sub(" ", text).strip().lower()
            return text if text else None

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
        
        # 제안 모델 처럼 길이가 3보다 짧으면 제거
        df = df.copy()
        df['clean_text'] = df['text'].apply(clean_text)
        df = df[df['clean_text'].notna() & (df['clean_text'].str.len() >= 3)].reset_index(drop=True)
        print(f"텍스트 전처리 후 (유효 텍스트 길이 >= 3): {len(df)}개")
        # 이미지 url 없으면 제거
        df['image_url'] = df['images'].apply(first_image_url)
        df = df[df['image_url'].notna()].reset_index(drop=True)
        print(f"이미지 URL 추출 후: {len(df)}개")

        return df
    
    def add_log_label(self, df):

        """
        helpful_vote에 log(x+1) 변환 적용
        """

        df = df.copy()
        df['log_vote'] = np.log(df['helpful_vote'].astype(np.float32) + 1)

        # 필요한 컬럼만 남겨서 다음 작업 할수 있게 하기!
        df = df[['clean_text', 'image_url', 'log_vote']].copy()

        print(f"\n최종 데이터: {len(df)}개")
        print(f"log_vote 분포:\n{df['log_vote'].describe()}")

        return df

    def map_existing_images(self, df):
        """
        제안 모델에 있는 이미지가 진짜 있는지 확인 하는작업.
        사실 사용하지는 않지만 동일하게 샘플을 가져가는게 중요하니깐
        이 작업도 동일하게 진행
        """

        image_dir = self.config['data']['image_dir']

        os.makedirs(image_dir, exist_ok=True)

        image_paths = []
        failed_indices = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc="기존 이미지 매핑"):
            url = row['image_url']

            # 제안모델과 동일한 파일명 규칙
            # 파일명: "{row_index}_{md5(image_url + row_index)[:6]}.jpg"
            h = hashlib.md5((url + str(i)).encode()).hexdigest()[:6]
            fname = f"{i}_{h}.jpg"
            fpath = os.path.join(image_dir, fname)

            if os.path.exists(fpath):
                image_paths.append(fpath)
            else:
                # 확장자가 jpg가 아닐 수도 있으므로 같은 prefix를 가진 파일을 한 번 더 찾기
                fallback_paths = sorted(glob.glob(os.path.join(image_dir, f"{i}_{h}.*")))

                if fallback_paths:
                    image_paths.append(fallback_paths[0])
                else:
                    # 이미지가 없을때 지울 모음에 추가
                    failed_indices.append(i)

        # 이미지 파일이 없는 row 제거
        df = df.drop(index=failed_indices).reset_index(drop=True)

        # 제안 모델 방식 처럼 그대로 남기기
        df['image_path'] = image_paths

        success = len(image_paths)
        fail = len(failed_indices)

        print(f"\n이미지 매핑 완료: 성공 {success}, 실패 {fail} (실패 행 제거됨)")
        print(f"최종 데이터: {len(df)}개")

        return df

    def split(self, df):

        """train : val : test = 7 : 1 : 2 비율로 분할"""

        cfg = self.config
        test_size = cfg['split']['test_size']
        val_size  = cfg['split']['val_size']

        df_trainval, df_test = train_test_split(
            df, test_size=test_size,
            random_state=cfg['training']['random_seed']
        )

        df_train, df_val = train_test_split(
            df_trainval, test_size=val_size,
            random_state=cfg['training']['random_seed']
        )

        print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

        return df_train, df_val, df_test

    def run(self):

        """전체 전처리 파이프라인 실행"""

        print("==== 데이터 파이프라인 실행 ====")

        print("Step 1/6. 데이터 로딩")
        df = self.load_data()

        print("Step 2/6. 유효 행 필터링")
        df = self.filter_valid_rows(df)

        print("Step 3/6. 텍스트 전처리 + 이미지 URL 추출")
        df = self.preprocess_text_and_extract_image(df)

        print("Step 4/6. log label 생성")
        df = self.add_log_label(df)

        print("Step 5/6. 실제 이미지 파일 존재 여부 확인")
        df = self.map_existing_images(df)

        print("Step 6/6. train/val/test 분할")
        df.reset_index(drop=True, inplace=True)

        print("==== 데이터 파이프라인 종료 ====")

        return self.split(df)
