# 데이터를 모델에 맞게 다듬는 곳

import gzip
import html
import re
import unicodedata

import numpy as np
import pandas as pd
import textstat
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
CTRL_RE = re.compile(r"[\u0000-\u001F\u007F]")
WS_RE = re.compile(r"\s+")

class Preprocessor:

    def __init__(self, config):
        self.config = config

    def load_data(self):

        """gzip에서 리뷰 데이터를 읽어 반환 (텍스트 + 이미지 URL 포함)"""

        cfg   = self.config
        nrows = cfg['data']['sample_size']

        print(f"데이터 로딩 시작... sample_size={nrows}")

        with gzip.open(cfg['data']['review_path'], 'rb') as f:
            reviews = pd.read_json(f, lines=True, nrows=nrows)

            reviews = reviews[['text', 'helpful_vote', 'images']]
            print(f"로딩된 크기: {reviews.shape}")

            return reviews

    def filter_valid_rows(self, df):

        """CLAFRHP와 같은 기준으로 유효한 행만 남기기"""

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
        print(f"필터링 후 (helpful_vote>0 & 텍스트/이미지 존재): {len(df)}개")

        return df
    
    def preprocess_text_and_extract_image(self, df):

        """CLAFRHP와 같은 방식으로 clean_text 생성 + 첫 이미지 URL 추출"""

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
        
        df = df.copy()
        df['clean_text'] = df['text'].apply(clean_text)
        df = df[df['clean_text'].notna() & (df['clean_text'].str.len() >= 3)].reset_index(drop=True)
        print(f"텍스트 전처리 후 (유효 텍스트 길이 >= 3): {len(df)}개")

        df['image_url'] = df['images'].apply(first_image_url)
        df = df[df['image_url'].notna()].reset_index(drop=True)
        print(f"이미지 URL 추출 후: {len(df)}개")

        return df

    def add_log_label(self, df):

        """helpful_vote에 log 변환 적용"""

        df = df.copy()
        df['log_vote'] = np.log(df['helpful_vote'].astype(np.float32) + 1)

        print(f"\n최종 데이터: {len(df)}개")
        print(f"log_vote 분포:\n{df['log_vote'].describe()}")

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

    def compute_handcrafted_features(self, df):

        """텍스트 기반 수작업 피처 생성: review_length, gunning_fog_index"""

        print("수작업 피처 생성 중... (gunning_fog_index 계산 중, 시간이 걸릴 수 있음)")
        df = df.copy()
        df['review_length'] = df['clean_text'].apply(lambda t: len(t.split()))
        df['gunning_fog_index'] = df['clean_text'].apply(textstat.gunning_fog)
        print("수작업 피처 생성 완료!")

        return df


    def run(self):

        """전체 전처리 파이프라인 실행"""

        print("==== 데이터 파이프라인 실행 ====")

        print("Step 1/6. 데이터 로딩")
        df = self.load_data()

        print("Step 2/6. 유효 행 필터링")
        df = self.filter_valid_rows(df)

        print("Step 3/6. 텍스트 전처리 + 이미지 URL 추출")
        df = self.preprocess_text_and_extract_image(df)

        print("Step 4/6. 수작업 피처 생성")
        df = self.compute_handcrafted_features(df)

        print("Step 5/6. log 라벨 생성")
        df = self.add_log_label(df)

        print("Step 6/6. train/val/test 분할")
        df.reset_index(drop=True, inplace=True)

        print("==== 데이터 파이프라인 종료 ====")

        return self.split(df)