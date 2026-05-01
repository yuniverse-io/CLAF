# 텍스트를 숫자로 바꾸고 Word2Vec으로 각 단어의 의미 벡터를 만드는 파일

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

def tokenize_text(text):
    """
    Word2Vec 학습용 토큰화 함수
    """
    return str(text).split()


def build_tokenizer_and_embeddings(train_review, val_review, test_review, config):
    
    """
    텍스트 -> 숫자 시퀀스 변환 + Word2Vec 임베딩 매트릭스 생성

    """
    
    print("Word2Vec 학습을 위한 토큰화 중...")
    tokenized_train = train_review.apply(tokenize_text)
    
    print("Word2Vec 학습 중...")
    w2v = Word2Vec(
        sentences=tokenized_train,
        vector_size=config['model']['embedding_dim'],
        window=config['w2v']['window'],
        min_count=config['w2v']['min_count'],
        workers=config['w2v']['workers']
    )
    print(f"Word2Vec 학습 완료. 어휘 크기: {len(w2v.wv)}")
    
    # 단어장 만들기
    tokenizer = Tokenizer(
        num_words=config['model']['max_words'],
        oov_token='<OOV>',
        filters='',
        lower=False
    )

    # 훈련용 데이터로 훈련
    tokenizer.fit_on_texts(train_review)

    # 텍스트 -> 숫자 -> 패딩 함수
    def encode(texts):
        """
        만약 짧으면 뒤를 0으로 채움
        만약 길면 max_seq_len 만큼 앞에서 자르기
        """
        return pad_sequences(
            tokenizer.texts_to_sequences(texts),
            maxlen=config['model']['max_seq_len'],
            padding='post',
            truncating='post'
        )
    
    review_train = encode(train_review)
    review_val   = encode(val_review)
    review_test  = encode(test_review)

    # 단어장 꺼내기
    word_index  = tokenizer.word_index
    # 최종 단어 개수
    max_words = config['model']['max_words']
    total_words = min(max_words - 1, len(word_index))

    # 빈 임베딩 행렬 만들기: 행=단어 수, 열=100차원
    embedding_matrix = np.zeros((total_words + 1, config['model']['embedding_dim']))

    # 단어장 한 줄씩 보면서 Word2Vec 벡터 채워넣기
    for word, i in word_index.items():
        # Word2Vec에 없는 단어는 0으로 그냥 진행
        if i > total_words:
            continue
        if word in w2v.wv:
            embedding_matrix[i] = w2v.wv[word]
        
    print(f"임베딩 매트릭스 크기: {embedding_matrix.shape}")
    
    return review_train, review_val, review_test, embedding_matrix, total_words