# 리뷰 텍스트를 BERT 임베딩(768차원 벡터)으로 변환하는 파일
# HP-BERT에서 clean_text를 BERT CLS 임베딩으로 변환한다.

import numpy as np
from transformers import BertTokenizer, TFBertModel


def encode_texts(texts, tokenizer, bert_model, config, split_name="Data"):
    """
    텍스트 리스트를 BERT CLS 토큰 임베딩으로 변환한다.

    입력:
    - texts: clean_text 리스트
    - tokenizer: 한 번 로딩한 BERT tokenizer
    - bert_model: 한 번 로딩한 BERT model
    - config: 설정 파일
    - split_name: Train / Val / Test 로그 출력용 이름

    반환:
    - (N, 768) 형태의 numpy 배열

    """

    print(f"{split_name} BERT 인코딩 시작... (총 {len(texts)}개 문장)")

    embeddings = []

    # BERT 인코딩용 배치 크기.
    # 학습 batch_size와는 다르다.
    # GPU/CPU 메모리 상황에 따라 config로 빼도 되지만, 우선 기존 코드와 같이 32로 둔다.
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding='max_length',
            truncation=True,
            max_length=config['model']['bert_max_len'],
            return_tensors='tf'
        )

        outputs = bert_model(**encoded)

        # [CLS] 토큰 벡터만 사용한다.
        # last_hidden_state shape: (batch, seq_len, 768)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        embeddings.append(cls_embeddings)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  {split_name} 진행 중: {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(embeddings)

    print(f"{split_name} BERT 인코딩 완료. 임베딩 shape: {embeddings.shape}")

    return embeddings


def build_bert_embeddings(train_review, val_review, test_review, config):
    """
    train/val/test 리뷰 텍스트를 각각 BERT CLS 임베딩으로 변환한다.
    """

    print(f"BERT 모델 로딩 중: {config['model']['bert_model_name']}")

    tokenizer = BertTokenizer.from_pretrained(
        config['model']['bert_model_name']
    )

    bert_model = TFBertModel.from_pretrained(
        config['model']['bert_model_name']
    )

    print("Train 리뷰 인코딩 중...")
    x_train = encode_texts(
        train_review,
        tokenizer,
        bert_model,
        config,
        split_name="Train"
    )

    print("Val 리뷰 인코딩 중...")
    x_val = encode_texts(
        val_review,
        tokenizer,
        bert_model,
        config,
        split_name="Val"
    )

    print("Test 리뷰 인코딩 중...")
    x_test = encode_texts(
        test_review,
        tokenizer,
        bert_model,
        config,
        split_name="Test"
    )

    return x_train, x_val, x_test
