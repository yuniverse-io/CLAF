# 리뷰 텍스트를 BERT 임베딩(768차원 벡터)으로 변환하는 파일

import numpy as np
from transformers import BertTokenizer, TFBertModel


def _encode(texts, tokenizer, bert_model, config):

    """텍스트 리스트를 BERT CLS 토큰 임베딩으로 변환 → (N, 768) 반환"""

    print(f"BERT 인코딩 시작... (총 {len(texts)}개 문장)")
    embeddings = []

    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config['model']['bert_max_len'],
            return_tensors='tf'
        )

        # last_hidden_state: (batch, seq_len, 768)
        outputs = bert_model(**encoded)

        # [CLS] 토큰 (index 0) 벡터만 추출 → (batch, 768)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embeddings)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  진행 중: {i + batch_size}/{len(texts)}")

    embeddings = np.vstack(embeddings)
    print(f"인코딩 완료. shape: {embeddings.shape}")

    return embeddings


def build_bert_embeddings(train_review, val_review, test_review, config):

    """BERT를 한 번만 로드해 train/val/test 모두 인코딩"""

    print(f"BERT 모델 로딩 중: {config['model']['bert_model_name']}")
    tokenizer  = BertTokenizer.from_pretrained(config['model']['bert_model_name'])
    bert_model = TFBertModel.from_pretrained(config['model']['bert_model_name'])

    print("Train 리뷰 인코딩 중...")
    x_train = _encode(train_review, tokenizer, bert_model, config)

    print("Val 리뷰 인코딩 중...")
    x_val   = _encode(val_review,   tokenizer, bert_model, config)

    print("Test 리뷰 인코딩 중...")
    x_test  = _encode(test_review,  tokenizer, bert_model, config)

    return x_train, x_val, x_test
