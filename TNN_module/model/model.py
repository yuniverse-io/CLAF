# TNN 모델을 만드는 파일

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dense
)
from tensorflow.keras.optimizers import Adam


def build_model(total_words, embedding_matrix, config):
    """
    TNN 모델 생성 및 컴파일

    흐름:
    리뷰 → 임베딩 → Conv1D(kernel=1) → MaxPooling → ┐
                  → Conv1D(kernel=2) → MaxPooling → Concatenate → Dense → 예측값
                  → Conv1D(kernel=3) → MaxPooling → ┘
    """

    print("모델 생성 시작...")

    # 값들 저장 해두기
    vocab_size    = total_words + 1
    max_seq_len   = config['model']['max_seq_len']
    embedding_dim = config['model']['embedding_dim']
    conv_filters  = config['model']['conv_filters']
    kernel_sizes  = config['model']['conv_kernel_sizes']
    dense_units   = config['model']['dense_units']
    learning_rate = config['training']['learning_rate']

    # 입력
    text_input = Input(shape=(max_seq_len,), name="Input_Text")

    # 임베딩 (Word2Vec 훈련은 묶기)
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        name="Embedding_Text"
    )(text_input)

    # 병렬 Conv1D
    pools = []
    for k in kernel_sizes:
        conv = Conv1D(filters=conv_filters, kernel_size=k,
                      padding='same', activation='relu')(embedding)
        pool = GlobalMaxPooling1D()(conv)   # 각 채널에서 가장 강한 신호 하나만 추출
        pools.append(pool)

    # 모든 채널 합치기
    concat = Concatenate()(pools)
    hidden = Dense(dense_units, activation='relu')(concat)
    output = Dense(1, activation='linear')(hidden)

    model = Model(inputs=text_input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )

    print("모델 생성 완료!")

    return model