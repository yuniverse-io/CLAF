# HP-BERT 모델을 만드는 파일

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


def build_model(config):
    """
    HP-BERT 모델 생성 및 컴파일

    흐름:
    BERT CLS 임베딩 (768차원) → Dense(768, relu) → Dense(1, linear) → log_vote 예측값
    """

    print("모델 생성 시작...")

    bert_dim = config['model']['bert_dim']
    learning_rate = config['training']['learning_rate']

    # BERT 임베딩 입력
    text_input = Input(shape=(bert_dim,), name='Bert_Input')

    # 특징 추출
    x = Dense(768, activation='relu', name='Hidden_Layer')(text_input)

    # 최종 예측
    prediction = Dense(1, activation='linear', name='Predict_Layer')(x)

    model = Model(inputs=[text_input], outputs=prediction)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )

    print("모델 생성 완료!")

    return model
