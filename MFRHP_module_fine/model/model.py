# 실제 모델을 만드는 곳
# Frozen end-to-end 버전: BERT/VGG16을 모델 안에 넣되 trainable=False로 묶는다.

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Multiply,
                                     Bidirectional, GRU, Flatten, Dropout,
                                     Reshape, Layer, Lambda)
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel


class SelfAttention(Layer):

    """
    Scaled Dot-Product Self-Attention

    흐름:
      input (batch, dim) → Q, K, V 행렬 생성
      → attn = softmax(Q @ K.T / sqrt(dim_q_k))
      → output = attn @ V  →  (batch, embed_dim_v)
    """

    def __init__(self, embed_dim_q_k, embed_dim_v, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.embed_dim_q_k = embed_dim_q_k
        self.embed_dim_v   = embed_dim_v

    def build(self, input_shape):
        dim      = input_shape[-1]
        self.Wq  = self.add_weight(name="Wq", shape=(dim, self.embed_dim_q_k))
        self.Wk  = self.add_weight(name="Wk", shape=(dim, self.embed_dim_q_k))
        self.Wv  = self.add_weight(name="Wv", shape=(dim, self.embed_dim_v))

    def call(self, inputs):
        q     = tf.matmul(inputs, self.Wq)
        k     = tf.matmul(inputs, self.Wk)
        v     = tf.matmul(inputs, self.Wv)
        scale = tf.math.sqrt(tf.cast(self.embed_dim_q_k, tf.float32))

        attn_weights = tf.nn.softmax(
            tf.matmul(q, k, transpose_b=True) / scale, axis=-1
        )
        return tf.matmul(attn_weights, v)


class CoAttention(Layer):

    """
    Cross-modal Co-Attention

    inputs = [inputs_qk, inputs_v]
      inputs_qk (batch, dim) → Q, K 생성
      inputs_v  (batch, dim) → V 생성
      → attn = softmax(Q @ K.T / sqrt(dim_q_k))
      → output = attn @ V
    """

    def __init__(self, embed_dim_q_k, embed_dim_v, **kwargs):
        super(CoAttention, self).__init__(**kwargs)
        self.embed_dim_q_k = embed_dim_q_k
        self.embed_dim_v = embed_dim_v

    def build(self, input_shape):
        qk_shape, v_shape = input_shape
        qk_dim = qk_shape[-1]
        v_dim = v_shape[-1]

        self.Wq = self.add_weight(name="Wq", shape=(qk_dim, self.embed_dim_q_k))
        self.Wk = self.add_weight(name="Wk", shape=(qk_dim, self.embed_dim_q_k))
        self.Wv = self.add_weight(name="Wv", shape=(v_dim, self.embed_dim_v))

        super().build(input_shape)

    def call(self, inputs):
        inputs_qk, inputs_v = inputs

        q = tf.matmul(inputs_qk, self.Wq)
        k = tf.matmul(inputs_qk, self.Wk)
        v = tf.matmul(inputs_v, self.Wv)
        scale = tf.math.sqrt(tf.cast(self.embed_dim_q_k, tf.float32))

        attn_weights = tf.nn.softmax(
            tf.matmul(q, k, transpose_b=True) / scale,
            axis=-1
        )
        return tf.matmul(attn_weights, v)


class FrozenBertCLS(Layer):

    """BERT를 Keras Layer로 감싸 CLS 벡터만 반환한다."""

    def __init__(self, model_name, trainable_backbone=False, **kwargs):
        super(FrozenBertCLS, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained(model_name, name='Frozen_BERT')
        self.bert.trainable = bool(trainable_backbone)

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training,
        )
        return outputs.last_hidden_state[:, 0, :]

    @property
    def backbone_trainable(self):
        return self.bert.trainable


def _build_frozen_vgg16(config):

    """VGG16 fc1 출력을 반환하는 frozen feature extractor"""

    image_size = config['model']['vgg16_image_size']
    vgg = VGG16(
        weights='imagenet',
        include_top=True,
        input_shape=(image_size, image_size, 3)
    )
    extractor = Model(
        inputs=vgg.input,
        outputs=vgg.get_layer('fc1').output,
        name='Frozen_VGG16_fc1'
    )
    extractor.trainable = bool(config['model'].get('vgg16_trainable', False))

    return extractor


def build_model(config):

    """
    MFRHP Frozen End-to-End

    흐름:
      [Text Branch]
        input_ids / attention_mask → Frozen BERT → CLS(768)
        CLS → Reshape(1,768) → BiGRU(128) → Flatten (256,)
        review_length/readability와 결합 → text_feature

      [Image Branch]
        image tensor → Frozen VGG16(fc1, 4096)
        pixel/brightness와 결합 → image_feature

      [Fusion]
        self-attention + co-attention → Dense → helpfulness 예측
    """

    print("모델 생성 시작... Frozen end-to-end MFRHP")

    text_input_dim  = config['model']['text_input_dim']
    image_input_dim = config['model']['image_input_dim']
    bert_max_len    = config['model']['bert_max_len']
    image_size      = config['model']['vgg16_image_size']
    embed_dim_q_k   = config['model']['embed_dim_q_k']
    embed_dim_v     = config['model']['embed_dim_v']
    dropout_rate    = config['training']['dropout_rate']
    lr              = config['training']['learning_rate']

    # Frozen backbone들. 모델 안에서 forward는 하지만 기본값은 학습하지 않는다.
    bert_backbone = FrozenBertCLS(
        config['model']['bert_model_name'],
        trainable_backbone=config['model'].get('bert_trainable', False),
        name='Frozen_BERT_CLS'
    )
    vgg16_backbone = _build_frozen_vgg16(config)

    # Attention 레이어 (공유 가중치: text/image 동일 레이어 사용)
    self_attn = SelfAttention(embed_dim_q_k=embed_dim_q_k, embed_dim_v=embed_dim_v,
                              name='SelfAttention')
    co_attn   = CoAttention(embed_dim_q_k=embed_dim_q_k,   embed_dim_v=embed_dim_v,
                            name='CoAttention')

    # 1. Text Branch
    review_length_input = Input(shape=(1,), name='ReviewLengthInput')
    review_length_dense = Dense(1, activation='relu',
                                name='Dense_Review_Length')(review_length_input)

    readability_input = Input(shape=(1,), name='ReadabilityInput')
    readability_dense = Dense(1, activation='relu',
                              name='Dense_Readability')(readability_input)

    hand_text = Concatenate(name='Hand_Text')([review_length_dense,
                                               readability_dense])

    input_ids = Input(shape=(bert_max_len,), dtype=tf.int32, name='BERTInputIds')
    attention_mask = Input(shape=(bert_max_len,), dtype=tf.int32, name='BERTAttentionMask')

    bert_cls = bert_backbone([input_ids, attention_mask])

    bert_expand = Reshape((1, text_input_dim),
                          name='BERT_Reshape')(bert_cls)
    bi_gru      = Bidirectional(GRU(128, return_sequences=False),
                                name='BiGRU')(bert_expand)
    bi_gru      = Flatten(name='BiGRU_Flatten')(bi_gru)

    text_feature = Concatenate(name='Text_Feature_Concat')([bi_gru,
                                                            hand_text])
    text_feature = Dense(64, activation='relu',
                         name='Dense_Text_Feature')(text_feature)
    self_text    = self_attn(text_feature)
    self_text    = Dropout(rate=dropout_rate, name='Dropout_Text')(self_text)

    # 2. Image Branch
    pixel_input  = Input(shape=(1,), name='PixelInput')
    pixel_dense  = Dense(1, activation='relu',
                         name='Dense_Pixel')(pixel_input)

    brightness_input = Input(shape=(1,), name='BrightnessInput')
    brightness_dense = Dense(1, activation='relu',
                             name='Dense_Brightness')(brightness_input)

    hand_image = Concatenate(name='Hand_Image')([pixel_dense,
                                                 brightness_dense])

    image_input = Input(shape=(image_size, image_size, 3), name='ImageInput')
    image_valid_input = Input(shape=(1,), name='ImageValidInput')

    vgg16_feature = vgg16_backbone(image_input)
    vgg16_feature = Multiply(name='VGG16_Valid_Mask')([vgg16_feature, image_valid_input])

    vgg16_dense = Dense(image_input_dim, activation='linear',
                        name='Dense_VGG16_0')(vgg16_feature)
    vgg16_dense = Dense(2048, activation='relu',
                        name='Dense_VGG16_1')(vgg16_dense)
    vgg16_dense = Dense(1024, activation='relu',
                        name='Dense_VGG16_2')(vgg16_dense)

    image_feature = Concatenate(name='Image_Feature_Concat')([vgg16_dense,
                                                              hand_image])
    image_feature = Dense(64, activation='relu',
                          name='Dense_Image_Feature')(image_feature)
    self_image    = self_attn(image_feature)
    self_image    = Dropout(rate=dropout_rate, name='Dropout_Image')(self_image)

    # 3. Co-Attention Fusion
    co_text_image = co_attn([text_feature, image_feature])
    co_image_text = co_attn([image_feature, text_feature])
    co_feature    = Multiply(name='Co_Feature')([co_text_image,
                                                 co_image_text])

    # 4. Final Prediction
    overall    = Concatenate(name='Overall_Concat')([self_text,
                                                     self_image,
                                                     co_feature])
    overall    = Dense(64, activation='relu', name='Dense_Overall')(overall)
    prediction = Dense(1,  activation='linear', name='Prediction')(overall)

    model = Model(
        inputs=[review_length_input, readability_input,
                input_ids, attention_mask,
                pixel_input, brightness_input,
                image_input, image_valid_input],
        outputs=prediction
    )
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )

    print("모델 생성 완료!")
    print(f"BERT trainable: {bert_backbone.backbone_trainable}")
    print(f"VGG16 trainable: {vgg16_backbone.trainable}")

    return model
