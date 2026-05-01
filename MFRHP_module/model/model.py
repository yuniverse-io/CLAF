# 실제 모델을 만드는 곳

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Multiply,
                                     Bidirectional, GRU, Flatten, Dropout,
                                     Reshape, Layer)
from tensorflow.keras.optimizers import Adam


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



def build_model(config):

    """
    MFRHP: Co-Attention based Multi-modal Fusion for Review Helpfulness Prediction

    흐름:
      [Text Branch]
        review_length(1) → Dense(1, relu) ─┐
        readability(1)   → Dense(1, relu) ─┤→ hand_text (2,)
        bert(768) → Reshape(1,768) → BiGRU(128) → Flatten (256,)
        Concat(BiGRU, hand_text) → Dense(64, relu) → text_feature (64,)
        → SelfAttention → Dropout → self_text (embed_dim_v,)

      [Image Branch]
        pixel(1)      → Dense(1, relu) ─┐
        brightness(1) → Dense(1, relu) ─┤→ hand_image (2,)
        vgg16(4096) → Dense(4096) → Dense(2048) → Dense(1024)
        Concat(VGG16, hand_image) → Dense(64, relu) → image_feature (64,)
        → SelfAttention → Dropout → self_image (embed_dim_v,)

      [Co-Attention Fusion]
        CoAttn(text_feature → qk, image_feature → v) → co_text_image
        CoAttn(image_feature → qk, text_feature → v) → co_image_text
        Multiply(co_text_image, co_image_text) → co_feature (embed_dim_v,)

      [Final]
        Concat(self_text, self_image, co_feature) → Dense(64, relu) → Dense(1, linear)
    """

    print("모델 생성 시작...")

    text_input_dim  = config['model']['text_input_dim']    # 768
    image_input_dim = config['model']['image_input_dim']   # 4096
    embed_dim_q_k   = config['model']['embed_dim_q_k']     # 64
    embed_dim_v     = config['model']['embed_dim_v']       # 32
    dropout_rate    = config['training']['dropout_rate']   # 0.8
    lr              = config['training']['learning_rate']  # 0.001

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
                                               readability_dense])           # (batch, 2)

    bert_input  = Input(shape=(text_input_dim,), name='BERTInput')
    bert_expand = Reshape((1, text_input_dim),
                          name='BERT_Reshape')(bert_input)                   # (batch, 1, 768)
    bi_gru      = Bidirectional(GRU(128, return_sequences=False),
                                name='BiGRU')(bert_expand)                   # (batch, 256)
    bi_gru      = Flatten(name='BiGRU_Flatten')(bi_gru)                     # (batch, 256)

    text_feature = Concatenate(name='Text_Feature_Concat')([bi_gru,
                                                            hand_text])      # (batch, 258)
    text_feature = Dense(64, activation='relu',
                         name='Dense_Text_Feature')(text_feature)            # (batch, 64)
    self_text    = self_attn(text_feature)                                   # (batch, embed_dim_v)
    self_text    = Dropout(rate=dropout_rate, name='Dropout_Text')(self_text)

    # 2. Image Branch
    pixel_input  = Input(shape=(1,), name='PixelInput')
    pixel_dense  = Dense(1, activation='relu',
                         name='Dense_Pixel')(pixel_input)

    brightness_input = Input(shape=(1,), name='BrightnessInput')
    brightness_dense = Dense(1, activation='relu',
                             name='Dense_Brightness')(brightness_input)

    hand_image = Concatenate(name='Hand_Image')([pixel_dense,
                                                 brightness_dense])          # (batch, 2)

    vgg16_input = Input(shape=(image_input_dim,), name='VGG16Input')
    vgg16_dense = Dense(image_input_dim, activation='linear',
                        name='Dense_VGG16_0')(vgg16_input)
    vgg16_dense = Dense(2048, activation='relu',
                        name='Dense_VGG16_1')(vgg16_dense)
    vgg16_dense = Dense(1024, activation='relu',
                        name='Dense_VGG16_2')(vgg16_dense)                   # (batch, 1024)

    image_feature = Concatenate(name='Image_Feature_Concat')([vgg16_dense,
                                                              hand_image])   # (batch, 1026)
    image_feature = Dense(64, activation='relu',
                          name='Dense_Image_Feature')(image_feature)         # (batch, 64)
    self_image    = self_attn(image_feature)                                 # (batch, embed_dim_v)
    self_image    = Dropout(rate=dropout_rate, name='Dropout_Image')(self_image)

    # 3. Co-Attention Fusion
    co_text_image = co_attn([text_feature, image_feature])
    co_image_text = co_attn([image_feature, text_feature])
    co_feature    = Multiply(name='Co_Feature')([co_text_image,
                                                 co_image_text])             # (batch, embed_dim_v)

    # 4. Final Prediction
    overall    = Concatenate(name='Overall_Concat')([self_text,
                                                     self_image,
                                                     co_feature])            # (batch, 3*embed_dim_v)
    overall    = Dense(64, activation='relu', name='Dense_Overall')(overall)
    prediction = Dense(1,  activation='linear', name='Prediction')(overall)

    model = Model(
        inputs=[review_length_input, readability_input, bert_input,
                pixel_input, brightness_input, vgg16_input],
        outputs=prediction
    )
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )

    print("모델 생성 완료!")

    return model
