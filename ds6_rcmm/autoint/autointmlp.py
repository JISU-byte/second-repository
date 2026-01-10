import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Flatten, Embedding, BatchNormalization
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import backend as K
import numpy as np
import math

# 임베딩 레이어
class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embedding_size, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.total_dim = sum(field_dims)
        self.embedding_size = embedding_size
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        self.embedding = tf.keras.layers.Embedding(input_dim=self.total_dim, output_dim=self.embedding_size)

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.embedding.set_weights([tf.keras.initializers.GlorotUniform()(shape=self.embedding.weights[0].shape)])

    def call(self, x):
        x = tf.cast(x, tf.int32)
        offset = tf.constant(self.offsets, dtype=tf.int32)
        x = x + offset
        return self.embedding(x)

# 다층 퍼셉트론
class MultiLayerPerceptron(Layer):
    def __init__(self, input_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, init_std=0.0001, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        hidden_units = [input_dim] + list(hidden_units)
        if output_layer:
            hidden_units += [1]
        self.linears = [Dense(units, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=init_std),
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)) for units in hidden_units[1:]]
        self.activation = tf.keras.layers.Activation(activation)
        if self.use_bn:
            self.bn = [BatchNormalization() for _ in hidden_units[1:]]
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.use_bn:
                x = self.bn[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        return x

# 멀티 헤드 어텐션
class MultiHeadSelfAttention(Layer):
    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_res:
            self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                         dtype=tf.float32,
                                         initializer=TruncatedNormal(seed=self.seed))

        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores =  tf.nn.softmax(inner_product)

        result = tf.matmul(self.normalized_att_scores, values)
        result = tf.concat(tf.split(result, self.head_num, ), axis=-1)
        result = tf.squeeze(result, axis=0)

        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        result = tf.nn.relu(result)

        return result

    def compute_output_shape(self, input_shape):

        return (None, input_shape[1], self.att_embedding_size * self.head_num)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num, 'use_res': self.use_res,'seed': self.seed}
        base_config = super(MultiHeadSelfAttention, self).get_config()
        base_config.update(config)
        return base_config

# AutoIntMLP 레이어
class AutoIntMLP(Layer):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0.4, init_std=0.0001):
        super(AutoIntMLP, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size
        self.embed_output_dim = self.num_fields * self.embedding_size

        self.final_layer = Dense(1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=init_std))

        self.dnn = MultiLayerPerceptron(
            input_dim=self.embed_output_dim,
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
            init_std=init_std,
            output_layer=True
        )

        self.int_layers = [
            MultiHeadSelfAttention(
                att_embedding_size=embedding_size,
                head_num=att_head_num,
                use_res=att_res
            )
            for _ in range(att_layer_num)
        ]

        self.flatten = Flatten()

    def call(self, inputs, training=False):
        embed_x = self.embedding(inputs)
        dnn_embed = self.flatten(embed_x)

        att_input = embed_x
        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = self.flatten(att_input)
        att_output = self.final_layer(att_output)

        dnn_output = self.dnn(dnn_embed, training=training)

        y_pred = tf.nn.sigmoid(tf.nn.relu(att_output) + dnn_output)

        return y_pred

# AutoIntMLP 모델 래퍼
class AutoIntMLPModel(tf.keras.Model):
    def __init__(self, field_dims, embedding_size, dnn_hidden_units=(32, 32),
                 dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False,
                 dnn_dropout=0.4, att_layer_num=3, att_head_num=2, att_res=True, init_std=0.0001):
        super(AutoIntMLPModel, self).__init__()
        self.autoIntMLP_layer = AutoIntMLP(
            field_dims, embedding_size,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            l2_reg_dnn=l2_reg_dnn,
            dnn_use_bn=dnn_use_bn,
            dnn_dropout=dnn_dropout,
            att_layer_num=att_layer_num,
            att_head_num=att_head_num,
            att_res=att_res,
            init_std=init_std
        )

    def call(self, inputs, training=False):
        return self.autoIntMLP_layer(inputs, training=training)
