from keras.layers import Layer, MultiHeadAttention, LayerNormalization, Attention
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Add, \
    Activation, concatenate, GlobalAveragePooling1D, Reshape
from keras.models import Model
from keras.regularizers import l1_l2
from keras import regularizers


def create_transformer_model(input_dim, l2_reg=0.001):
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001))(inputs)
    x = Reshape((8, 32))(x)
    x1 = TransformerLayer(units=64, num_heads=8)(x)
    x2 = TransformerLayer(units=64, num_heads=8)(x1)
    x = x + x1 + x2
    x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(64, activation='relu', kernel_regularizer=l1_l2(l2_reg))(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_cnn_model_pro(input_shape, l1=0.001, l2=0.001):
    inputs = Input(shape=input_shape)
    x = Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l1_l2(l1, l2))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x1 = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l1_l2(l1, l2))(x)
    residual = Conv1D(32, 1, activation='relu', padding='same')(x)
    x2 = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l1_l2(l1, l2))(x1)
    residual = Conv1D(64, 1, activation='relu', padding='same')(residual)
    x = Add()([x2, residual])
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1, l2))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1, l2))(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
