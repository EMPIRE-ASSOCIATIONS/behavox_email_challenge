import numpy as np
from keras.layers import Dense, Input, LSTM, Concatenate
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import regularizers

def create_model(
        sentence_padding=512,
        word_embedding_size=300,
        lstm1_output_size=200,
        lstm2_output_size=300,
        regularizer=0.01,
        read_again=True):

    input_ = Input(shape=(sentence_padding, word_embedding_size), dtype='float32')
    regularizer = regularizers.l2(l=regularizer)
    x = Bidirectional(LSTM(lstm1_output_size, return_sequences=read_again, trainable=True,
                           kernel_regularizer=regularizer,
                           bias_regularizer=regularizer),
                      trainable=True,
                      merge_mode='concat')(input_)
    if read_again:
        x = Concatenate()([x, input_])
        x = Bidirectional(LSTM(lstm2_output_size, return_sequences=False, trainable=True,
                               kernel_regularizer=regularizer,
                               bias_regularizer=regularizer),
                          trainable=True,
                          merge_mode='concat')(x)
    x = Dense(1, activation='sigmoid', trainable=True,
              kernel_regularizer=regularizer,
              bias_regularizer=regularizer)(x)

    return Model(inputs=input_, outputs=x)


if __name__ == '__main__':
    model = create_model(sentence_padding=30,
                         word_embedding_size=300,
                         trainable=True,
                         regularizer=0.01,
                         read_again=True)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    print(model.predict(np.array([[[1, 1, 1, 1, 1],
                                   [3, 4, 5, 6, 7]]])))
