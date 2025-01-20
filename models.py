# models.py
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, RepeatVector, Reshape, Bidirectional, GRU, Input, Concatenate, SimpleRNN
from tcn import TCN
from tensorflow.keras.callbacks import EarlyStopping

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def build_model(look_back, n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=(look_back, n_features)))
    model.add(TCN(return_sequences=False, kernel_size=2, nb_filters=32))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
    return model

def build_model_MLP(look_back, n_features, h=1):
    model = Sequential()
    # Flatten layer will convert (look_back, n_features) -> (look_back * n_features,)
    model.add(Flatten(input_shape=(look_back, n_features)))
    
    # You can add more hidden layers as needed:
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(64, activation='relu'))  # Example additional layer
    
    # Output layer for 'h' predictions
    model.add(Dense(h))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=rmse)
    return model

def build_model_LSTM(look_back, n_features, h=1):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=False, input_shape=(look_back, n_features)))
    model.add(Dense(h))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=rmse)
    return model

import tensorflow as tf

def build_seq2seq_TCN(look_back, n_features, h=1):
    model = Sequential()
    model.add(
        TCN(
            input_shape=(look_back, n_features),
            return_sequences=True,   # <--- 改成 True
            kernel_size=2,
            nb_filters=32
        )
    )
    # 对 TCN 每个时间步的输出都加上一层 Dense(h)
    model.add(TimeDistributed(Dense(h)))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=rmse)  # 假设您定义了 rmse
    return model

def build_model_TCN(look_back, n_features, h=1):
    model = Sequential()
    model.add(TCN(input_shape=(look_back, n_features), return_sequences=False, kernel_size=2, nb_filters=32))
    model.add(Dense(h))
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=rmsle)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=rmse)
    return model

def build_seq2seq_tcn_lstm(look_back,
                           n_features,
                           h,
                           latent_dim=64,
                           nb_filters=32,
                           kernel_size=2):
    # ------- Encoder 部分 (TCN) -------
    encoder_inputs = Input(shape=(look_back, n_features), name='encoder_input')
    encoder_output = TCN(nb_filters=nb_filters,
                         kernel_size=kernel_size,
                         return_sequences=False,
                         name='tcn_encoder')(encoder_inputs)
    state_h = Dense(latent_dim, activation='linear', name='encoder_state_h')(encoder_output)
    state_c = Dense(latent_dim, activation='linear', name='encoder_state_c')(encoder_output)

    # ------- Decoder 部分 (LSTM) -------
    decoder_inputs = Input(shape=(h, n_features), name='decoder_input')
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='lstm_decoder')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    
    # 調整 Dense 層以輸出單一特徵
    decoder_dense = Dense(1, activation='linear', name='decoder_output_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # ------- 建立模型 -------
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=rmse)
    model.summary()
    return model


from keras.layers import Dense, Dropout, BatchNormalization, TimeDistributed


def build_model_TCN_II(look_back, n_features, h=1):
    model = Sequential()
    model.add(
        TCN(
            input_shape=(look_back, n_features),
            nb_filters=32,
            kernel_size=2,
            dilations=[1, 2, 4],
            return_sequences=True,
            dropout_rate=0.2  # TCN 本身帶有 dropout 參數
        )
    )
    # 在 TCN 輸出之後可再加一層 BN
    model.add(BatchNormalization())
    
    # 第二層 TCN
    model.add(
        TCN(
            nb_filters=64, 
            kernel_size=2, 
            dilations=[1, 2, 4],
            return_sequences=False, 
            dropout_rate=0.2
        )
    )
    # 再加一層 BN
    model.add(BatchNormalization())

    # 全連接層
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))  # 額外的 dropout
    model.add(Dense(h))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error'
    )
    return model

def build_combined_model(look_back, n_features):
    input_tcn = Input(shape=(look_back, n_features))
    tcn_out = TCN(return_sequences=False)(input_tcn)

    input_combined = Input(shape=(look_back, n_features))
    bi_lstm = Bidirectional(LSTM(units=50, return_sequences=True))(input_combined)
    tcn_combined = TCN(return_sequences=False)(bi_lstm)
    combined_output = Concatenate(axis=-1)([tcn_out, tcn_combined])
    output_layer = Dense(units=1)(combined_output)
    final_model = Model(inputs=[input_tcn, input_combined], outputs=output_layer)
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
    return final_model

def build_model_sensors(look_back, n_features, h=1):
    model = Sequential()
    model.add(Conv1D(filters=2, kernel_size=2, activation='relu', input_shape=(look_back, n_features)))
    model.add(GRU(100, activation='relu', return_sequences=True))
    model.add(GRU(100, activation='relu', return_sequences=False))
    model.add(Dense(h))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
    return model

def build_model_cnn(look_back, n_features, h):
    model = Sequential()
    model.add(Conv1D(filters=2, kernel_size=2, activation='relu', input_shape=(look_back, n_features)))
    model.add(Flatten())
    model.add(Dense(h))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
    return model

def build_model_cnn_lstm(look_back, n_features):
    model = Sequential()
    model.add(Conv1D(filters=2, kernel_size=2, activation='relu', input_shape=(look_back, n_features)))
    model.add(LSTM(100, activation='relu', return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
    return model

def build_model_tcn_gru(look_back, n_features, h):
    model = Sequential()
    model.add(TCN(input_shape=(look_back, n_features), return_sequences=True))
    model.add(GRU(100, activation='relu', return_sequences=False))
    model.add(Dense(h))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=rmse)
    return model

def train_model(model, X_train_scaled, Y_train, X_validation_scaled, Y_validation):
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(
        X_train_scaled,
        Y_train, 
        epochs=100,
        batch_size=128, 
        validation_data=(X_validation_scaled, Y_validation),
        callbacks=[early_stopping]  
    )
    return history

def train_model_2(model, X_train_scaled, Y_train, X_validation_scaled, Y_validation):
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(
        [X_train_scaled, X_train_scaled],
        Y_train, 
        epochs=100, 
        batch_size=128, 
        validation_data=([X_validation_scaled, X_validation_scaled], Y_validation),
        callbacks=[early_stopping]  
    )
    return history
