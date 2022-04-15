import math
import time

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras import backend as K
# from utils import plot_confusion_matrix
from sklearn.model_selection import train_test_split

from utils.visualization import plot_learn_model, plot_ae_history


class AE(Model):
    def __init__(self):
        super(AE, self).__init__()


class StackedAE(AE):
    def __init__(self):
        super(StackedAE, self).__init__()


class DenoisingAE(AE):
    def __init__(self):
        super(DenoisingAE, self).__init__()


def create_model_initial(input_dim):
    '''
    Create a model for autoencoder.
    '''
    inp = Input(shape=(input_dim,))
    encoder = Dense(int(math.ceil(0.75 * input_dim)), activation="tanh")(inp)
    encoder = Dense(int(math.ceil(0.5 * input_dim)), activation="tanh")(encoder)
    encoder = Dense(int(math.ceil(0.33 * input_dim)), activation="tanh")(encoder)
    encoder = Dense(int(math.ceil(0.25 * input_dim)), activation="tanh")(encoder)
    decoder = Dense(int(math.ceil(0.33 * input_dim)), activation="tanh")(encoder)
    decoder = Dense(int(math.ceil(0.5 * input_dim)), activation="tanh")(decoder)
    decoder = Dense(int(math.ceil(0.75 * input_dim)), activation="tanh")(decoder)
    decoder = Dense(input_dim)(decoder)
    return Model(inp, decoder), Model(inp, encoder)


def create_model_12(input_dim):
    # number features = 10%
    inp = Input(shape=(input_dim,))
    encoder = Dense(int(math.ceil(0.75 * input_dim)), activation="tanh")(inp)
    encoder = Dense(int(math.ceil(0.5 * input_dim)), activation="tanh")(encoder)
    encoder = Dense(int(math.ceil(0.25 * input_dim)), activation="tanh")(encoder)
    encoder = Dense(int(math.ceil(0.1 * input_dim)), activation="tanh")(encoder)
    decoder = Dense(int(math.ceil(0.25 * input_dim)), activation="tanh")(encoder)
    decoder = Dense(int(math.ceil(0.5 * input_dim)), activation="tanh")(decoder)
    decoder = Dense(int(math.ceil(0.75 * input_dim)), activation="tanh")(decoder)
    decoder = Dense(input_dim)(decoder)
    return Model(inp, decoder), Model(inp, encoder)


def create_model_23(input_dim):
    '''
    Create a model for autoencoder.

    Input_dim: 115
    Bottle_neck dim: 0.33*115 = 38
    Output_dim: 115
    '''

    inp = Input(shape=(input_dim,))
    encoder = Dense(int(math.ceil(0.75 * input_dim)), activation="tanh")(inp)
    encoder = Dense(int(math.ceil(0.5 * input_dim)), activation="tanh")(encoder)
    encoder = Dense(int(math.ceil(0.33 * input_dim)), activation="tanh")(encoder)
    encoder = Dense(int(math.ceil(0.2 * input_dim)), activation="tanh")(encoder)
    decoder = Dense(int(math.ceil(0.33 * input_dim)), activation="tanh")(encoder)
    decoder = Dense(int(math.ceil(0.5 * input_dim)), activation="tanh")(decoder)
    decoder = Dense(int(math.ceil(0.75 * input_dim)), activation="tanh")(decoder)
    decoder = Dense(input_dim)(decoder)
    return Model(inp, decoder), Model(inp, encoder)


def create_model_29(input_dim, act_func='tanh', last_act_func='tanh'):  # 23 features
    inp = Input(shape=(input_dim,))
    encoder = Dense(int(math.ceil(0.75 * input_dim)), activation=act_func)(inp)
    encoder = Dense(int(math.ceil(0.5 * input_dim)), activation=act_func)(encoder)
    encoder = Dense(int(math.ceil(0.33 * input_dim)), activation=act_func)(encoder)
    encoder = Dense(int(math.ceil(0.25 * input_dim)), activation=act_func)(encoder)
    decoder = Dense(int(math.ceil(0.33 * input_dim)), activation=act_func)(encoder)
    decoder = Dense(int(math.ceil(0.5 * input_dim)), activation=act_func)(decoder)
    decoder = Dense(int(math.ceil(0.75 * input_dim)), activation=act_func)(decoder)
    decoder = Dense(input_dim, activation=last_act_func)(decoder)
    return Model(inp, decoder), Model(inp, encoder)


def sampling_model(distribution_params):
    mean, log_var = distribution_params
    epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
    print(epsilon)
    return mean + K.exp(log_var / 2) * epsilon


def create_vae_model_29(input_dim, act_func='tanh', last_act_func='tanh'):  # 23 features
    inp = Input(shape=(input_dim,))
    encoder = Dense(int(math.ceil(0.75 * input_dim)), activation=act_func)(inp)
    encoder = Dense(int(math.ceil(0.5 * input_dim)), activation=act_func)(encoder)
    encoder = Dense(int(math.ceil(0.33 * input_dim)), activation=act_func)(encoder)
    encoder = Dense(int(math.ceil(0.25 * input_dim)), activation=act_func)(encoder)

    decoder = Dense(int(math.ceil(0.33 * input_dim)), activation=act_func)(encoder)
    decoder = Dense(int(math.ceil(0.5 * input_dim)), activation=act_func)(decoder)
    decoder = Dense(int(math.ceil(0.75 * input_dim)), activation=act_func)(decoder)
    decoder = Dense(input_dim, activation=last_act_func)(decoder)
    return Model(inp, decoder), Model(inp, encoder)


def create_model(input_dim, num_features=29, act_func='tanh', last_act_func='tanh'):
    '''
    Create a model for autoencoder.
    '''
    if num_features == 12:
        return create_model_12(input_dim=input_dim)
    elif num_features == 23:
        return create_model_23(input_dim=input_dim)
    elif num_features == 29:
        return create_model_29(input_dim=input_dim, act_func=act_func, last_act_func=last_act_func)
    else:
        return create_model_29(input_dim=input_dim, act_func=act_func, last_act_func=last_act_func)


def ae_process(X, num_features):  # , X_test):
    X_train, X_val = train_test_split(X, test_size=0.2, shuffle=True)
    model, encoder = create_model(X_train.shape[1], num_features=num_features)  # top_n_features)
    model.compile(loss="mean_squared_error", optimizer="adam")
    cp = ModelCheckpoint(filepath=f"dump_models/model.h5",
                         monitor='val_loss',
                         save_best_only=True,
                         verbose=0)
    # es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    start = time.time()
    #     epochs = 100
    epochs = 50
    batch_size = 200
    history = model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, X_val),
                        verbose=1,
                        callbacks=[cp, es])

    end = time.time()
    print('time')
    print(end - start)
    print(model.summary())
    print("Plot Models")
    plot_learn_model(model=model, img_file='autoencoder_model.png')
    plot_learn_model(model=encoder, img_file='encoder_model.png')
    X = encoder.predict(X)
    # X_test = encoder.predict(X_test)
    # plot history
    plot_ae_history(history)
    return X, encoder  # , X_test, encoder


def denoising_ae_process(X, num_features, noise_factor=0.5, epochs=50, batch_size=200):  # , X_test):
    X_train, X_val = train_test_split(X, test_size=0.2, shuffle=True)
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_val_noisy = X_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_val.shape)
    model, encoder = create_model(X_train.shape[1], num_features=num_features)  # top_n_features)
    model.compile(loss="mean_squared_error", optimizer="adam")
    cp = ModelCheckpoint(filepath=f"dump_models/model.h5",
                         monitor='val_loss',
                         save_best_only=True,
                         verbose=0)
    # es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    start = time.time()
    #     epochs = 100
    # epochs = 50
    # batch_size = 200  # 128, 256, 512 need to be tested
    history = model.fit(X_train_noisy, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val_noisy, X_val),
                        verbose=1,
                        callbacks=[cp, es])

    end = time.time()
    print('time')
    print(end - start)
    print(model.summary())
    print("Plot Models")
    plot_learn_model(model=model, img_file='autoencoder_model.png')
    plot_learn_model(model=encoder, img_file='encoder_model.png')
    X = encoder.predict(X)
    # X_test = encoder.predict(X_test)
    # plot history
    plot_ae_history(history)
    return X, encoder  # , X_test, encoder
