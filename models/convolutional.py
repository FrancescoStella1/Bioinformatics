import keras_tuner as kt
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling1D, BatchNormalization, Conv1D, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, Input, PReLU

import config as cfg


def _compile_cnn_model(model: Model) -> Model:
    model.compile(optimizer=cfg.models["optimizer"],
                  loss=cfg.models["loss"],
                  metrics=cfg.models["metrics"])
    return model


def build_model(hp: kt.HyperParameters) -> Model:
    input_layer = Input(shape=(cfg.data_retrieval["window_size"], 4), name="sequence_data")
    hidden = Conv1D(hp.Choice(f"conv_1", [16, 32, 64, 128]),
                    kernel_size=hp.Int(f"kernel_size_1", min_value=2, max_value=8, step=2),
                    name="cnn_conv_1")(input_layer)
    hidden = PReLU(name="cnn_prelu_1")(hidden)
    hidden = BatchNormalization(name="cnn_bn_1")(hidden)
    hidden = Dropout(.8, name="cnn_dp_1")(hidden)

    for i in range(2, 5):
        if hp.Boolean(f"cnn_conv_layer_{i}"):
            hidden = Conv1D(hp.Choice(f"conv_{i}", [16, 32, 64, 128]),
                            kernel_size=hp.Int(f"kernel_size_{i}", min_value=2, max_value=8, step=2),
                            name=f"cnn_conv_{i}")(hidden)
            hidden = PReLU(name=f"cnn_prelu_{i}")(hidden)

            hidden = AveragePooling1D(pool_size=hp.Int(f"pool_size_{i}", min_value=2, max_value=6, step=2),
                                      strides=hp.Int(f"stride_{i}", min_value=1, max_value=3, step=1),
                                      padding=hp.Choice(f"padding_{i}", ["valid", "same"]), name=f"cnn_avg_pool_{i}")(hidden)
            hidden = BatchNormalization(name=f"cnn_bn_{i}")(hidden)

            hidden = Dropout(hp.Choice(f"cnn_dp_{i}", [.1, .2, .3, .4, .5]), name=f"cnn_dp_{i}")(hidden)

    hidden = GlobalAveragePooling1D(name="cnn_global_pool")(hidden)
    
    hidden = Dense(hp.Choice("cnn_dense", [32, 64, 128]), name="cnn_dense")(hidden)
    hidden = PReLU(name="cnn_last_prelu")(hidden)
    if hp.Boolean("batch_norm_final"):
        hidden = BatchNormalization(name="cnn_bn_final")(hidden)

    hidden = Dropout(.2, name="cnn_dp_final")(hidden)

    output_layer = Dense(1, activation="linear", name="cnn_output_layer")(hidden)

    cnn = Model(inputs=input_layer, outputs=output_layer, name="CNN_regression_model")
    cnn = _compile_cnn_model(cnn)
    return cnn
