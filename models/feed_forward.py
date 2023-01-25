import keras_tuner as kt
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, PReLU

import config as cfg


def _compile_ffnn_model(model: Model) -> Model:
    """
    Compiles the FFNN model.

    :param model: The initialized FFNN model.
    :return: Returns the compiled FFNN model.
    """
    model.compile(optimizer=cfg.models["optimizer"],
                  loss=cfg.models["loss"],
                  metrics=cfg.models["metrics"])
    return model


def build_model(hp: kt.HyperParameters) -> Model:
    input_shape = hp.Fixed("input_shape", hp.get("input_shape"))
    input_layer = Input(shape=(hp.get("input_shape"),), name="epigenomic_data")
    hidden = Dense(hp.Int("ffnn_dense_1", min_value=16, max_value=128, step=16), name="ffnn_dense_1")(input_layer)
    hidden = PReLU(name="ffnn_prelu_1")(hidden)
    hidden = Dense(hp.Int("ffnn_dense_2", min_value=16, max_value=64, step=16), name="ffnn_dense_2")(hidden)
    hidden = PReLU(name="ffnn_prelu_2")(hidden)
    if hp.Boolean("ffnn_first_bn"):
        hidden = BatchNormalization(name="ffnn_bn_1")(hidden)
        hidden = Dropout(hp.Float("ffnn_dp_1", min_value=.0, max_value=.3, step=.1), name="ffnn_dp_1")(hidden)
    else:
        hidden = Dropout(hp.Float("ffnn_dp_1", min_value=.0, max_value=.3, step=.1), name="ffnn_dp_1")(hidden)
    hidden = Dense(hp.Int("ffnn_dense_3", min_value=16, max_value=64, step=16), name="ffnn_dense_3")(hidden)
    hidden = PReLU(name="ffnn_prelu_3")(hidden)
    if hp.Boolean("ffnn_second_bn"):
        hidden = BatchNormalization(name="ffnn_bn_2")(hidden)
        hidden = Dropout(hp.Float("ffnn_dp_2", min_value=.0, max_value=.3, step=.1), name="ffnn_dp_2")(hidden)
    else:
        hidden = Dropout(hp.Float("ffnn_dp_2", min_value=.0, max_value=.3, step=.1), name="ffnn_dp_2")(hidden)
    
    hidden = Dense(hp.Int("ffnn_dense_4", min_value=16, max_value=64, step=16), name="ffnn_dense_4")(hidden)
    hidden = PReLU(name="ffnn_prelu_4")(hidden)

    if hp.Boolean("ffnn_third_bn"):
        hidden = BatchNormalization(name="ffnn_bn_3")(hidden)
        hidden = Dropout(hp.Float("ffnn_dp_3", min_value=.0, max_value=.3, step=.1), name="ffnn_dp_3")(hidden)
    else:
        hidden = Dropout(hp.Float("ffnn_dp_3", min_value=.0, max_value=.3, step=.1), name="ffnn_dp_3")(hidden)

    output_layer = Dense(1, activation="linear", name="ffnn_output_layer")(hidden)

    model = Model(inputs=input_layer, outputs=output_layer, name="FFNN_regression_model")
    model = _compile_ffnn_model(model)
    return model
