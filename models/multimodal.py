from typing import Optional

from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Concatenate, Dense, Dropout, Layer, PReLU

import config as cfg


def _compile_mmnn_model(model: Model) -> Model:
    model.compile(optimizer=cfg.models["optimizer"],
                  loss=cfg.models["loss"],
                  metrics=cfg.models["metrics"])
    return model


def initialize_mmnn_model(input_shape: Optional[int] = None, window_size: Optional[int] = None,
                          input_epigenomic_data: Optional[Layer] = None,
                          input_sequence_data: Optional[Layer] = None,
                          last_hidden_ffnn: Optional[Layer] = None,
                          last_hidden_cnn: Optional[Layer] = None) -> Model:
    if input_shape is None and (input_epigenomic_data is None or last_hidden_ffnn is None):
        raise ValueError("Either the input shape or the features selection layer and the input epigenomic "
                         "layer must be provided.")
    if window_size is None and (input_sequence_data is None or last_hidden_cnn is None):
        raise ValueError("Either the input shape or the features selection layer and the input sequence "
                         "layer must be provided.")
    """
    if input_shape is not None:
        _, input_epigenomic_data, last_hidden_ffnn = ffnn.initialize_ffnn_model(input_shape)
    if window_size is not None:
        _, input_sequence_data, last_hidden_cnn = cnn.initialize_cnn_model(window_size)
    """
    hidden = Concatenate(name="mmnn_concat")([last_hidden_ffnn.input, last_hidden_cnn.input])
    hidden = Dense(64, name="mmnn_dense_1")(hidden)
    hidden = PReLU(name="mmnn_prelu_1")(hidden)
    hidden = BatchNormalization(name="mmnn_bn_1")(hidden)
    hidden = Dropout(.4, name="mmnn_dp_1")(hidden)
    hidden = Dense(16, name="mmnn_dense_2")(hidden)
    hidden = PReLU(name="mmnn_prelu_2")(hidden)
    hidden = BatchNormalization(name="mmnn_bn_2")(hidden)
    hidden = Dropout(.2, name="mmnn_dp_2")(hidden)

    output_layer = Dense(1, activation="linear", name="mmnn_output_layer")(hidden)

    model = Model(inputs=[input_epigenomic_data.input, input_sequence_data.input], outputs=output_layer,
                  name="MMNN_boosted_regression_model" if input_shape is None else "MMNN_regression_model")
    model = _compile_mmnn_model(model)
    return model
