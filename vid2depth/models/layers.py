from tensorflow.keras import layers, regularizers


def conv(x, features, kz, stride=2, activation='relu', norm=True, l2=None):
    x = layers.Conv2D(features, kz, stride, padding='same', activation=activation, kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
    if norm:
        x = layers.BatchNormalization()(x)
    return x


def deconv(x, features, kz, stride=2, activation='relu', norm=True, l2=None):
    x = layers.Conv2DTranspose(features, kz, stride, padding='same', activation=activation, kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
    if norm:
        x = layers.BatchNormalization()(x)
    return x
