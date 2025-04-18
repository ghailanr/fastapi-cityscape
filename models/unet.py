import numpy as np
from tensorflow import keras
from keras import layers
from PIL import Image
import io

IMG_SIZE = (256, 512)
NUM_CLASSES = 8
WEIGHTS_PATH = "models/unet-cityscape_with_aug.weights.h5"


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    return keras.Model(inputs, outputs)


def load_model():
    model = get_model(IMG_SIZE, NUM_CLASSES)
    model.load_weights(WEIGHTS_PATH)
    return model


def predict_mask(uploaded_file, model):
    image = Image.open(uploaded_file.file).convert("RGB")
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    mask = np.argmax(prediction, axis=-1).astype(np.uint8)

    mask_img = Image.fromarray(mask * (255 // NUM_CLASSES)).convert("L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    result = buf.getvalue()
    buf.close()
    return result
