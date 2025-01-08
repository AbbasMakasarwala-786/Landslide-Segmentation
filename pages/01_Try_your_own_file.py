import os
import streamlit as st
from PIL import Image
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Activation,MaxPool2D,Conv2DTranspose,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from tensorflow.keras.models import load_model
import gdown

file_id = '1IgoGnmrtGOFzkgzcarvUjsUnZzyjWp54'

# Local filename to save the model weights
output = "weights.h5"

# Download weights if not already downloaded
if not os.path.exists(output):
    print("Downloading model weights...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

def process_image(uploaded_image):
    image = tf.image.decode_image(uploaded_image.read(), channels=3)
    image = tf.image.resize(image, (128, 128))  # Resize for model input

    image = tf.cast(image, tf.float32) / 255.0

    st.image(image.numpy(), caption="Uploaded Image", width=300)

    model = build_unet((128, 128, 3))  # Build and load weights into the model
    
    image_input = np.expand_dims(image.numpy(), axis=0)

    y_hat = model.predict(image_input)
    y_hat = np.squeeze(y_hat, axis=0)

    y_hat = np.clip(y_hat, 0, 1)
    st.markdown("### Segmented image")
    st.image(y_hat, caption="Predicted Image",width=350)

def build_unet(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder blocks (conv + maxpool)
    s1, p1 = Encoder_block(inputs, 64)
    s2, p2 = Encoder_block(p1, 128)
    s3, p3 = Encoder_block(p2, 256)
    s4, p4 = Encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='u-net')
    model.load_weights('model/weights.h5')  # Load weights

    return model

def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def Encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = tf.keras.layers.MaxPool2D((2, 2))(x)  # Maxpool for downsampling
    return x, p

def decoder_block(inputs, skip, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, 2, strides=2, padding='same')(inputs)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

st.markdown("### Upload Your Landslide Image to be segemented ")


upload_image = st.file_uploader("Upload an image of Landslide")

if upload_image:
    process_image(upload_image)  # Process and display images after upload
