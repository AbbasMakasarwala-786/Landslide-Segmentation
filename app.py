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

image_folder='image_folder'
image_files=['image_100.png','image_100.png','image_1059.png','image_1064.png']
st.markdown("# Landslide Image Segemntation using U-Net  ")

file_id = '1IgoGnmrtGOFzkgzcarvUjsUnZzyjWp54'

# Local filename to save the model weights
output = "weights.h5"

# Download weights if not already downloaded
if not os.path.exists(output):
    print("Downloading model weights...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)


selected_image=st.selectbox("Select an image from this options",image_files)


if selected_image:

    image_path=os.path.join(image_folder,selected_image)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.markdown(
        """
        <style>
        .centered-image {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 10vh;
            left:70%;
            display:grid;
            place-item:center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="centered-image">', unsafe_allow_html=True)

    st.image(image_rgb,caption="Input image selected",width=300)

    def conv_block(inputs,num_filters):
        x=Conv2D(num_filters,3,padding='same')(inputs)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)

        x=Conv2D(num_filters,3,padding="same")(x)
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        return x

    def Encoder_block(inputs,num_filters):
        x=conv_block(inputs,num_filters)# for the skip connection
        p=MaxPool2D((2,2))(x)# output for the Encoder block
        return x,p


    def decoder_block(inputs,skip,num_filters):
        x=Conv2DTranspose(num_filters,2,strides=2,padding='same')(inputs)
        x=Concatenate()([x,skip])
        x=conv_block(x,num_filters)
        return x
        

    def build_unet(input_shape):
        inputs=Input(shape=input_shape)
        s1,p1=Encoder_block(inputs,64)
        s2,p2=Encoder_block(p1,128)
        s3,p3=Encoder_block(p2,256)
        s4,p4=Encoder_block(p3,512)


        b1=conv_block(p4,1024)

        #inputs,skip,num_filters
        d1=decoder_block(b1,s4,512)
        d2=decoder_block(d1,s3,256)
        d3=decoder_block(d2,s2,128)
        d4=decoder_block(d3,s1,64)

        outputs=Conv2D(1,1,padding='same',activation='sigmoid')(d4)
        model=Model(inputs=inputs,outputs=outputs,name='u-net')
        model.load_weights(output)  

        return model
    
    test_data=os.path.join(image_folder,selected_image)
    image = tf.io.read_file(test_data)
    image = tf.image.decode_png(image, channels=3)  # Decode image as RGB
    image = tf.image.resize(image, (128, 128)) 
    model=build_unet((128,128,3))
    a=np.expand_dims(image,axis=0)
    y_hat=model.predict(a)
    y_hat=np.squeeze(y_hat,axis=0)
    st.markdown("### Segmented image")
    st.image(y_hat,caption="Predicted Image",width=350)

