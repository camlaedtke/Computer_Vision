import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D



def unet(input_height, input_width, n_classes = 3):
    
    img_input = tf.keras.layers.Input(shape=(input_height, input_width, 3))

    # -------------------------- Encoder --------------------------
    
    c1 = Conv2D(64, 3, padding='same', activation="selu")(img_input)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, 3, padding='same', activation="selu")(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(128, 3, padding='same', activation="selu")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, 3, padding='same', activation="selu")(c2)
    p2 = MaxPooling2D((2,2))(c2)
    p2 = Dropout(0.1)(p2)
    
    c3 = Conv2D(256, 3, padding='same', activation="selu")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, 3, padding='same', activation="selu")(c3)
    p3 = MaxPooling2D((2,2))(c3)
    p3 = Dropout(0.2)(p3)
    
    c4 = Conv2D(512, 3, padding='same', activation="selu")(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, 3, padding='same', activation="selu")(c4)
    p4 = MaxPooling2D((2,2))(c4)
    p4 = Dropout(0.3)(p4)
    
    # ------------------------ Bottleneck -------------------------
    
    c5 = Conv2D(1024, 3, padding='same', activation="selu")(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(1024, 3, padding='same', activation="selu")(c5)
    c5 = Dropout(0.5)(c5)
    
    # -------------------------- Decoder --------------------------
    
    u6 = concatenate([UpSampling2D(2)(c5), c4])
    c6 = Conv2D(512, 3, padding='same', activation='selu')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(256, 3, padding='same', activation='selu')(c6)
    c6 = Dropout(0.3)(c6)
    
    u7 = concatenate([UpSampling2D(2)(c6), c3])
    c7 = Conv2D(256, 3, padding='same', activation='selu')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(128, 3, padding='same', activation='selu')(c7)
    c7 = Dropout(0.2)(c7)

    u8 = concatenate([UpSampling2D(2)(c7), c2])
    c8 = Conv2D(128, 3, padding='same', activation='selu')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(64, 3, padding='same', activation='selu')(c8)
    c8 = Dropout(0.1)(c8)

    u9 = concatenate([UpSampling2D(2)(c8), c1]) 
    c9 = Conv2D(64, 3, padding='same', activation='selu')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(64, 3, padding='same')(u9)
    c9 = Activation('selu')(c9)
    c9 = Conv2D(n_classes, 3, padding='same')(c9)
    
    output = Activation("softmax", dtype='float32')(c9)
    
    return tf.keras.Model(inputs=img_input, outputs=output)


def segnet(input_height, input_width, pool_size=(2,2), n_classes = 3):
    
    img_input = tf.keras.layers.Input(shape=(input_height, input_width, 3))

    # -------------------------- VGG Encoder 1 --------------------------
    c1 = Conv2D(64, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(img_input)
    c1 = Conv2D(64, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c1)
    p1, p1_ind = MaxPoolingWithArgmax2D(pool_size, dtype="float32")(c1)
    # -------------------------- VGG Encoder 2 --------------------------
    c2 = Conv2D(128, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(p1)
    c2 = Conv2D(128, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c2)
    p2, p2_ind = MaxPoolingWithArgmax2D(pool_size, dtype="float32")(c2)
    # -------------------------- VGG Encoder 3 --------------------------
    c3 = Conv2D(256, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(p2)
    c3 = Conv2D(256, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c3)
    c3 = Conv2D(256, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c3)
    p3, p3_ind = MaxPoolingWithArgmax2D(pool_size, dtype="float32")(c3)
    # -------------------------- VGG Encoder 4 --------------------------
    c4 = Conv2D(512, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(p3)
    c4 = Conv2D(512, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c4)
    c4 = Conv2D(512, 3, padding='same', activation="selu", kernel_initializer = 'he_normal')(c4)
    p4, p4_ind = MaxPoolingWithArgmax2D(pool_size, dtype="float32")(c4)

    
    # ---------------------- Maxpool Index Decoder 1 --------------------
    u3 = MaxUnpooling2D(pool_size, dtype="float32")([p4, p4_ind])
    c6 = Conv2D(512, 3, padding='same', kernel_initializer = 'he_normal')(u3)
    c6 = BatchNormalization()(c6)
    c6 = Activation("selu")(c6)
    c6 = Conv2D(512, 3, padding='same', kernel_initializer = 'he_normal')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation("selu")(c6)
    c6 = Conv2D(256, 3, padding='same', kernel_initializer = 'he_normal')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation("selu")(c6)
    # ---------------------- Maxpool Index Decoder 2 --------------------
    u4 = MaxUnpooling2D(pool_size, dtype="float32")([c6, p3_ind])
    c7 = Conv2D(256, 3, padding='same', kernel_initializer = 'he_normal')(u4)
    c7 = BatchNormalization()(c7)
    c7 = Activation("selu")(c7)
    c7 = Conv2D(256, 3, padding='same', kernel_initializer = 'he_normal')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation("selu")(c7)
    c7 = Conv2D(128, 3, padding='same', kernel_initializer = 'he_normal')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation("selu")(c7)
    # ---------------------- Maxpool Index Decoder 3 --------------------
    u5 = MaxUnpooling2D(pool_size, dtype="float32")([c7, p2_ind])
    c8 = Conv2D(128, 3, padding='same', kernel_initializer = 'he_normal')(u5)
    c8 = BatchNormalization()(c8)
    c8 = Activation("selu")(c8)
    c8 = Conv2D(64, 3, padding='same', kernel_initializer = 'he_normal')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation("selu")(c8)
    # ---------------------- Maxpool Index Decoder 4 --------------------
    u6 = MaxUnpooling2D(pool_size, dtype="float32")([c8, p1_ind])
    c9 = Conv2D(64, 3, padding='same', kernel_initializer = 'he_normal')(u6)
    c9 = BatchNormalization()(c9)
    c9 = Activation("selu")(c9)
    c9 = Conv2D(64, 3, padding='same', kernel_initializer = 'he_normal')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation("selu")(c9)
    c9 = Conv2D(n_classes, 3, padding='same')(c9)
    
    output = Activation("softmax", dtype='float32')(c9)
    
    return tf.keras.Model(inputs=img_input, outputs=output)   



def unet_xception(input_height,  input_width, n_classes = 4):
    
    inputs = tf.keras.layers.Input(shape=(input_height, input_width, 3))

    ######## [First half of the network: downsampling inputs] ########

    # Entry block
    x = Conv2D(64, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("selu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [128, 256, 512]:
        x = Activation("selu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("selu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ######## [Second half of the network: upsampling inputs] ########

    previous_block_activation = x  # Set aside residual

    for filters in [512, 256, 128, 64]:
        x = Activation("selu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("selu")(x)
        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    x = Conv2D(n_classes, 3, padding="same")(x)
    
    outputs = Activation("softmax", dtype='float32')(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model