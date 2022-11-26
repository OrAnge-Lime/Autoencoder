from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

def build_autoencoder(img_size):
    input_img = Input(shape=(img_size, img_size, 3))

    # encoding
    nn = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    nn = MaxPooling2D((2, 2), padding='same')(nn)
    nn = Conv2D(32, (3, 3), activation='relu', padding='same')(nn)
    encoded = MaxPooling2D((2, 2), padding='same')(nn)


    # learns how to remove the noise from the input images
    nn = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    nn = UpSampling2D((2, 2))(nn)
    nn = Conv2D(32, (3, 3), activation='relu', padding='same')(nn)
    nn = UpSampling2D((2, 2))(nn) # rebuild the images to the original dimensions
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(nn)

    model = Model(input_img, decoded)
    
    return model