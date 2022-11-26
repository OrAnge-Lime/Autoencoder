import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import Model

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to the noise image", default="noise_img.png")
args = parser.parse_args()

IMG_SIZE = 128
NOISE_IMG = args.path

# MODEL
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

nn = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
nn = MaxPooling2D((2, 2), padding='same')(nn)
nn = Conv2D(32, (3, 3), activation='relu', padding='same')(nn)
encoded = MaxPooling2D((2, 2), padding='same')(nn)

nn = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
nn = UpSampling2D((2, 2))(nn)
nn = Conv2D(32, (3, 3), activation='relu', padding='same')(nn)
nn = UpSampling2D((2, 2))(nn) 
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(nn)

autoencoder = Model(input_img, decoded)
autoencoder.load_weights("autoencoder.h5")

img = Image.open(NOISE_IMG)
img = img.resize((IMG_SIZE,IMG_SIZE))
img = np.asarray(img)
img = img / 255

test = autoencoder.predict(img.reshape(1, IMG_SIZE, IMG_SIZE, 3))

fig, axis = plt.subplots(1, 2)
fig.set_size_inches(18.5, 10.5)
axis[0].imshow(img)
axis[1].imshow(test.reshape(IMG_SIZE, IMG_SIZE, 3))
fig.savefig("fig.png")

res = Image.fromarray((test.reshape(IMG_SIZE, IMG_SIZE, 3)*255).astype(np.uint8))

res.save("denoised_img.png")

