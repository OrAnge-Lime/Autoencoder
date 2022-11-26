import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from model import build_autoencoder

IMG_SIZE = 128

def add_noise(img, noise_factor = 0.5):
    img = img / 255
    x = img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape) 
    x = np.clip(x, 0., 1.)
    return x

img = Image.open("test.png")
x = add_noise(np.asarray(img))
x = Image.fromarray((x * 255).astype(np.uint8))

fig, axis = plt.subplots(1, 2)
axis[0].imshow(img)
axis[1].imshow(x)

import os 

x_train = []
x_train_noisy = []
DATASET_DIR = "ds/"

for img in os.listdir(DATASET_DIR):
        image = Image.open(DATASET_DIR + img)
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = np.asarray(image)

        x_train.append(image / 255)
        x_train_noisy.append(add_noise(image))
        

autoencoder = build_autoencoder(IMG_SIZE)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

autoencoder.fit(np.asfarray(x_train_noisy), np.asfarray(x_train),
                epochs=200,
                batch_size=128,
                shuffle=True)
                
autoencoder.save_weights("autoencoder.h5")

img = Image.open("test.png")
img = img.resize((IMG_SIZE,IMG_SIZE))
x = add_noise(np.asarray(img), noise_factor=0.3)
print(x.shape)

test = autoencoder.predict(x.reshape(1, IMG_SIZE, IMG_SIZE, 3))

fig, axis = plt.subplots(1, 3)
fig.set_size_inches(18.5, 10.5)
axis[0].imshow(img)
axis[1].imshow(x)
axis[2].imshow(test.reshape(IMG_SIZE, IMG_SIZE, 3))
fig.savefig("training_res.png")