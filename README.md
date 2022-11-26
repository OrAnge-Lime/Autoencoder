# Autoencoder
## Description
An autoencoder is a special type of neural network that is trained to copy its input to its output. For example, given an image of a handwritten digit, an autoencoder first encodes the image into a lower dimensional latent representation, then decodes the latent representation back to an image.
This architecture can be used for example in image deniosing. This repository represents simple image denoiser with using Autoencoder neural network that was trained on a small cat-images dataset.

![training_res](https://user-images.githubusercontent.com/71509624/204093141-f7c8c6c6-7b6a-4192-94d6-554ed20ed16e.png)

## Usage
To use this code you need to follow te comand `python3 usage.py <image path>`, where <image path> is a path to image wich need to be denoised. For testing you can use noise_img.png to which I already add some noise.
As a result you will get two images:
 1. `fig.png` - shows the origin noised image and the result of denoising at once
 
 ![fig](https://user-images.githubusercontent.com/71509624/204092993-11f9bd96-e287-44a7-90b7-0e8375084877.png)
 
 2. `denoised_img.png` - shows the denoised image in proper resolution
 
 ![denoised_img](https://user-images.githubusercontent.com/71509624/204093001-d94c4d0c-cd5e-4c09-85ff-0396dceecccd.png)

