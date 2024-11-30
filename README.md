# MNIST_with_pbits_GC

This repository contains a MATLAB implementation of the training of the MNSIT dataset using probabilistic bits (p-bits).  It includes the graph-colored CPU version of the algorithm implemented in

"Training deep Boltzmann networks with sparse Ising machines"
by Shaila Niazi, Shuvro Chowdhury, Navid Anjum Aadit, Masoud Mohseni, Yao Qin & Kerem Y. Camsari
Nature Electronics volume 7, pages 610-619 (2024)

The results in the paper were obtained by implementing a similar version of this code on FPGA. Additionally, the "Outputs" folder contains the weights and bias values trained for 100 epochs that have been accomplished using FPGA. Those values can be used to run the inference code ("main_test.m"). As sampling in MATLAB is expensive, checking the accuracy for the full test set will take an enormous time (for a quick check, one can test the accuracy for 100 images, for example). 

## Requirements

- MATLAB (developed and tested on recent versions)
## Dataset
The full dataset has been provided here.
One can directly download the MNIST dataset from the website: https://yann.lecun.com/exdb/mnist/index.html

## Usage

1. Clone this repository:
   ```
   git clone https://github.com/ShuvroChowdhury/MNIST_with_pbits_GC.git
   ```

2. Open MATLAB and navigate to the cloned directory.

3. Run the main script:
   ```matlab
   main_train.m
   ```
   ```matlab
   main_test.m
   ```
   ```matlab
   Image_generation.m
   ```

4. The scripts will train and plot the training and the test accuracy. Image generation code is also provided.

## Code Structure

- `main_train.m`: The script sets up parameters and trains for MNIST. It also includes the inference for training images if one wants to check that while training.
- `main_test.m`: The main script tests the accuracy of both training and test set images once the training has been performed.
- `Image_generation.m`: The script will generate images using the trained weights and biases according to the label.

## Customization

The training code includes the features for training, both full MNIST and MNIST/100, based on the selection. MNIST/100 includes 100 images (10 for every 10 digits). You can modify the following parameters in the `main.m` script:

- `num_images`: Number of images to be trained
- `size_batch`: Size of a batch of images
- `num_samples`: Number of samples to be read
- `NL`: Number of epochs
- `num_samples_to_wait_neg`: Number of sweeps to wait between two samples read in the negative phase (to emulate the sampling in FPGA)
- `num_samples_to_wait`: Number of sweeps to wait between two samples read during image testing 
- `eps`: Learning parameter
- `lambda`: Regularization parameter
- `al_pha`: Momentum parameter
- `beta`: Inverse temperature % kept to 1 while training

The inference code can be used for both training and test set images. You can modify the following parameters in the `main_test.m` script:

- `num_truth_tables`: Number of images from the training set
- `num_images_test`: Number of images from the test set
- `num_samples`: Number of samples to be read
- `num_samples_to_wait`: Number of sweeps to wait between two samples read during image testing 
- `beta`: Inverse temperature
- 
The image generation code can generate images according to the label. You can modify the following parameters in the `main_test.m` script:

- `image_index`: the label to clamp based on the image
- `num_samples`: Number of samples to be read
- `num_samples_to_wait`: Number of sweeps to wait between two samples read during image testing 
- `beta`: Inverse temperature
- Jout and hout: Different weights and biases can be loaded from different epochs

## Contributing

Contributions to improve the code or extend its functionality are welcome. Please feel free to submit issues or pull requests.


## Acknowledgements

This implementation is based on the algorithms as described in:

- "A Practical Guide to Training Restricted Boltzmann Machines" by Geoffrey E. Hinton

## Contact

If you have any questions or suggestions, please open an issue in this repository or contact Shaila Niazi (sniazi@ucsb.edu).
