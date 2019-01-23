# Image Classification

## How to Approach the Image Classification problem? 
 
1) Try **Transfer Learning** (aka Feature Extraction). 
    * Use Pretrained model weights.
    * Don't include the top (ie., remove the head of the network)
    * Extract the features from the image and store the feacture vectors.
    * Build the classifier based on the extracted features.
    * Based on the model, classify the test images.

Let's say, you got a reasonable (baseline) accuracy. How the next question is how to improve it?

## How to improve the accuracy?

2) Try **Transfer Learning (Feature Extraction)** with different deep Neural network model and examine whether that bumps up the accuracy.

3) Try **Transfer Learning with Fine Tuning**
    * Pick the neural network with pre-trained weights without including the top.
    * Add, new head (Fully connected layer) and train the new head by freezing the weights of the early part of the network (which has CONV / Pool layers).
    * Unfreeze some Conv layers in the early part of the network and re-train with the training images.
    * Predict the classes for the test images.

4) Try **Feature Extraction with Data Augmentation**
    * During feature extraction of an image, you can create augmented images (say, using ImageDataGenerator() in keras) and extract the features for the same.
    * Build classifier based on extracted features and classify the test images.
    * **This approach may result in reducing the accuracy** as it reduces the **overfit**.
    


