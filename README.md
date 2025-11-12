# Flower Recognition using Convolutional Neural Networks (CNN)

This repository contains an end-to-end deep learning project for classifying flower images using a Convolutional Neural Network (CNN). The goal is to identify different flower species based on image data, using TensorFlow/Keras for model development.

---

## Project Overview

The dataset includes images stored in class-specific folders. The notebook performs the following steps:

1. Loading image files and extracting labels  
2. Resizing and normalizing all input images  
3. Visualizing samples from the dataset  
4. Encoding labels as integers and one-hot vectors  
5. Creating data augmentation pipelines  
6. Building a CNN model  
7. Training and validating the model  
8. Plotting accuracy and loss curves  
9. Running predictions on test images  
10. Displaying correct and incorrect classifications  
11. Generating a confusion matrix for detailed analysis

---

## Dataset Structure

The dataset is organized with one folder per class. During preprocessing, the code iterates through each folder, loads images, resizes them, normalizes pixel values, and collects corresponding labels.

Example structure:
flowers/
daisy/
dandelion/
rose/
sunflower/
tulip/

Images are converted into NumPy arrays for model training.

---

## Model Architecture

The CNN used in this project includes:

- Convolutional layers with ReLU activation  
- Max Pooling layers  
- Dropout for regularization  
- Flattening layer  
- Dense layers  
- Softmax output layer for multi-class classification  

The model is compiled with:

- Loss: categorical crossentropy  
- Optimizer: Adam  
- Metric: accuracy  

The project also uses an ImageDataGenerator with augmentation:
- rotation range  
- zoom range  
- shear range  
- width and height shifts  
- horizontal flipping  

---

## Training and Evaluation

The dataset is split into training and testing sets.  
During training, the model tracks training accuracy, validation accuracy, training loss, and validation loss.

After training:

- Final accuracy and loss are computed  
- Predictions are generated for all test images  
- A confusion matrix is displayed  
- Correct and incorrect predictions are visualized  

The notebook also prints class names for better interpretation of predictions.

---

## Results

In the final run of the notebook, the model achieved an accuracy of approximately 71 percent on the test set.  
Accuracy may vary depending on the number of epochs and augmentation settings.

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- OpenCV  
- scikit-learn  

---

## How to Run

1. Upload the dataset (flower images) in the folder structure described above.  
2. Run the notebook cell by cell.  
3. Ensure that TensorFlow, NumPy, Matplotlib, and OpenCV are installed.  

Example installation:

pip install tensorflow numpy matplotlib opencv-python scikit-learn
