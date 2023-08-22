# Speech Emotion Recognition

**PRML Major Project - Speech Emotion Recognition**

Authors: Renu Sankhla (B21AI028), Arvind Kumar Sharma (B21AI006), Nitish Bhardwaj (B21AI056)
Date: 28 April 2023

## Contribution
- Renu Sankhla
- Arvind Kumar Sharma
- Nitish Bhardwaj


## Work & Models
- Dataset Creation
- Mel Spectrogram Dataset
- FFT Dataset
- Decision Tree
- Naive Bayes
- KNN
- Adaboost
- Random Forest
- SVM
- ANN with Tanh, sigmoid, and Relu activation

## Problem Statement: Speech Emotion Recognition
Speech Emotion Recognition (SER) aims to recognize human emotion and affective states from speech. This involves analyzing voice for tone and pitch variations that reflect underlying emotion.

## Pipeline
1. Data Preprocessing
2. Data Cleaning
3. Feature Engineering
4. Model Selection
5. Prediction
6. Conclusion

## Data Preprocessing
- Converting audio files to numpy arrays using the librosa library.
- Using only the first 2.9 seconds of audio files to ensure consistent intervals.

## Feature Engineering
- Fourier Transform: Capturing frequency information.
- Mel Spectrogram: Using time and frequency domains.
- Applying PCA and LDA to reduce dimensionality.

## Model Selection
1. KNN
2. SVM (Linear and rbf kernels)
3. Decision Tree
4. Adaboost
5. Random Forest
6. Naive Bayes
7. ANN with various activation functions

## Prediction
### KNN (Number of Neighbors: 8)
- KNN Classifier Report on Fourier Transform Dataset
  Accuracy: 0.958, Precision: 0.958, Recall: 0.958

- KNN Classifier Report on Mel Spectrogram Dataset
  Accuracy: 0.972, Precision: 0.972, Recall: 0.972

### SVM (Kernel = Linear and rbf)
- SVM Report on Fourier Transform Dataset
  Accuracy: 0.954, Precision: 0.954, Recall: 0.954

- SVM Report on Mel Spectrogram Dataset
  Accuracy: 0.958, Precision: 0.958, Recall: 0.958

### Decision Tree
- Decision Tree Report on Fourier Transform Dataset with PCA and LDA
  Weighted Accuracy with Gini: 0.29 (PCA), 0.23 (LDA)
  
- Decision Tree Report on Mel Spectrogram Dataset with PCA and LDA
  Weighted Accuracy with Gini: 0.25 (PCA), 0.23 (LDA)

### Adaboost
- Adaboost Report on Fourier Transform Dataset with PCA and LDA
  Weighted Accuracy with Gini: 0.24 (PCA), 0.21 (LDA)

- Adaboost Report on Mel Spectrogram Dataset with PCA and LDA
  Weighted Accuracy with Gini: 0.30 (PCA), 0.31 (LDA)

### Naive Bayes
- Naive Bayes Report on Fourier Transform and Mel Spectrogram Datasets
  Accuracy: 0.32 - 0.36, Precision: 0.33 - 0.34, Recall: 0.32 - 0.45, F1-score: 0.30 - 0.39

### Random Forest
- Random Forest Report on Fourier Transform and Mel Spectrogram Datasets
  Accuracy: 0.410 - 0.458, Precision: 0.443 - 0.436, Recall: 0.403 - 0.438, F1-score: 0.379 - 0.432

### ANN
- Accuracy comparison for different activation functions and datasets

## Conclusion
In this project, we explored various models for Speech Emotion Recognition. Our experiments indicate that certain models and features perform better than others for this task. The choice of features and models has a significant impact on the accuracy of emotion recognition from speech.

Feel free to refer to our detailed reports and analysis in the repository.

