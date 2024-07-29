# Fashion MNIST Classification Project

## Overview
This project involves the application of various machine learning algorithms to the Fashion MNIST dataset, a popular dataset for benchmarking image classification models. The dataset consists of 60,000 training images and 10,000 test images of fashion items, each image being a 28x28 grayscale image.

## Models and Performance

### Logistic Regression
- **In-sample Accuracy:** 86.07%
- **Out-of-sample Accuracy:** 85.49%
- **AUC Scores:** High AUC scores for most classes, with a slightly lower performance for class 6.

### Support Vector Machine (SVM)
- **Out-of-sample Accuracy:** 86.57%
- **AUC Scores:** High AUC scores across most classes, slightly better than Logistic Regression in several classes. Marginally better performance for class 6.

### Random Forest Classifier
- **In-sample Accuracy:** 88.25%
- **Out-of-sample Accuracy:** 86.03%
- **AUC Scores:** High AUC scores, particularly strong for classes 1, 5, 8, and 9. Slightly lower performance for class 6 compared to SVM.

### K-Nearest Neighbors (KNN)
- **In-sample Accuracy:** 82.88%
- **Out-of-sample Accuracy:** 82.00%
- **AUC Scores:** Not provided, but accuracy suggests decent performance with potential challenges in high-dimensional space and computational cost.

## Summary and Recommendations
- **Logistic Regression:** Provides a solid baseline with good overall performance and high interpretability. Suitable for quick insights and baseline comparisons.
- **SVM:** Achieves the highest out-of-sample accuracy (86.57%) with strong AUC scores, making it a robust choice for distinguishing between most classes. Slightly better at handling class 6 compared to other models.
- **Random Forest:** Exhibits the highest in-sample accuracy (88.25%) and strong out-of-sample performance. Excellent for capturing complex patterns and interactions, with the added benefit of feature importance insights.
- **KNN:** Demonstrates decent performance with lower accuracy compared to other models. KNN's simplicity is balanced by its computational cost and sensitivity to high-dimensional data.

### Overall Recommendation
- **Best Performance:** SVM and Random Forest models show the best overall performance, with SVM slightly leading in out-of-sample accuracy and better handling class 6.
- **Balanced Choice:** Random Forest offers a good balance with high accuracy, robustness to overfitting, and feature importance insights.

For future work, consider further tuning hyperparameters and exploring advanced models like neural networks to potentially improve performance, especially for challenging classes like class 6.

## Installation

To run this project, ensure you have the following libraries installed:
- numpy
- pandas
- scikit-learn
- plotly

You can install them using pip:
```bash
pip install numpy pandas scikit-learn plotly
