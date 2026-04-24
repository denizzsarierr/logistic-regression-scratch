# Logistic Regression from Scratch (Binary Classification)
This project is implementing Logistic Regression using NumPy**, without using Scikit-learn’s model implementation.

The goal is to undersant mathematical and practical working of Logistic Regression, including gradient descent optimization, loss computation, and decision threshold tuning.

---

## Problem Statement

The goal is to predict customer churn basen on different behaviors and features. This is a binary classification problem.

---

## What I Built ?
- Logistic Regression algorithm implemented from scratch by using NumPy library.
- Sigmoud Activation Function
- Gradient Descent Algorithm to minimize cost
- Binary Cross Entropy Loss
- Feature Preprocessing Pipeline
- Threshold Tuning for Optimal Performance
- Model Evalution using F1,Recall,Precision and Accuracy metrics.

---

## Pipeline
 - Load dataset
 - Data preprocessing:
   - Encoding Categorical Variables
   - Handling Missing Values
   - Feature Scaling
 - Train-test split
 - Training logistic regression model from scratch
 - Evaluate model's performance
 - Tune classification threshold
 - Visualize results

---

## Key Concepts (formulas) Implemented
### Sigmoid Function
The sigmoid (logistic) function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

It outputs a probability value between 0 and 1.

In the implementation, a small constant $( \epsilon )$ is used to clip predicted probabilities away from 0 and 1. 

This prevents numerical instability in the logarithm function, since log(0) is undefined.

---

### Loss Function (Binary Cross Entropy)
The loss function for logistic regression is defined as:

$$
L(y, \hat{y}) = -\left[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})\right]
$$

The cost function is the average of the binary cross-entropy loss over all training samples:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

In the implementation, the cost function values are stored in a list to visualize the convergence of gradient descent.

For logistic regression, the cost function is minimized using gradient descent.

After derivation using the chain rule and sigmoid function, the gradients are:

Weight Gradient:

$$
\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)
$$

Bias Gradient:

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

The term $( \hat{y} - y )$ represents the prediction error.

The gradients show how much each weight contributes to that error.

Gradient descent updates parameters in the direction that reduces this error.

---

## Results

- Best Threshold: **0.4**
- Best F1 Score: **0.6064**
- Accuracy Score: **0.79**
- Model evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## Visualizations

The project includes:

- Cost vs Iterations plot (convergence of training)
- Confusion Matrix
- F1 Score vs Threshold plot

All plots are saved in the `/results` folder.

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (for preprocessing & metrics only)

## Project Structure

lr-from-scratch/

│

├── src/

│ ├── logistic_regression.py

│ ├── preprocessing.py

│ ├── evaluate.py

│
├── train.py

├── requirements.txt

├── README.md

│

├── results/

│ ├── cost_plot.png

│ ├── confusion_matrix.png

│ ├── f1_vs_threshold.png


## How to Run?

pip install -r requirements.txt

python train.py

--- 

## What I Learned
- How logistic regression works internally
- What is the logic behind basic learning model
- Importance of feature scaling
- Gradient descent optimization in practice (weight and bias updating)
- Importance of vectorizing on efficiency, especially on large datasets.
- Effect of decision threshold on classification performance
- Analyzing confusion matrix
- End-to-end ML pipeline design

---
