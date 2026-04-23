import pandas as pd
from preprocessing import DataPreprocessor
from logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from evaluate import evaluate_model, evaluate_thresholds,plot_confusion_matrix
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("customerdata.csv")

# Preprocessing
processor = DataPreprocessor()
df = processor.clean_data(df)

X = df.drop("Churn",axis = 1)
y = df.Churn

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)

# Scaling
X_train, X_test = processor.scale_features(X_train, X_test)

# Train model
n_iters = 1000
learning_rate = 0.01

model = LogisticRegression(n_iters=n_iters, learning_rate=learning_rate)

cost_history = model.fit(X_train, y_train)


plt.figure(figsize = (5,5))
plt.plot(range(n_iters),cost_history)
plt.title("Cost vs Iterations")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.savefig('results/cost_plot.png')
plt.close()

# Results of the model with threshold = 0.5
threshold = 0.5
probs = model.predict_prob(X_test)
prediction = (probs > threshold).astype(int)
evaluate_model(y_test,prediction)

# Tuning threshold

thresholds = [0.5,0.4,0.3,0.2]

f1_scores = evaluate_thresholds(y_test,probs,thresholds)

plt.plot(thresholds,f1_scores,marker = 'o')
plt.xlabel("Thresholds")
plt.ylabel("F1 Score")
plt.savefig('results/f1_vs_threshold.png')
plt.close()


# Setting threshold and predicting.
threshold = 0.4
prediction = (probs > threshold).astype(int)
evaluate_model(y_test,prediction)


plot_confusion_matrix(y_test,prediction)
