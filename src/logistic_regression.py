import numpy as np

class LogisticRegression:


    def __init__(self,n_iters = 100,learning_rate = 0.01):

        self.n_iters = n_iters
        self.learning_rate = learning_rate

        self.weights = None
        self.bias = 0.0

    # Logistic function - Sigmoid
    def sigmoid(self,z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self,X,y):
        

        n_samples , n_features = X.shape


        self.weights = np.zeros(n_features)
        cost_history = []
        loss = 0
        # Gradient Descent
        for _ in range(self.n_iters):
            Z = np.dot(X,self.weights) + self.bias

            A = self.sigmoid(Z)

            # Use epsilon to avoid undefined log values.
            epsilon = 1e-8
            A = np.clip(A,epsilon,1 - epsilon)


            loss = -(y * np.log(A) + (1 - y) * np.log(1 - A))
            cost = np.mean(loss)
            cost_history.append(cost)

            # dL / dz
            dZ = A - y


            # X has a shape (n_samples,n_features) and dZ has (n_samples).
            dw = np.dot(X.T,dZ) / n_samples

            db = np.sum(dZ) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return cost_history


    def predict_prob(self,X):

        

        Z = np.dot(X,self.weights) + self.bias
        
        return self.sigmoid(Z)