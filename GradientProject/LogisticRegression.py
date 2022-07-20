from variable import Variable
import math
import numpy as np

class LogisticRegression():
    
    def __init__(self, iterations, learning_rate = 0.01):
        self.iterations = iterations
        self.learning_rate = learning_rate
        
    def fit(self,X,y):
        rows,cols = np.shape(X)
        
        m = []
        y_predicted = []
        
        b = Variable(name = 'b')
        
        cost_list = []
        
        for i in range(cols):
            m.append(Variable(name = 'm_' + str(i)))
            
        for i in range(rows):
            y_predicted.append(1 / (1 + Variable.exp(sum(np.array(m * X[i] + b)))))
            
        for i in range(rows):
            cost_list.append(y[i] * Variable.log(y_predicted[i]) + (1 - y[i]) * Variable.log(1 - y_predicted[i]))
        
        cost = -1 * sum(cost_list)
        
        self.M = np.random.rand(cols)
        self.b = np.random.rand()
        
        descent_minvalues = {'b': self.b}
        iter_dict = {'b': self.b}
        
        for i in range(cols):
            descent_minvalues.update({'m_' + str(i): self.M[i]})
            iter_dict.update({'m_' + str(i): self.M[i]})
            
        for i in range(self.iterations):
            step = cost.grad(iter_dict)
            
            self.b -= self.learning_rate * step[0]
            
            iter_dict['b'] = self.b
            
            self.M -= self.learning_rate * step[1:]
            
            for i in range(cols):
                iter_dict['m_' + str(i)] = self.M[i]
            
            cost_temp = cost.evaluate(iter_dict)
            
            if cost_temp < cost.evaluate(descent_minvalues):
                descent_minvalues['b'] = self.b
                for i in range(cols):
                    descent_minvalues['m_' + str(i)] = self.M[i]
        
        return descent_minvalues
        
    def predict(self,x):
        
        y_predictions = []
        rows,cols = np.shape(x)
        
        for i in range(rows):
            cur_score = 1 / (1 + math.exp(sum(np.array(self.M * x[i] + self.b))))
            y_predictions.append(0 if cur_score < 0.5 else 1)
        
        return y_predictions
        
# Testing: every array of X values corresponds to ONE Y Value 
X,y = np.array([[3,4,5,6,8],[1,2.9,3.1,4,5],[2,5,6,8,9],[2.5,3,3.1,3.8,4]]), np.array([0,1,0,1])
model = LogisticRegression(iterations=1000,learning_rate=0.001)
model.fit(X, y)
X_test, y_true = np.array([[1,2,3,4,5],[1,2.7,3.2,4.6,6],[1,2.9,3.1,3.4,4],[2,3,6,7,8]]), np.array([1,1,1,0])
y_preds = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_preds))
#We should get around 1.0 since the training data made datasets that were centered around 3 correct and the other ones wrong.