import numpy as np
import math

class Variable():
    
    num_of_variables = 0
    all_variables = []
    
    def __init__(self, name=None, evaluate=None, grad=None):
        if evaluate == None:
            self.evaluate = lambda values: values[self.name]
        else:
            self.evaluate = evaluate
            
        if name != None:
            self.name = name          # its key in the evaluation dictionary
            self.all_variables.append(self)
        
        if grad == None:
            Variable.num_of_variables += 1
            self.current = Variable.num_of_variables
            self.grad = self.build_gradient
        else:
            self.grad = grad
            
    def build_gradient(self, variables):
        
        total = self.num_of_variables
        current = self.current

        output = [0] * total
        output[current - 1] = 1
        
        return np.array(output)
    
    def reset_list(list_name):
        list_name = []
        return list_name
    
    def print_numbers(self):
        print("Variable number {} of {}".format(self.current, Variable.num_of_variables))
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) + other, grad = lambda values: self.grad(values))
            
        return Variable(evaluate = lambda values: self.evaluate(values) + other.evaluate(values), grad = lambda values: self.grad(values) + other.grad(values))
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) - other, grad = lambda values: self.grad(values))
            
        return Variable(evaluate = lambda values: self.evaluate(values) - other.evaluate(values), grad = lambda values: self.grad(values) + other.grad(values)*-1)
    
    def __rsub__(self, other):
        return -1 * self + other 
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) * other, grad = lambda values: self.grad(values) * other)
            
        return Variable(evaluate = lambda values: self.evaluate(values) * other.evaluate(values), grad = lambda values: self.evaluate(values) * other.grad(values) + self.grad(values) * other.evaluate(values))
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Variable(evaluate = lambda values: self.evaluate(values) / other, grad = lambda values: self.grad(values) * other**-1)
            
        return Variable(evaluate = lambda values: self.evaluate(values) / other.evaluate(values), grad = lambda values: -1 * self.evaluate(values) * other.grad(values) * other.evaluate(values)**-2 + self.grad(values) * other.evaluate(values)**-1)
    
    def __rtruediv__(self, other):
        return self**-1 * other
    
    def __pow__(self,other):
        return Variable(evaluate = lambda values: self.evaluate(values) ** other, grad = lambda values: other * self.evaluate(values)**(other - 1) * self.grad(values))
    
    def __rpow__(self, other):
        return Variable(evaluate = lambda values: other ** self.evaluate(values))
    
    def exp(self):
        return Variable(evaluate = lambda values: math.e ** self.evaluate(values), grad = lambda values: math.e ** self.evaluate(values) * self.grad(values))
    
    def log(self):
        return Variable(evaluate = lambda values: math.log(self.evaluate(values)), grad = lambda values: self.evaluate(values)**-1 * self.grad(values))
