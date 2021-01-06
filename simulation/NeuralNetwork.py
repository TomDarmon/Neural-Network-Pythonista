from random import random, randrange
from math import e
from array_functions import *
from random import choice
import numpy as np
import time


def cross_over(nn1, nn2):
		child = Neural_Network(nn1.inputs_number, nn1.hidden_number, nn1.outputs_number)
		param_1 = [nn1.weights_IH, nn1.weights_HO, nn1.bias_H, nn1.bias_O]
		param_2 = [nn2.weights_IH, nn2.weights_HO, nn2.bias_H, nn2.bias_O]
		child_param = []
		
		for i in range(4):
			what_param = randrange(1, 3)
			if what_param == 1:
				child_param.append(param_1[i])
			elif what_param == 2:
				child_param.append(param_2[i])
				
				
		child.weights_IH = child_param[0]
		child.weights_HO = child_param[1]
		child.bias_H = child_param[2]
		child.bias_O = child_param[3]
		
		return child
		

class Neural_Network:
	
	def __init__(self, inputs, hidden, outputs):
		
		self.inputs_number = inputs
		self.hidden_number = hidden
		self.outputs_number = outputs
		
		self.weights_IH = np.random.rand(self.hidden_number, self.inputs_number)

		self.bias_H = np.random.rand(self.hidden_number, 1)
		
		self.weights_HO = np.random.rand(self.outputs_number, self.hidden_number)
		
		self.bias_O = np.random.rand(self.outputs_number,1)
		
		
		self.learning_rate = 0.1
		self.mutation_prob = 0.3
		
	def settings(self):
		print("Weights_IH : ")
		print(self.weights_IH)
		
		print("Bias_H : ")
		print(self.bias_H)
		
		print("Weights_HO : ")
		print(self.weights_HO) 
		
		print("Bias_O : ")
		print(self.bias_O)
	
	def predict(self, inputs):
		if type(inputs) != np.ndarray:
			raise Exception("The inputs is not a numpy array")
		if inputs.shape[0] != self.inputs_number:
			raise Exception("The inputs size doesn't match the neuralnetwork inputs number'")
		
		#passing from input to hidden
		hidden = np.dot(self.weights_IH, inputs)
		hidden += self.bias_H
		hidden = vectorize_sigmoid(hidden)
		
		#hidden to output
		outputs = np.dot(self.weights_HO, hidden)
		outputs += self.bias_O
		outputs = vectorize_sigmoid(outputs)
		
		return outputs
		
			
		
		
		
	def mutate(self):
		mutation_choices = [self.weights_IH, self.bias_H, self.bias_O, self.weights_HO]
		'''random_index = randrange(0, 4)
		array_to_mutate = mutation_choices[random_index]
		mutate(array_to_mutate, self.mutation_prob)	
		mutation_choices[random_index] = array_to_mutate'''
		for array in mutation_choices:
			array = mutate(array, self.mutation_prob)

	def train(self, inputs, targets, test_type = True): 
		if test_type:
			if type(inputs) != np.ndarray:
				raise Exception("The inputs is not a Matrix")
			if inputs.shape[0] != self.inputs_number:
				raise Exception("The inputs size doesn't match the neuralnetwork inputs number'")
			if targets.shape[0] != self.outputs_number:
				raise Exception("The targets size doesn't match the neuralnetwork outputs number'")
		
		#passing from input to hidden
		hidden = np.dot(self.weights_IH, inputs)
		hidden += self.bias_H
		hidden = vectorize_sigmoid(hidden)
		
		#hidden to output
		outputs = np.dot(self.weights_HO, hidden)
		outputs += self.bias_O
		outputs = vectorize_sigmoid(outputs)
		
		#/////////////////backpropagation/////////////////
		
		outputs_errors = targets - outputs
		
		#Now I have the error of the outputs, how to use it ?
		#Time for gradient descent (to tweak weights and bias)
		#deltatm = learning rate * x * error (linear regression)                                 
		#deltatb = learning rate * error
		#deltatweights_H = learning rate * x * derivative of sigmoid * H
		
		gradient_O = vectorize_dsigmoid(outputs)
		gradient_O = np.multiply(gradient_O, outputs_errors)
		gradient_O = gradient_O *  self.learning_rate
		#gradient_O calculated
		
		hidden_transposed = hidden.T
		deltat_weights_HO = np.dot(gradient_O, hidden_transposed)
		
		#I adjust the weights
		self.weights_HO += deltat_weights_HO
		#I adjust the bias
		self.bias_O += gradient_O
		
		#hidden_errors = W_HO(transposed) * outputs_errors
		weights_HO_transposed = self.weights_HO.T
		hidden_errors = np.dot(weights_HO_transposed, outputs_errors)

		#deltatweights_I = learning rate * x .(hadamard) derivative of sigmoid *(element) I
		gradient_H = vectorize_dsigmoid(hidden)
		gradient_H = np.multiply(gradient_H, hidden_errors)
		gradient_H = gradient_H * self.learning_rate
		#gradient_H calculated 
		
		inputs_transposed = inputs.T
		deltat_weights_IH = np.dot(gradient_H, inputs_transposed)
		
		#same I adjust the weights and bias
		self.weights_IH += deltat_weights_IH
		self.bias_H += gradient_H
		
		
	
training_inputs = np.array([np.array([[1],[1]]), np.array([[1],[0]]), np.array([[0],[0]]), np.array([[0],[1]])])

training_labels = np.array([[0]]), np.array([[1]]), np.array([[0]]), np.array([[1]])


"""nn = Neural_Network(2, 3, 1)
start = time.time()
for i in range(10000):
	random_index = randrange(0, 3)
	nn.train(training_inputs[random_index], training_labels[random_index])
end = time.time()
time_taken = end - start
print(time_taken)"""
