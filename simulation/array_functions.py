import numpy as np
from random import random, randrange
from math import e
from NeuralNetwork import *

sigmoid = lambda x: 1/(1+pow(e,-x))

vectorize_sigmoid = np.vectorize(sigmoid)
	
dsigmoid = lambda x: x * (1 - x) # not the real derivative because the outputs are already sigmoide
	
vectorize_dsigmoid = np.vectorize(dsigmoid)

def zeros_array(r, c):
  l = []
  for i in range(r):
    l.append([0 for i in range(c)])
  A = np.array(l)
  return A
  

def randomize(A): #buggy
	r = A.shape[0]
	c = A.shape[1]
	for i in range(r):
		for j in range(c):
			a = random()
			A[i][j] = a
			
def array(n, m):
	l = []
	for i in range(n):
		l.append([0 for i in range(m)])
	return np.array(l) 
	

def randomize(A):
		for i in range(A.shape[0]):
			for j in range(A.shape[1]):
				A[i][j] += np.random.rand()
				
def mutate(array_to_mutate, p):
	array = array_to_mutate
	row = array.shape[0]
	col = array.shape[1]
	for i in range(row):
		for j in range(col):
			if random() < p:
				#different type of mutations can happen
				
				kind_of_mutation_index = randrange(1,4)
				
				
				if kind_of_mutation_index == 1: 
					array[i][j] *= random() * 1.5 # multiply by a small number
	
				if kind_of_mutation_index == 2:
					array[i][j] += random() # add a small number
						
				if kind_of_mutation_index == 3: #change the sign
					array[i][j] *= -1
					
	return array





