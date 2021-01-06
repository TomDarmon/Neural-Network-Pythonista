
from random import randrange, random

class Matrix:
	
	def __init__(self, matrix = []):
		self.matrix = matrix
		if self.matrix != []:
			self.r = len(self.matrix)
			self.c = len(self.matrix[0])
		else:
			self.matrix = []
			self.r = 0
			self.c = 0
			
	def toList(self):
		if self.r == 1:
			liste = [self.matrix[i] for i in range(self.r)]
			return liste[0]			
		else:
			liste = []
			new_l = []
			for i in range(self.r):
				liste.append(self.matrix[i])
			for el in liste:
				new_l += el
			return new_l
				
	def mutate(self, prob):
		for i in range(self.r):
			for j in range(self.c):
				if random() < prob:
					#different type of mutations can happen
					kind_of_mutation_index = randrange(1,4)
					
					if kind_of_mutation_index == 1: 
						self.matrix[i][j] *= random() * 1.5 # multiply by a small number
	
					if kind_of_mutation_index == 2:
						self.matrix[i][j] += random() # add a small number
						
					if kind_of_mutation_index == 3: #change the sign
						self.matrix[i][j] *= -1
					
	
		
						
		
	def max_index(self): #FIND THE INDEX OF THE MAX IN A VECTOR
		if self.c != 1:
			raise Exception("The Matrix must be a vector to find the max index (self.c > 1)")
		max = self.matrix[0][0]
		max_index = 0
		for i in range(self.r):
			if self.matrix[i][0] > max:
				max = self.matrix[i][0]
				max_index = i
		return max_index
		
		
	def show(self):
		i = 0
		while i < self.r:
			print(self.matrix[i])
			i += 1
	
	def zeros(self, r, c):
		l = []
		for i in range(r):
			l.append([0 for i in range(c)])
		self.r = r
		self.c = c
		self.matrix = l
			
	def randomize(self, r, c): #buggy
		self.zeros(r,c)
		for i in range(r):
			for j in range(c):
				self.matrix[i][j] = random() 
			

#ADD 2 matrix, 2 numbers (positive and negative)

	def add(self, B):
		if type(B) == Matrix:
			C = Matrix() #result matrix
			C.zeros(self.r, self.c) 
			for i in range(self.r):
				for j in range(self.c):
					C.matrix[i][j] = self.matrix[i][j] + B.matrix[i][j]
			return C
		elif (type(B) == int) or (type(B) == float):
			C = Matrix()
			C.zeros(self.r, self.c)
			for i in range(self.r):
				for j in range(self.c):
					C.matrix[i][j] = self.matrix[i][j] + B
			return C
		else:
			raise Exception("The object is not a matrix or an int")
			
			
#Only substract MATRIX !			

	def substract(self, B):
		if type(B) == Matrix:
			C = Matrix() #result matrix
			C.zeros(self.c, self.r) 
			for i in range(self.r):
				for j in range(self.c):
					C.matrix[i][j] = self.matrix[i][j] - B.matrix[i][j]
			return C
		else:
			raise Exception("The object is not a matrix or an int")
			
	def hadamard_mult(self, B):
		if self.c == B.c and self.r == B.r:
			result = Matrix()
			result.zeros(B.r, B.c)
			for i in range(B.r):
				for j in range(B.c):
					result.matrix[i][j] = self.matrix[i][j] * B.matrix[i][j]
			return result
		else:
			raise Exception("The 2 matrix are not the same size")
				
	def multiply(self, B):
		if isinstance(B, Matrix):
			if self.c != B.r:
				raise Exception("Dimensions don't match'")
			C = Matrix()
			C.zeros(self.r, B.c)
			for i in range(self.r):
				for j in range(B.c):
					for k in range(B.r):
						C.matrix[i][j] += self.matrix[i][k] * B.matrix[k][j]
			return C
		if (isinstance(B, int)) or (isinstance(B, float)):
			C = Matrix()
			C.zeros(self.r, self.c)
			for i in range(self.r):
				for j in range(self.c):
					C.matrix[i][j] = self.matrix[i][j] * B
			return C
		else:
			raise Exception("can only multiply int, float or matrix")
			
	def map(self, function): 
		C = Matrix()
		C.zeros(self.r, self.c)
		for i in range(self.r):
			for j in range(self.c):
				C.matrix[i][j] = function(self.matrix[i][j])
		return C
			
	def map_static(self, function):
		for i in range(self.r):
			for j in range(self.c):
				self.matrix[i][j] = function(self.matrix[i][j])
		
			
	def transpose(self):
		C = Matrix()
		C.zeros(self.c, self.r) #we change the number of raw with the number of cols
		for i in range(self.r):
			for j in range(self.c):
				C.matrix[j][i] = self.matrix[i][j]
		return C
		
	def max(self):
		max = None
		for i in range(self.r):
			for j in range(self.c):
				if self.matrix[i][j] > max:
					max = self.matrix[i][j]
		return max
