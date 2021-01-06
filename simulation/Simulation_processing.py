import numpy as np
from random import randrange, random
from random import random, randrange, choice
from math import e

#####SETTIGNS#####

settings = dict()
settings['pop_size'] = 50  # number of agents to spawn
settings['food_quant'] = 40
settings['color'] = (0, 255, 0)
settings['x_min'] = 0  # left side of the arena
settings['x_max'] = 1100  # right side of the arena
settings['y_min'] = 0  # lower bound of the arena
settings['y_max'] = 850  # upper bound of the arena
settings['traps_quant'] = 0
settings['bad_agents_quant'] = 0
settings['food_creators_quant'] = 0
settings['mutation_rate'] = 0.1

############ALL THE DRAWING############

def new_generation(self):
	self.generation_number += 1
	self.generation_number_text.remove_from_parent()
	self.generation_number_text = LabelNode(
		f"generation {self.generation_number}", position=self.size / 2, parent=self)

	#self.agents = choose_next_generation(self.old_agents)
	self.agents = generate_agents(settings)
	self.foods = generate_foods(settings)

	##selection need to be done here##

def setup(self):
	self.generation_number = 1
	self.generation_number_text = LabelNode(f'Generation {self.generation_number}', position=self.size / 2, parent=self)
	self.agents = generate_agents(settings)
	self.foods = generate_foods(settings)

	self.old_foods = self.foods
	self.old_agents = []

def draw(self):
	self.agents, self.old_agents, self.foods = simulate(settings, self.agents, self.foods, self.old_agents)
	for agent in self.agents:
		agent.frame_alive += 1


#####NEURALNETWORK LIB######

#The matrix lib is used this way : A = Matrix([[1,2,3],[4,5,6]]) 2 x 3 matrix 


def sigmoid(x):
	return 1 / (1 + pow(e, -x))


def dsigmoid(
		x):  # not the real derivative because the outputs are already sigmoided
	return x * (1 - x)


class Neural_Network:
	def __init__(self, inputs, hidden, outputs):
		self.inputs_number = inputs
		self.hidden_number = hidden
		self.outputs_number = outputs

		self.weights_IH = Matrix()
		self.weights_IH.randomize(self.hidden_number, self.inputs_number)

		self.bias_H = Matrix()
		self.bias_H.randomize(self.hidden_number, 1)

		self.weights_HO = Matrix()
		self.weights_HO.randomize(self.outputs_number, self.hidden_number)

		self.bias_O = Matrix()
		self.bias_O.randomize(self.outputs_number, 1)

		self.learning_rate = 0.1

	def settings(self):
		print("Weights_IH : ")
		print(self.weights_IH.show())

		print("Bias_H : ")
		print(self.bias_H.show())

		print("Weights_IH : ")
		print(self.weights_IH.show())

		print("Bias_O : ")
		print(self.bias_O.show())

	def predict(self, inputs):
		if type(inputs) != Matrix:
			raise Exception("The inputs is not a Matrix")
		if inputs.r != self.inputs_number:
			raise Exception(
				"The inputs size doesn't match the neuralnetwork inputs number'")

		#passing from input to hidden
		hidden = Matrix.multiply(self.weights_IH, inputs)
		hidden.add(self.bias_H)
		hidden.map_static(sigmoid)
		#now hidden is calculated

		#hidden to output
		outputs = Matrix.multiply(self.weights_HO, hidden)
		outputs.add(self.bias_O)
		outputs.map_static(sigmoid)
		#calculated

		return outputs

	def mutate(self):
		mutation_choices = [
			self.weights_IH, self.bias_H, self.bias_O, self.weights_HO
		]
		random_index = randrange(0, 4)
		matrix_to_mutate = mutation_choices[random_index]

		# I want to change a number between one and the total number of parmaeters in the matrix (raws * cols)
		mutation_choices[random_index] = matrix_to_mutate.mutate(prob=0.1)

	def train(self, inputs, targets):

		if type(inputs) != Matrix:
			raise Exception("The inputs is not a Matrix")
		if inputs.r != self.inputs_number:
			raise Exception(
				"The inputs size doesn't match the neuralnetwork inputs number'")
		if targets.r != self.outputs_number:
			raise Exception(
				"The targets size doesn't match the neuralnetwork outputs number'")

		#passing from input to hidden
		hidden = Matrix.multiply(self.weights_IH, inputs)
		hidden.add(self.bias_H)
		hidden.map_static(sigmoid)

		#hidden to output
		outputs = Matrix.multiply(self.weights_HO, hidden)
		outputs.add(self.bias_O)
		outputs.map_static(sigmoid)

		#/////////////////backpropagation/////////////////

		outputs_errors = Matrix.substract(targets, outputs)
		#Now I have the error of the outputs, how to use it ?
		#Time for gradient descent (to tweak weights and bias)
		#deltatm = learning rate * x * error (linear regression)                                 
		#deltatb = learning rate * error
		#deltatweights_H = learning rate * x * derivative of sigmoid * H

		gradient_O = Matrix.map(outputs, dsigmoid)
		gradient_O = Matrix.hadamard_mult(gradient_O, outputs_errors)
		gradient_O = Matrix.multiply(gradient_O, self.learning_rate)
		#gradient_O calculated

		hidden_transposed = hidden.transpose()
		deltat_weights_HO = Matrix.multiply(gradient_O, hidden_transposed)

		#I adjust the weights
		self.weights_HO = self.weights_HO.add(deltat_weights_HO)
		#I adjust the bias
		self.bias_O = self.bias_O.add(gradient_O)

		#hidden_errors = W_HO(transposed) * outputs_errors
		weights_HO_transposed = self.weights_HO.transpose()
		hidden_errors = Matrix.multiply(weights_HO_transposed, outputs_errors)

		#deltatweights_I = learning rate * x .(hadamard) derivative of sigmoid *(element) I
		gradient_H = Matrix.map(hidden, dsigmoid)
		gradient_H = Matrix.hadamard_mult(gradient_H, hidden_errors)
		gradient_H = Matrix.multiply(gradient_H, self.learning_rate)
		#gradient_H calculated 

		inputs_transposed = inputs.transpose()
		deltat_weights_IH = Matrix.multiply(gradient_H, inputs_transposed)

		#same I adjust the weights and bias
		self.weights_IH = self.weights_IH.add(deltat_weights_IH)
		self.bias_H = self.bias_H.add(gradient_H)
		
		"""
training_inputs = [Matrix([[1],[1]]), Matrix([[1],[0]]), Matrix([[0],[0]]), Matrix([[0],[1]])]

training_labels = [Matrix([[0]]), Matrix([[1]]), Matrix([[0]]), Matrix([[1]])]

	nn = Neural_Network(2,3,1)
for i in range(50000):
	next_data_index = randrange(len(training_inputs))
	next_input = training_inputs[next_data_index]
	next_label = training_labels[next_data_index]
	nn.train(next_input, next_label)

	"""


####MATRIX LIB####


class Matrix:
	def __init__(self, matrix=[]):
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
					kind_of_mutation_index = randrange(1, 5)

					if kind_of_mutation_index == 1:
						self.matrix[i][j] *= random() * 1.5  # multiply by a small number

					if kind_of_mutation_index == 1:
						self.matrix[i][j] += random()  # add a small number

					if kind_of_mutation_index == 1:  #change the sign
						self.matrix[i][j] *= -1

					self.matrix[i][j] *= random()

	def max_index(self):  #FIND THE INDEX OF THE MAX IN A VECTOR
		if self.c != 1:
			raise Exception(
				"The Matrix must be a vector to find the max index (self.c > 1)")
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

	def randomize(self, r, c):  #buggy
		self.zeros(r, c)
		for i in range(r):
			for j in range(c):
				self.matrix[i][j] = random()

#ADD 2 matrix, 2 numbers (positive and negative)	

	def add(self, B):
		if type(B) == Matrix:
			C = Matrix()  #result matrix
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
			C = Matrix()  #result matrix
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
		C.zeros(self.c, self.r)  #we change the number of raw with the number of cols
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


############END OF MATRIX LIB#######

############ Class And Functions for the simulation ###########


class agent():
	
	def __init__(self, settings):

		self.brain = Neural_Network(3, randrange(6, 20), 4)

		self.size = 15
		self.speed = 2
		self.color = (0, 255, 0)
		self.fitness = 0

		self.x = np.random.uniform(settings['x_min'], settings['x_max'])
		self.y = np.random.uniform(settings['y_min'], settings['y_max'])
		self.x_ini = self.x
		self.y_ini = self.y
		self.frame_alive = 0
		self.food_eaten = 0

		self.directions = []
		self.number_of_direction = 0

	def move(self, next_move):

		if next_move == 'UP':
			self.y += self.speed
			self.directions.append('UP')

		if next_move == 'DOWN':
			self.y -= self.speed
			self.directions.append('UP')

		if next_move == 'RIGHT':
			self.x += self.speed
			self.directions.append('UP')

		if next_move == 'LEFT':
			self.x -= self.speed
			self.directions.append('UP')

	def display(self):
		stroke(self.color)
		fill(self.color)
		stroke_weight(1)  #Donne une epaisseur au trait
		ellipse(self.x, self.y, self.size, self.size)
		no_stroke()
		no_fill()

	def collision(self, object):
		if np.abs(dist(self, object)) < object.size:
			return True
		else:
			return False

	def grows(self):
		self.size += 5

		############ Bad_aggent CLASS herits from agent ###########


"""class bad_agent(agent):
	def __init__(self, settings):
		self.x = np.random.uniform(settings['x_min'], settings['x_max'])
		self.y = np.random.uniform(settings['y_min'], settings['y_max'])
		self.size = 25
		self.speed = 0.1
		self.direction_x = np.random.uniform(randrange(-10, 10), randrange(-10, 10))
		self.direction_y = np.random.uniform(randrange(-10, 10), randrange(-10, 10))
		self.color = (255, 0, 0)
"""

############ FOOD CLASS ###########


class food():
	def __init__(self, settings, position=None):

		if (position == None):
			self.x = np.random.uniform(settings['x_min'], settings['x_max'])
			self.y = np.random.uniform(settings['y_min'], settings['y_max'])
			self.food_created = 10
		else:
			self.x = position[0]
			self.y = position[1]

		self.size = 10
		self.speed = 0.1

	def display(self):
		stroke(255, 0, 255)
		fill(255, 0, 255)
		stroke_weight(1)  #Donne une epaisseur au trait
		ellipse(self.x, self.y, self.size, self.size)
		no_stroke()

	def move(self):
		self.x += np.random.uniform(randrange(-10, 10), randrange(-10, 10))
		self.y += np.random.uniform(randrange(-10, 10), randrange(-10, 10))
		self.speed = self.speed * 0.5


class food_creator():
	def __init__(self, settings):
		self.x = np.random.uniform(settings['x_min'], settings['x_max'])
		self.y = np.random.uniform(settings['y_min'], settings['y_max'])
		self.food_created = 10

	def display(self):
		stroke('brown')
		fill('brown')
		stroke_weight(1)  #Donne une epaisseur au trait
		Rect(self.x, self.y, self.size, self.size)
		no_stroke()

	def create_food(self):
		new_food = []
		for i in range(self.food_created):
			new_food.append(food(settings, (self.x, self.y)))
		return new_food

		############ TRAP CLASS ###########


class trap():
	def __init__(self, settings):
		self.x = np.random.uniform(settings['x_min'], settings['x_max'])
		self.y = np.random.uniform(settings['y_min'], settings['y_max'])
		self.size = 50

	def display(self):
		stroke(255,0,0)
		fill(255,0,0)
		stroke_weight(1)  #Donne une epaisseur au trait
		ellipse(self.x, self.y, self.size, self.size)
		no_stroke()


def agent_in_screen(agents, old_agents):
	for agent in agents:
		if (0 < agent.x < 1100) and (0 < agent.y < 820):
			old_agents.append(agent)
	return agents, old_agents


def fitness(agents):
	for agent in agents:
		agent.UP_number = agent.directions.count("UP")
		agent.DOWN_number = agent.directions.count("DOWN")
		agent.RIGHT_number = agent.directions.count("RIGHT")
		agent.LEFT_number = agent.directions.count("LEFT")

		if agent.UP_number != 0:
			agent.number_of_direction += 1
		if agent.DOWN_number != 0:
			agent.number_of_direction += 1
		if agent.RIGHT_number != 0:
			agent.number_of_direction += 1
		if agent.LEFT_number != 0:
			agent.number_of_direction += 1

		###FITNESS FORMULA### <==== VERY BAD ATM

		agent.fitness = agent.number_of_direction * 1.5 + agent.food_eaten * 0.5 + 0.0005 * agent.frame_alive


def normalize(x, max, min):
	normalized = (x - min) / (max - min)
	return normalized


def think_next_movement(agent, foods):
	# I'm looking for the closest food then I find the distance 
	closest_food = foods[0]
	for food in foods:
		if dist(agent, food) < dist(agent, closest_food):
			closest_food = food
	dist_closest_food = dist(agent, closest_food)

	x_position = agent.x
	y_position = agent.y

	normalized_x_position = normalize(x_position, 1112, 0)
	normalized_y_position = normalize(y_position, 834, 0)
	normalized_distance_of_food = normalize(dist_closest_food, 1112, 0)

	prediction = agent.brain.predict(Matrix([[normalized_distance_of_food], [normalized_x_position], [normalized_y_position]]))  
	#Take as an input the dist between the agent and the closest food

	prediction_index = prediction.max_index()

	if prediction_index == 0:
		return 'UP'

	if prediction_index == 1:
		return 'DOWN'

	if prediction_index == 2:
		return 'RIGHT'

	if prediction_index == 3:
		return 'LEFT'


def display(agents, foods):
	for agent in agents:
		agent.display()
	for food in foods:
		food.display()


def generate_bad_agents(settings):
	bad_agents = []
	for i in range(settings['bad_agents_quant']):
		bad_agents.append(bad_agent(settings))
	return bad_agents


def generate_agents(settings):
	agents = []
	for i in range(settings['pop_size']):
		agents.append(agent(settings))
	return agents


def generate_foods(settings):
	foods = []
	for i in range(settings['food_quant']):
		foods.append(food(settings))
	return foods


def generate_traps(settings):
	traps = []
	for i in range(settings['traps_quant']):
		traps.append(trap(settings))
	return traps


def generate_food_creators(settings):
	food_creators = []
	for i in range(settings['food_creators_quant']):
		food_creators.append(food_creator(settings))
	return food_creators


def dist(object1, object2):
	return np.sqrt((object1.x - object2.x)**2 + (object1.y - object2.y)**2)


def choose_next_generation(old_agents):
	new_agents = []
	for agent in old_agents:
		if agent.fitness > 1:
			new_agents.append(agent)
	return new_agents


def simulate(settings, agents, foods, old_agents):

	for agent in agents:
		next_move = think_next_movement(agent, foods)  #Store the next move as a string
		agent.move(next_move)
		for food in foods:
			if agent.collision(food):
				agent.grows()
				agent.food_eaten += 1
		foods = [food for food in foods if not agent.collision(food)]  #we delete every food if it has been eate

	agents, old_agents = agent_in_screen(agents, old_agents)  #Suppress the agents out of the screen																	
	display(agents, foods)
	fitness(agents)

	return agents, old_agents, foods

