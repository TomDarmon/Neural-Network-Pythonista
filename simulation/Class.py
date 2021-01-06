from Functions import *
from NeuralNetwork import *
from scene import *
from random import random

############ AGENT CLASS ###########


class agent:
	
	def __init__(self, settings):
		
		self.brain = Neural_Network(1, 4, 4) 

		self.size = 15
		self.speed = randrange(4, 6) + 4 * random()
		self.color = (0, 255, 0)
		
		self.fitness = 0 #fitness = number of food eaten / number of food to eat 
		self.food_eaten = 0
		
		self.x = 0
		self.y = 0
		
		#self.x = np.random.uniform(settings['x_min'], settings['x_max'])
		#self.y = np.random.uniform(settings['y_min'], settings['y_max'])
		
		self.x_ini = self.x
		self.y_ini = self.y
		self.size_ini = self.size

		
	def move(self, next_move):
		
		if next_move == 'UP':
			self.y += self.speed	
			
		if next_move == 'DOWN':
			self.y -= self.speed		
			
		if next_move == 'RIGHT':
			self.x += self.speed	
			
		if next_move == 'LEFT':
			self.x -= self.speed
		
	def display(self):
		stroke(self.color)
		fill(self.color)
		stroke_weight(1)  #Donne une epaisseur au trait
		ellipse(self.x, self.y, self.size, self.size)
		no_stroke()
		no_fill()


	def collision(self, object):
		if np.abs(dist(self, object)) < self.size:
			return True
		else:
			return False

	def grows(self):
		self.size += 5
		
	def refresh(self):
		
		self.size = self.size_ini
		self.color = (0, 255, 0)
		
		self.x = self.x_ini
		self.y = self.y_ini
	
		
		
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
		
		self.x_ini = self.x
		self.y_ini = self.y
		self.size_ini = self.size

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
		
	def refresh(self):
		self.x = self.x_ini
		self.y = self.y_ini
		self.size = self.size_ini

class food_creator:
	
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


class trap:
	
	def __init__(self, settings):
		self.x = np.random.uniform(settings['x_min'], settings['x_max'])
		self.y = np.random.uniform(settings['y_min'], settings['y_max'])
		self.size = 50

	def display(self):
		stroke('orange')
		fill('orange')
		stroke_weight(1)  #Donne une epaisseur au trait
		ellipse(self.x, self.y, self.size, self.size)
		no_stroke()
		
		

def dist(object1, object2):
	return np.sqrt((object1.x - object2.x)**2 + (object1.y - object2.y)**2)
