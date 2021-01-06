
from scene import *
from random import randrange, random, choice
from Functions import *
from Class import *
from settings import *
from time import sleep
from copy import copy
from array_functions import *
from NeuralNetwork import *


	
	
class MyScene(Scene):

	def new_generation_display(self):
		self.generation_number += 1
		self.generation_number_text.remove_from_parent()
		self.generation_number_text = LabelNode(f"generation {self.generation_number}", position = self.size / 2, parent = self)
	
		
		self.agents = reset_agents(self.agents) #does not reset the fitness (we need it)
		
		self.best_agents = find_best_agents(self.agents, 4)
		
		self.childs_brain = cross_over_next_generation_brain(self.best_agents) #make N / 2 childs
		self.childs = generate_agents_with_brain(len(self.childs_brain), self.childs_brain)
		self.next_generation = mutate_next_generation(self.best_agents)
		
		self.agents = self.next_generation + self.best_agents + self.childs
		
		
		
		self.agents = refresh_fitness(self.agents)
		
		
		#self.next_generation = mutate_next_generation(self.best_agents)
		
		self.fresh_agents = generate_agents(settings, 1)
		
		self.agents = self.agents + self.fresh_agents
	
		self.loop = 0
		
		print(len(self.agents))
		
	
		self.foods = generate_foods(settings)
		
		'''
		self.best_agents_generated = regenerate_agents(self.best_agents, 20) #Take the 5 agents and generate more based on the fitness (higher fitness higher number of the agents)
		
		self.next_generation = mutate_next_generation(self.best_agents_generated)
		
		self.fresh_agents = generate_agents2(5)
		
		#self.agents = self.best_agents + self.next_generation + self.fresh_agents
		self.agents = self.next_generation
		'''
		
		
		
		
	"""def same_generation_display(self):
		self.agents = reset_agents(self.agents)
		
		self.foods = self.foods_ini
		self.loop = 0 """
		
	
	def setup(self):
		self.generation_number = 1
		self.generation_number_text = LabelNode(f'Generation {self.generation_number}', position = self.size / 2, parent = self)
		
		self.agents = generate_agents(settings, 2)
		self.foods = generate_foods(settings)	
		self.foods_ini = self.foods #copy of all the food
		
		self.loop = 0
		
	def draw(self):
		self.best_agent = find_best_agents(self.agents, 1) #store the best agent in a list
		self.best_agent = self.best_agent[0]
		
		self.loop += 1
		
		self.agents, self.foods = simulate(settings, self.agents, self.foods, show = True)
				
		if (self.loop > 400):
			self.new_generation_display()
			
	def touch_began(self, touch):
		(x, y) = touch.location
		if (x > 1112 / 2):
			self.new_generation_display()
			
			
run(MyScene(), show_fps = True)

