#!python3
# ==================================================
# Computational Society for Bocconi Students project
# ==================================================
#
# The goal is to simulate a population of agents and their interactions over
# time.

from scene import *
from random import randrange, random
from Functions import *
from Class import *
from settings import *
from time import sleep
from copy import copy
from array_functions import *
	
	
#best = agents = do_n_simulations(settings, agent, 50)
	
class MyScene(Scene):

	def new_generation_display(self):
		self.generation_number += 1
		self.generation_number_text.remove_from_parent()
		self.generation_number_text = LabelNode(f"generation {self.generation_number}", position = self.size / 2, parent = self)
		
		
		self.best_agents = find_best_agents(self.agents)
		self.best_agents = 10 * self.best_agents
		self.agents = mutate_next_generation(self.best_agents)
		self.agents = reset_agents(self.agents, self.old_agents)
		#self.agents = color_in_red(self.agents, 1)
		
		self.foods = generate_foods(settings)
		self.old_agents = []
		self.loop = 0
		
	def same_generation_display(self):
		self.agents = reset_agents(self.agents, self.old_agents)
		self.foods = self.foods_ini
		self.old_agents = []
		self.loop = 0
		
	
	def setup(self):
		self.generation_number = 1
		self.generation_number_text = LabelNode(f'Generation {self.generation_number}', position = self.size / 2, parent = self)
		
		self.agents = generate_agents(settings, 1)
		self.foods = generate_foods(settings)	
		self.foods_ini = self.foods #copy of all the food
		
		self.old_agents = []
		self.best_agents = [] #no best agents since gen 1
		self.loop = 0
		
	def draw(self):
		self.loop += 1
		
		self.agents, self.old_agents, self.foods = simulate(settings, self.agents, self.foods, self.old_agents, show = True)
		
		if (len(self.agents) == 0) or (self.loop > 600):
			self.agents = []
			self.new_generation_display()
		
	def touch_began(self, touch):
		(x, y) = touch.location
		if (x > 1112 / 2):
			self.old_agents += self.agents
			self.agents = []
			self.new_generation_display()
		else:
			self.same_generation_display()
		
run(MyScene(), show_fps = True)

