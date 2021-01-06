from Class import *
from NeuralNetwork import *

def agent_in_screen(agents, old_agents):
	for agent in agents:
		if not ( -30 < agent.x < 1150): 
			old_agents.append(agent)
			agents.pop(agents.index(agent))
		if not (-30 < agent.y < 850):
			old_agents.append(agent)
			agents.pop(agents.index(agent))
	return agents, old_agents

def fitness(settings, agents):
	for agent in agents:
		agent.fitness = agent.food_eaten / settings['food_quant']
	return agents
		
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
	dist_to_border_x = 1112 - agent.x
	dist_to_border_y = 834 - agent.y
	
	normalized_x_dist = normalize(dist_to_border_x, 1112, 0)
	normalized_y_dist = normalize(dist_to_border_y, 834, 0)
	normalized_distance_of_food = normalize(dist_closest_food, 1112, 0)
	
	prediction = agent.brain.predict(np.array([[normalized_distance_of_food], [normalized_x_dist], [normalized_y_dist]])) #Take as an input the dist between the agent and the closest food
	
	prediction_max = max(prediction)
	for i in range(len(prediction)):
		if prediction[i] == prediction_max:
			prediction_index = i
			break
	
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


def generate_agents(settings, pop_size):
	if pop_size == 1:
		agents = []
		for i in range(settings['pop_size']):
			agents.append(agent(settings))
		return agents
	elif pop_size == 2:
		agents = []
		for i in range(settings['pop_size 2']):
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
	
def simulate(settings, agents, foods, old_agents, show):

	agents = fitness(settings, agents)
	
	for agent in agents:
		next_move = think_next_movement(agent, foods) #Store the next move as a string
		agent.move(next_move)
		for food in foods:
			if agent.collision(food):
				agent.grows()
				agent.food_eaten += 1
				
		foods = [food for food in foods if not agent.collision(food)]  #we delete every food if it has been eaten
											
	agents, old_agents = agent_in_screen(agents, old_agents) #Suppress the agents out of the screen and put them in old_agents
	
	if show == True:														
		display(agents, foods)
	return agents, old_agents, foods
	
	
####### FUNCTION TO SIMULATE SILENTLY #######

def silent_simulation(agent, settings):
	old_agent = [] #list containing the agent at the end of the simulation 
	weighted_fitness = 0 #We are going to give this weighted_fitness to the agent at the end of the simulation 
	number_of_loops = 0 #In case the agent is block to break the loop
	
	for i in range(settings['number_of_test']): #first loop to repeat the test n times and have a weighted fitness
		foods = generate_foods(settings) #new food at each new test
		while True: # simulation loop (1 simulation at a time)
			number_of_loops += 1
			agent, old_agent, foods = simulate(settings, agent, foods, old_agent, show = False)
			
			
	
			if not ((len(old_agent)) == 0): #
				weighted_fitness += old_agent[0].fitness / (settings['food_quant'] * settings['number_of_test'])
				old_agent[0].weighted_fitness = weighted_fitness
				return old_agent[0]
				
			if number_of_loops > 1500: #If there is too much loops
				print('too much loops')
				agent_to_test[0].weighted_fitness = 0
				return agent_to_test[0]
	
	old_agent[0].weighted_fitness = weighted_fitness
		
		
def do_n_simulations(settings, best_agents, n):
	total_agents = []
	for i in range(n):
		agent = best_agents[i]
		agent_simulated = silent_simulation(agent, settings) #we calculated the weighted fitness for 1 agent
		total_agents.append(agent_simulated) #all the agent simulated
		print(f' # {i} Weighted Fitness : {agent_simulated.weighted_fitness}')
	
	best_agents = find_best_agents(total_agents, best_agents)
	return best_agents
	
def find_best_agents(total_agents):
	best_agents = total_agents[:10]
	for agent in total_agents: #we compare all the agent simulated
		for best_agent in best_agents: #with the best 10 agents
			if agent.weighted_fitness > best_agent.weighted_fitness:
				agent_to_add = agent
				agent_to_delete = best_agent
				best_agents.append(agent_to_add)
				best_agents.remove(agent_to_delete)
				
				break # we break the loop to add this agent only 1 time
	return best_agents
	
	
def reset_agents(agents, old_agents = []):
	agents = old_agents + agents
	for agent in agents:
		agent.refresh()
	return agents
	
		
def mutate_next_generation(best_agents):
	for agent in best_agents:
		agent.brain.mutate()
	return best_agents
	
def next_generation(agents):
	best_agents = do_n_simulations(settings)
	return best_agents
	
	
def color_in_red(agents, n):
	i = 0
	for agent in agents:
		agent.color = (255, 0, 0)
		if i > n:
			break
	return agents
		
