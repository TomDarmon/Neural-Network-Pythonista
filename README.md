# Neural-Network-Pythonista and NeuroEvolution Game

This project is an implementation that works in the Pythonista environment on iPad. It includes a neuroevolution project built on top of a toy NeuralNetwork library.

The simulation is written from scratch, utilizing a drawing library called Scene from Pythonista. The game is similar to agar.io, where agents move in a 2D space to eat food and grow. The goal is to create a neuroevolution algorithm in order for each agent to have a brain that improves at each iteration.

[![IMG-1002.png](https://i.postimg.cc/Kcr5rhTT/IMG-1002.png)](https://postimg.cc/rzsWMbQy)

## Matrix
This is a toy library for linear algebra, written in Python (which is slow). It provides functions for creating a neural network with backpropagation, such as multiplication, addition, subtraction, dot product, Hadamard product, and transposition. Additionally, the library includes functions used for implementing a genetic algorithm, such as matrix mutation. (Matrix library is not used anymore)

## Functions
This code contains general functions to make the simulation work, such as creating agents, displaying them, and simulating their behavior. However, it needs to be refactored as it is currently not well-organized. Additionally, there is a `silent_simulation` function that runs the simulation without any drawing, making it faster but without visualization.

## Class
This section includes the implementation of all the classes of objects used in the game, such as agents, food, and other additional classes that were not properly implemented.

## array_functions
Some functions needed to be executed outside the Matrix object, such as mutation. Other functions were poorly designed and should have been implemented within the matrix class.

## NeuralNetwork
This is a simple neural network with only one hidden layer. It includes a backpropagation implementation in the `train` method. The network has passed the XOR problem test, demonstrating its ability to solve non-linear problems. The initial implementation used the Matrix library, but it was later refactored with numpy for improved performance. However, the training is still done in a Python loop. (Matrix library is not used anymore)

## Simulation_main
This code is where the magic happens! The game is run with different agents, each having a fitness function that starts at 0. Each time an agent eats food, its fitness increases. Higher fitness increases the chances of being selected for the next generation. The agents all start at the bottom left corner to learn to move diagonally before gaining fitness. If they started randomly at the center, eating food would be based on luck.

## settings
This file contains the parameters for the simulation, such as the number of food items and agents. The project was run on an iPad using Pythonista, which has limited computing power. Therefore, the simulation can only run properly (graphically) with fewer than 30 agents.
