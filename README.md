# Neural-Network-Pythonista and NuroevElution game
This is a toy neural implementation that works in the Pythonista environnement on IPAD. In addition a neuroevolution project was created on top of the toy NeuralNetwork library.

This is a simulation wrote from scratch (only using a drawing library from Pythonista called Scene), the game is a similar game to agar.io where agent are moving in 2D to eat food in order to grow. The goal is to create a neuroevolution algorithm in order for each agent to have a brain that will improve at each iteration.

This code was written as learning project in end of highschool / 1 st year of undergraduate school. Some of the implementations are terribly slow, some of them are not using python tools properly, please forgive me for that. 

[![IMG-1002.png](https://i.postimg.cc/Kcr5rhTT/IMG-1002.png)](https://postimg.cc/rzsWMbQy)

<b>matrix</b> : This is a toy library for linear algebra, it's written in Python (really slow). It has only functions to be able to make a neuralnetwork with backpropagation such as multiplication, addition, substraction, dot product, hadamard product and transposition. In addition the library has additional functions used to implement a genetic algorithm such as mutate the matrix. (Matrix LIBRARY IS NOT USED ANYMORE)

<b>Functions</b> : This code has general functions functions to make the simulation work (create agents, display them, simulate them) it needs to be refactored, it's not organized. In addition this code has silent_simulation function that run the simulation without any drawing (faster but no visualization).

<b>Class</b> : This is an implementation of all the class of object used in the small game (agents, food, additional classes that were not implemented properly).

<b>array_functions</b> : Some functions needed to be executed outside the Matrix object, mutate for example. Other functions were just bad design from myself and should have been implemented in the matrix class.

<b>NeuralNetwork</b> : This is a simple NN with onyl 1 hidden layer, with a backpropagation implementation in the train method. Overall it passed the test of the XOR problem so it can solve non linear problems. The first implementation was done using the Matrix library, then when it worked I refactored it with numpy so it can run faster. Still this is a toy implementation as the training is done in a python loop. (Matrix LIBRARY IS NOT USED ANYMORE)

<b>Simulation_main</b> : This code is where the magic happens ! The game is runned with different agents each with a fitness function (starting at 0). Each time an agent eats food the fitness increases, more fitness => more chances of getting selected for next generation. The agents all start at the bottom left corner in order to learn to move diagonally before gaining fitness (if they strated at the center randomly then eating food would be luck).
 
 
<b>settings</b> : This is the parameter file of the simulation with the number of food, of agents, etc... The project was runned on Ipad (as the Scene library is only in pythonista), therefore the computing power is small and the simulation can only run properly (graphically) with less than 30 agents. 


