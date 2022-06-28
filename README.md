# Personnal Project - Self Driving Car

This project aims to build a self driving car in a specific area.

PyGame is used to build the track and command the car

2 techniques are implemented:
 - **Reinforcement learning**:
    - Neural network is trained to drive the car with Deep Q-learning
    - Positive results on the simplest track after a really long training (more than 1 night)

- **Genetic algorithm**:
    - Neural network is trained to drive the car with Genetic Algorithm
    - Positive results on the simplest track after a small time

### Note 
This project is currently under development.

### How to use this project

To run the project with Genetic Algorithm, you need to run the following command:

```python
python3 Environment_Genetic.py

```

To run the project with Reinforcement learning, you need to run the following command:

```python
python3 Environment_DQN.py
```

Specific requirements need to be installed before (pygame, pytorch basically)

Paul Theron