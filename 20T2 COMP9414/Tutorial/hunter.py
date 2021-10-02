# -*- coding: utf-8 -*-
"""
Created on Fri May 17 (alfred); Sat May 18, 2019 (wobcke)

@author: wobcke (updated from alfred)

    A simple agent hunting for food on a square grid of size grid_size.
    The grid is the upper right quadrant with squares numbered (0..grid_end)
    Food is initially generated in the environment in random positions.
    Food can appear and disappear with some probability on each clock tick.
    The agent is initially positioned at a random location in the environment.
    The agent can only see food within a certain (Euclidean) distance.
    The agent can only move in a horizontal or vertical direction.
    If there is no food visible, the agent moves to the right.
    The objective is to eat as much food as possible.

    Parameters:
        Environment:
            grid_end: end coordinate of grid in X and Y direction, numbered from 0
            grid_size: size of grid, i.e. grid_end + 1
            food_del_chance: probability of a food item disappearing on each clock tick
            food_add_chance: probability of new food item appearing on each clock tick
        Agent:
            horizon: distance that agent can see
"""

from scipy.spatial import distance
import random


class Environment():
    # Env maintains a set of food and computes percepts and effects of agent actions
    # Actions are always successful if feasible
    # Only one food item can be in any location

    grid_end = 9  # Grid numbered from 0 to gridEnd in X and Y directions
    grid_size = grid_end + 1  # Size of grid
    init_food = 5  # Initial number of food items
    food_del_chance = 0.05  # Probabilty of food item disappearing on each cycle
    food_add_chance = 0.20  # Probabilty of food item appearing on each cycle

    def __init__(self):
        self.food = set()
        for i in range(self.init_food):
            X = random.randint(0, self.grid_end)
            Y = random.randint(0, self.grid_end)
            if not ((X, Y) in self.food):
                self.food.add((X, Y))
        print("Food at", end=" ")
        print(self.food)

    # food items in agent's field of view
    def get_percepts(self, agent):
        (agentX, agentY) = agent.location
        return {(X, Y) for (X, Y) in self.food if distance.euclidean((agentX, agentY), (X, Y)) <= agent.horizon}

    # consume food, add 1 to agent's score
    def eat(self, agent):
        if agent.location in self.food:
            self.food.remove(agent.location)
            agent.score += 1

    # movement actions
    def move_left(self, agent):
        (X, Y) = agent.location
        if X > 0:
            return (X - 1, Y)
        else:
            return (X, Y)

    # Movement actions
    def move_right(self, agent):
        (X, Y) = agent.location
        if X < self.grid_end:
            return (X + 1, Y)
        else:
            return (X, Y)

    def move_up(self, agent):
        (X, Y) = agent.location
        if Y < self.grid_end:
            return (X, Y + 1)
        else:
            return (X, Y)

    def move_down(self, agent):
        (X, Y) = agent.location
        if Y > 0:
            return (X, Y - 1)
        else:
            return (X, Y)

    # Environment update function
    def clock_tick(self):
        # possibly delete food items -- copy elements not deleted into new set
        food_set = set()
        for location in self.food:
            r = random.random()
            if (r > self.food_del_chance):
                food_set.add(location)
            else:
                print("Food disappears at", end=" ")
                print(location)
        self.food = food_set
        # possibly add new food item in random location
        r = random.random()
        if (r < self.food_add_chance):
            X = random.randint(0, self.grid_end)
            Y = random.randint(0, self.grid_end)
            if not ((X, Y) in self.food):
                self.food.add((X, Y))
                print("Food appears at", end=" ")
                print((X, Y))


class Agent():
    # Agent maintains set of food items seen...but does not update it very well

    horizon = 10  # the agent can only see food up to this distance

    def __init__(self, env):
        self.env = env
        self.location = (random.randint(0, env.grid_end), random.randint(0, env.grid_end))
        self.food_seen = set()
        self.score = 0
        print("Agent at", end=" ")
        print(self.location)

    def cycle(self):
        # perception
        percepts = self.env.get_percepts(self)
        self.food_seen.update(percepts)  # add new perceived food to set of food seen
        # deliberate mistake: should we delete items that have disappeared?

        # action selection and belief update
        if self.food_seen == set():  # if no food on the list, simply move right
            self.location = self.env.move_right(self)
            print("Agent performs move_right to", end=" ")
            print(self.location)
        else:
            # Move towards nearest location in food_seen according to Manhattan distance
            min_d = min(distance.cityblock(self.location, (X, Y)) for (X, Y) in self.food_seen)
            if min_d == 0:
                self.env.eat(self)
                print("Agent performs eat at", end=" ")
                print(self.location)
                self.food_seen.remove(self.location)
            else:
                best = {(X, Y) for (X, Y) in self.food_seen if distance.cityblock(self.location, (X, Y)) == min_d}
                (agentX, agentY) = self.location
                (X, Y) = best.pop()
                if agentX < X:
                    self.location = self.env.move_right(self)
                    print("Agent performs move_right to", end=" ")
                    print(self.location)
                elif agentX > X:
                    self.location = self.env.move_left(self)
                    print("Agent performs move_left to", end=" ")
                    print(self.location)
                elif agentY < Y:
                    self.location = self.env.move_up(self)
                    print("Agent performs move_up to", end=" ")
                    print(self.location)
                else:
                    self.location = self.env.move_down(self)
                    print("Agent performs move_down to", end=" ")
                    print(self.location)


class Simulator():
    # Simulate environment update and agent perception-action cycle

    def __init__(self):
        self.env = Environment()
        self.agent = Agent(self.env)

    def go(self, n):
        for i in range(n):
            self.env.clock_tick()
            self.agent.cycle()
        print("Agent score was %s" % self.agent.score)


simulator = Simulator()
simulator.go(20)