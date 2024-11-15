import numpy as np
import gym
from utils import discretize_action_space

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from math import *
import random

c = 1.0
action_bins=5
GAME_ACTIONS= action_bins*2

dis_action_space = discretize_action_space(action_bins=action_bins, action_min= -.078, action_max = .078)

def cus_copy(game):

    new_game = gym.make('penguin_env-v0')
    new_game.reset()
    new_game.set_states(game.model,game.data)

    return new_game

class Node:

    def __init__(self, game, done, parent, observation, action_index):
          
        # child nodes
        self.child = None
        
        # total rewards from MCTS exploration
        self.T = 0
        
        # visit count
        self.N = 0        
                
        # the environment
        self.game = game
        
        # observation of the environment
        self.observation = observation
        
        # if game is won/loss/draw
        self.done = done

        # link to parent node
        self.parent = parent
        
        # action index that leads to this node
        self.action_index = action_index
        
        
    def getUCBscore(self):

        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return float('inf')
        
        # We need the parent node of the current node 
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
            
        # We use one of the possible MCTS formula for calculating the node value
        return (self.T / self.N) + c * sqrt(log(top_node.N) / self.N)
    
    
    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None
       
        
    def create_child(self):

        
        if self.done:
            return
    
        actions = []
        games = []
        for dis_action in range(GAME_ACTIONS): 
            # dis_action= dis_action_space.get_action_in_bin(i)
            actions.append(dis_action)           
            new_game = cus_copy(self.game)
            games.append(new_game)
            
        child = {} 
        for action, game in zip(actions, games):

            dis_action= dis_action_space.get_action_in_bin(action)
            observation, reward, done, _ = game.step(dis_action)

            child[action] = Node(game, done, self, observation, action)                        
            
        self.child = child
                
            
    def explore(self):
        print("exploring...")

        # find a leaf node by choosing nodes with max U.
        
        current = self
        
        while current.child:

            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [ a for a,c in child.items() if c.getUCBscore() == max_U ]
            if len(actions) == 0:
                print("error zero length ", max_U)                      
            action = random.choice(actions)
            current = child[action]
            
        # play a random game, or expand if needed          
            
        if current.N < 1:
            current.T = current.T + current.rollout()
        else:
            print("current.create_child()")
            current.create_child()
            if current.child:
                current = current.child[random.choice(list(current.child))]
            current.T = current.T + current.rollout()
                
        current.N += 1      
                
        # update statistics and backpropagate
        parent = current
            
        while parent.parent:
            
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T 

        print("Done exploring!")          
            
            
    def rollout(self):
        print("performing rollouts...")
        
        if self.done:
            return 0        
        
        v = 0
        done = False
        new_game = cus_copy(self.game)
        while not done:
            action = new_game.action_space.sample()
            action= dis_action_space.get_discretized_action(action)
            observation, reward, done, _ = new_game.step(action)
            v = v + reward
            if done:
                new_game.reset()
                new_game.close()
                break             
        return v

    
    def next(self):

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')
        
        child = self.child
        
        max_N = max(node.N for node in child.values())
       
        max_children = [ c for a,c in child.items() if c.N == max_N ]
        
        if len(max_children) == 0:
            print("error zero length ", max_N) 
            
        max_child = random.choice(max_children)
        
        return max_child, max_child.action_index


def Policy_Player_MCTS(mytree, MCTS_POLICY_EXPLORE=20):  

    '''
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the best possible next action
    '''
    
    for i in range(MCTS_POLICY_EXPLORE):
        print("In loop..",i)
        mytree.explore()
    

    next_tree, next_action = mytree.next()
    print(next_action)
        
    # note that here we are detaching the current node and returning the sub-tree 
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    next_tree.detach_parent()
    
    return next_tree, next_action