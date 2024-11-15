import time
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

class Q_critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,400)
        self.fc2 = nn.Linear(401,300)
        self.fc3 = nn.Linear(300,1)
    
    def forward(self,xs):
        x, u = xs
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(torch.cat([out,u],dim=1)))
        out = self.fc3(out)
        return out

class policy_actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,400)
        self.fc2 = nn.Linear(400,300)
        self.fc3 = nn.Linear(300,1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Replay_Buffer():
    def __init__(self, mem_size, sample_size):
        self.mem_size = mem_size
        self.sample_size = sample_size
        self.states = [[] for m in range(mem_size)]
        self.actions = [[] for m in range(mem_size)]
        self.costs = [[] for m in range(mem_size)]
        self.next_states = [[] for m in range(mem_size)]
        self.index = 0
        self.is_full = False
    
    def store_data(self, state, action, cost, next_state):
        if self.index == self.mem_size:
            self.index = 0
            self.is_full = True
        self.states[self.index] = state
        self.actions[self.index] = action
        self.costs[self.index] = cost
        self.next_states[self.index] = next_state
        self.index += 1
    
    def sample_data(self):
        if self.is_full is False and self.index < self.sample_size:
            raise Exception("Not enough samples yet")
        if self.is_full is False:
            ixs = random.sample(range(0,self.index),self.sample_size)
        else:
            ixs = random.sample(range(0,self.mem_size),self.sample_size)
        sts_0 = np.zeros((self.sample_size,4))
        acs = np.zeros((self.sample_size,1))
        cts = np.zeros_like(acs)
        sts_1 = np.zeros_like(sts_0)
        for ixx, ix in enumerate(ixs):
            sts_0[ixx] = self.states[ix]
            acs[ixx] = self.actions[ix]
            cts[ixx] = self.costs[ix]
            sts_1[ixx] = self.next_states[ix]
        
        sts_0 = torch.from_numpy(sts_0).float()
        acs = torch.from_numpy(acs).float()
        cts = torch.from_numpy(cts).float()
        sts_1 = torch.from_numpy(sts_1).float()
        return sts_0, acs, cts, sts_1
       
def calculate_cost(obs, action, done, truncated):
    cost = 25*obs[1]**2 + 20*obs[0]**2 + 0.0001*action**2
    if done or truncated:
        cost += 10
    return cost

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

torch.autograd.set_detect_anomaly(True)
replay_memory_size = 1024
bsize = 64

replay = Replay_Buffer(replay_memory_size,bsize)

lrQ = .001
lrP = .0001
alpha = .99
tau = .001
loss_function_Q = nn.MSELoss()

critic = Q_critic()
actor = policy_actor()



optim_critic = optim.Adam(critic.parameters(),lrQ)
optim_actor = optim.Adam(actor.parameters(),lrP)

training = False

if training is True:
    critic_target = type(critic)()
    actor_target = type(actor)()
    critic_target.load_state_dict(critic.state_dict())
    actor_target.load_state_dict(actor.state_dict())
    env = gym.make('InvertedPendulum-v4',render_mode="human")
    observation, _ = env.reset()

    for i in tqdm(range(50000),"Time Step no. :"):
        action = torch.clip(actor(torch.from_numpy(observation).float())+np.random.normal(0.0,0.15),-1,1).reshape((1,1))
        action = action.detach().numpy()[0]
        action_scaled = 3*action
        if i == 0:
            print(action)
            print(action_scaled)

        observation_new, reward, done, truncated, info = env.step(action_scaled)
        ct = calculate_cost(observation,action_scaled,done,truncated)
        replay.store_data(observation, action, ct, observation_new)

        if i>=2*bsize:
            state0_batch, action_batch, cost_batch, state1_batch = replay.sample_data()
            q_batch = critic([state0_batch,action_batch])
            y_batch = cost_batch + alpha*critic_target([state1_batch,actor_target(state1_batch)])
            lossQ = loss_function_Q(q_batch,y_batch)
            optim_critic.zero_grad()
            lossQ.backward()
            optim_critic.step()

            optim_actor.zero_grad()
            loss_actor = critic([state0_batch,actor(state0_batch)])
            loss_actor = loss_actor.mean()
            loss_actor.backward()
            optim_actor.step()
            soft_update(critic_target,critic,tau)
            soft_update(actor_target,actor,tau)
        
        if truncated or done:
            observation, _ = env.reset()
        else:
            observation = observation_new
    torch.save(actor.state_dict(), 'actor.pt')
    torch.save(critic.state_dict(), 'critic.pt')
     
    env.close()

testing = True
if testing is True:
    env = gym.make('InvertedPendulum-v4', render_mode="rgb_array")
    actor.load_state_dict(torch.load('actor.pt'))
    actor.eval()
    observation, _ = env.reset()
    frames = []
    fps = 60
    for i in tqdm(range(1000),'timestep: '):
        if len(frames) < 0.02 * i * fps:
            frame = env.render()
            frames.append(frame)
        action = actor(torch.from_numpy(observation).float()).reshape((1,1))
        action = action.detach().numpy()[0]
        action_scaled = 3*action
        observation, reward, done, truncated, info = env.step(action_scaled)

        if truncated or done:
            observation, _ = env.reset()
    size = np.shape(frame)[0:2]
    result = cv2.VideoWriter('invertedPendulum_DDPG.avi',cv2.VideoWriter_fourcc(*'MJPG'),fps,size)
    for frame in frames:
        result.write(frame)
        cv2.imshow('frame',frame)
        cv2.waitKey(int(1/fps*1000))
    cv2.destroyAllWindows()
    result.release()
env.close()
