import gym
import numpy as np
import itertools
import torch
import mujoco
from model import QNetwork, ValueNetwork, GaussianPolicy
from replay_buffer import ReplayMemory
import utils
from sac import SAC

import gym_penguin

env = gym.make('penguin_env-v0')

# env.seed(200)
# env.action_space.seed(200)
torch.manual_seed(200)
np.random.seed(200)

agent = SAC(env.observation_space.shape[0], env.action_space)
memory = ReplayMemory(capacity=1000000, seed=200)

# Training Loop
total_numsteps = 0
updates = 0

updates_per_step=1
batch_size=256
start_steps = 10000
num_steps = 1000001
eval = True


for i_episode in range(20):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        print("starting...")
        if start_steps > total_numsteps:
            print("Sampling...")
            action = env.action_space.sample()  # Sample random action
        else:
            print("Selecting action...")
            action = agent.select_action(state)  # Sample action from policy
            print(action)

        if len(memory) > batch_size:
            # Number of updates per step in environment
            for i in range(updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, batch_size, updates)

                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        mask = 1 if episode_steps == env.max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        #writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

#save model
agent.save_checkpoint

gym_penguin.test_model(env=env, model=agent, num_steps=20)

env.close()