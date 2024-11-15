import mujoco as mj
import numpy as np
import cv2
from decimal import Decimal
from typing import Dict
from stable_baselines3.common.buffers import DictReplayBuffer

# class CustomReplayBuffer:
#     def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
#         self.buffer_size = buffer_size
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
#         self.next_obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
#         self.action_buf = np.zeros((buffer_size, action_dim), dtype=np.float32)
#         self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
#         self.done_buf = np.zeros(buffer_size, dtype=bool)
#         self.idx = 0
#         self.size = 0

#     def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
#         self.obs_buf[self.idx] = obs
#         self.next_obs_buf[self.idx] = next_obs
#         self.action_buf[self.idx] = action
#         self.reward_buf[self.idx] = reward
#         self.done_buf[self.idx] = done
#         self.idx = (self.idx + 1) % self.buffer_size
#         self.size = min(self.size + 1, self.buffer_size)

#     def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
#         idxs = np.random.choice(self.size, size=batch_size, replace=False)
#         # idxs=range(0,batch_size)
#         batch = {
#             'obs': self.obs_buf[idxs],
#             'next_obs': self.next_obs_buf[idxs],
#             'actions': self.action_buf[idxs],
#             'rewards': self.reward_buf[idxs],
#             'dones': self.done_buf[idxs],
#         }
#         return batch

#     def __len__(self) -> int:
#         return self.size

class CustomReplayBuffer(DictReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space):
        super().__init__(buffer_size, observation_space, action_space)
    
    def add(self, obs, next_obs, action, reward, done, info):
        super().add(obs, next_obs, action, reward, done)


def is_done(data):
    # Check if episode is done
    done = False
    if np.abs(data.geom('axle').xpos[2]) < 0.06:
        done = True

    return done

def get_obs(data):

    axle_qs = data.qpos[0:7]
    right_qs = data.qpos[7:9]
    left_qs = data.qpos[11:13]
    axles_qvs = data.qvel[0:6]
    right_qvs = data.qvel[6:8]
    left_qvs = data.qvel[10:12]
    axle_qacc = data.qacc[0:6]
    right_qacc = data.qacc[6:8]
    left_qacc = data.qacc[10:12]
    time = data.time

    full_obs = np.concatenate([axle_qs, right_qs, left_qs, axles_qvs, right_qvs,
                            left_qvs, axle_qacc, right_qacc, left_qacc, np.array([time])])
                        
    return full_obs.astype(np.float32)

def get_reward(data):
    # Using walker2D reward function net reward = healthy reward + forward reward - control loss
    #healthy reward
    if is_done(data) != True:
        healthy_reward = 100
    else:
        healthy_reward = -500

 
    forward_reward = (100)*(np.abs(data.qpos[0]) + np.abs(data.qpos[1])) + (50)*(np.abs(data.qvel[1]) + np.abs(data.qvel[0]))  #+ 50*(sway_vel)
 

    #control loss
    control_loss= 0.00001*(np.abs(data.ctrl[0]) + np.abs(data.ctrl[1]))

    
    net_reward = healthy_reward + forward_reward - control_loss

    return net_reward


replay_buffer_class= CustomReplayBuffer(buffer_size=10,observation_space=32, action_space=2)
 

def populate_replay_buffer(replay_buffer_class, buffer_size=10):

    # replay_buffer= CustomReplayBuffer(buffer_size=buffer_size, obs_dim=32, action_dim=2)

    model = mj.MjModel.from_xml_path("torque_robot.xml")
    data = mj.MjData(model)
    renderer = mj.Renderer(model,480,640)


    mj.mj_forward(model, data)
    renderer.update_scene(data,"track")
    #print(renderer.render())
    freq=1.6

    angular_freq=2*np.pi*freq
    amplitude=np.pi*.25
    amplitude = np.pi
    fps=60
    duration=5

    cnt=0
    frames = []

    t = 0
    #Q usefull: [0-8,11-12,17]

    for i in range(buffer_size):

        obs_buf= get_obs(data)

        ctrl = amplitude*np.sin(angular_freq*data.time)
        data.ctrl=np.array([ctrl,ctrl])
        mj.mj_step(model, data)

        #save to buffer
        action_buf= np.array([ctrl,ctrl])
        next_obs_buf= get_obs(data)
        reward_buf= get_reward(data)
        dones_buf= is_done(data)

        replay_buffer_class.add(obs= obs_buf, next_obs= next_obs_buf, action= action_buf, reward= reward_buf, done= dones_buf)


        if data.qpos[2] < .09:
            print('robot has fallen')
            
            time_old = data.time
            mj.mj_resetData(model,data)
            data.time = time_old

    print("Replay buffer populated!")


