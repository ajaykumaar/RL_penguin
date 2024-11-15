import gym
import numpy as np
import mujoco as mj
import cv2
# import cus_replay_buff

def quaternion_to_euler_angle(x, y, z, w=1):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return (X, Y, Z) 

class penguin_env(gym.Env):
    def __init__(self, xml='test_robot.xml', action_size=2, duration=10.0, frame_size=(480,640),camera="track", dt="0.003",seed=735,noise=0.001, randomize=False):
        # Load MuJoCo model
        self.max_episode_steps= 10000
        self.model = mj.MjModel.from_xml_path(xml)
        self.data = mj.MjData(self.model)
        self.renderer= mj.Renderer(self.model,frame_size[0],frame_size[1])
        self.truncation_time = duration
        self.dt = self.model.opt.timestep
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.randomize = randomize
        print(self.model.nq)
        print(self.model.nv)
        #mj.mj_forward(self.model, self.data)
        self.camera = camera

        self.renderer.update_scene(self.data,self.camera)
        #self.frames=[]

        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-.078, high=.078, shape=(action_size,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,))

    def step(self, action):
        # Take a step in the simulation
        #ctrl=action
        self.data.ctrl= action

        # Update Mujoco step
        mj.mj_step(self.model, self.data)

        # Get observation, reward, and done flag
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        truncated=True if self.data.time>=self.truncation_time else False
        info={}

        # return obs, reward, done, {}
        if truncated == True:
            return (obs,reward,done, info)
        elif truncated == False:
            return (obs,reward,done,info)


    def reset(self, seed=None, options=None):
        # Reset the simulation to initial state
        mj.mj_resetData(self.model,self.data)
        if self.randomize is True:
            qpos = self.data.qpos + self.rng.uniform(-self.noise,self.noise,size=self.model.nq)
            qvel = self.data.qvel + self.rng.uniform(-self.noise,self.noise,size=self.model.nv)

            self.data.qpos = qpos
            self.data.qvel = qvel
        mj.mj_forward(self.model, self.data)

        info={}
        obs= self._get_observation()
        #return (obs,info)
        return (obs)

    def render_frames(self):
        self.renderer.update_scene(self.data,self.camera)
        frame = self.renderer.render().copy()
        return frame

    def seed(self, seed=None):
        # Set the seed for the environment's random number generator
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        # Get observation from the simulation state

        axle_qs = self.data.qpos[0:7]
        right_qs = self.data.qpos[7:9]
        left_qs = self.data.qpos[11:13]
        axles_qvs = self.data.qvel[0:6]
        right_qvs = self.data.qvel[6:8]
        left_qvs = self.data.qvel[10:12]
        axle_qacc = self.data.qacc[0:6]
        right_qacc = self.data.qacc[6:8]
        left_qacc = self.data.qacc[10:12]
        time = self.data.time

        full_obs = np.concatenate([axle_qs, right_qs, left_qs, axles_qvs, right_qvs,
                              left_qvs, axle_qacc, right_qacc, left_qacc, np.array([time])])

        COM_euler= quaternion_to_euler_angle(x=full_obs[3], y=full_obs[4], z=full_obs[5])

        obs= np.concatenate([full_obs[:3], COM_euler, full_obs[11:14], full_obs[14:21]])

        #full_obs[:3] - [COM side position(x), COM forward(y), COM height(z)]
        #COM_euler - euler angle(XYZ) of COM
        #full_obs[11:14] - [COM sway vel(x), COM forward vel(y), COM vertical vel(z)]
        #full_obs[14:21]- COM ang vel (14-16), right hip ang vel (17), right motor ang vel(18), left hip ang vel(19), left motor ang vel(20)
        #print("observation",obs)
        return obs.astype(np.float32)


    def _get_reward(self):
        # time varying cost function for MCTS
        #healthy reward
        if self._is_done() != True:
            healthy_reward = 200
        else:
            healthy_reward = -100

        #forward velocity
        forward_reward = (500)*(np.abs(self.data.qpos[1])) + (100)*(np.abs(self.data.qvel[1]) + np.abs(self.data.qvel[0])) 

        #control loss
        control_loss= 0.0001*(np.abs(self.data.ctrl[0]) + np.abs(self.data.ctrl[1])) + 0.0001*(np.abs(self.data.qacc[7]) + np.abs(self.data.qacc[9]))
        # print(healthy_reward, forward_reward, control_loss)
        
        net_reward = healthy_reward + forward_reward - control_loss 

        return net_reward


    def _is_done(self):
        # Check if episode is done
        done = False
        if np.abs(self.data.geom('axle').xpos[2]) < 0.06:
            done = True

        return done




# Register the environment
gym.envs.register(id='penguin_env-v0', entry_point=penguin_env)

# Create an instance of the environment
# env = gym.make('penguin_env-v0')



def test_model(env, model, num_steps):
    #generates trajectory plots and saves video
    print("eval mode...")
    y_traj=[]
    x_traj=[]
    z_traj=[]
    frames=[]
    obs = env.reset()
    for i in range(1000):
        action, _states = model.select_action(obs)
        print(action)
        try:
            obs, rewards, dones, info = env.step(action)
        except:
            obs, rewards, dones,truncated, info = env.step(action)
        frames.append(env.render_frames())

        x_traj.append(obs[0])
        y_traj.append(obs[1])
        z_traj.append(obs[2])
    #env.render()
    fps=60
    # for f in frames:
    #     cv2.imshow('frame',f)
    #     cv2.waitKey(wait_time)
    # cv2.destroyAllWindows()

    print("saving video")
    size = np.shape(frames[0])[0:2]

    result = cv2.VideoWriter('video_progress/penguin_SAC_test0_'+str(num_steps)+'.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,size)

    for frame in frames:
        frame=cv2.resize(frame,size)
        result.write(frame)
        #cv2.imshow('frame',frame)
        #cv2.waitKey(int(1/fps*1000))
    print("done saving video")
    result.release()

