    def _get_reward(self):
        # Using walker2D reward function net reward = healthy reward + forward reward - control loss
        #healthy reward- discontinuous
        if self._is_done() != True:
            healthy_reward = 200
        else:
            healthy_reward = -100

        #forward velocity
        forward_reward = (1000)*(np.abs(self.data.qpos[1])) + (500)*(np.abs(self.data.qvel[1]) + np.abs(self.data.qvel[0])) 

        #control loss
        control_loss= 0.001*(np.abs(self.data.ctrl[0]) + np.abs(self.data.ctrl[1])) + 0.001*(np.abs(self.data.qacc[7]) + np.abs(self.data.qacc[9]))
        # print(healthy_reward, forward_reward, control_loss)

        net_reward = healthy_reward + forward_reward - control_loss 

        return net_reward

    def _get_reward(self):
        # Using walker2D reward function net reward = healthy reward + forward reward - control loss
        #Continuous healthy reward
 
        healthy_reward = (100)*np.abs(self.data.geom('axle').xpos[2]) 

        #forward velocity
        forward_reward = (1000)*(np.abs(self.data.qpos[1])) + (500)*(np.abs(self.data.qvel[1]) + np.abs(self.data.qvel[0])) 

        #control loss
        control_loss= 0.001*(np.abs(self.data.ctrl[0]) + np.abs(self.data.ctrl[1])) + 0.001*(np.abs(self.data.qacc[7]) + np.abs(self.data.qacc[9]))
        # print(healthy_reward, forward_reward, control_loss)

        net_reward = healthy_reward + forward_reward - control_loss 
        
        return net_reward 

    #Time varying rewards
    def _get_reward(self):
        # Using walker2D reward function net reward = healthy reward + forward reward - control loss
        #healthy reward

        #healthy_reward = 100*(self.data.time) /((100)*np.abs(self.data.geom('axle').xpos[2]))
        if self._is_done != True:
            healthy_reward = 100 

        #forward velocity
        forward_reward = (1000)*(np.abs(self.data.qpos[1])) + (500)*(np.abs(self.data.qvel[1]) + np.abs(self.data.qvel[0])) 

        #control loss
        control_loss= 0.001*(np.abs(self.data.ctrl[0]) + np.abs(self.data.ctrl[1])) #+ 0.001*(np.abs(self.data.qacc[7]) + np.abs(self.data.qacc[9]))
        
        #time varying penalty
        time_loss = 1*(self.data.time) /((10)*np.abs(self.data.qpos[1]))
        
        # print(healthy_reward, forward_reward, control_loss)

        net_reward = healthy_reward + forward_reward - control_loss - time_loss
        
        return net_reward


    #sigmoid scaled rewards


    def sigmoid(x, a=7.85, b=1.96, c=0.21):
        z = 1/(1 + np.exp(c*(-a*x + b*np.exp(2))))

    return z

    def _get_reward(self):
        # Using walker2D reward function net reward = healthy reward + forward reward - control loss
        #healthy reward

        healthy_reward = 10*np.abs(self.data.geom('axle').xpos[2])
        healthy_reward = sigmoid(healthy_reward)

        #forward velocity
        forward_reward = 10*(np.abs(self.data.qpos[1])) #+ (100)*(np.abs(self.data.qvel[1]) + np.abs(self.data.qvel[0])) 
        forward_reward = sigmoid(forward_reward)

        #control loss
        control_loss= 10*(np.abs(self.data.ctrl[0]) + np.abs(self.data.ctrl[1])) #+ (np.abs(self.data.qacc[7]) + np.abs(self.data.qacc[9]))


        net_reward = 30*healthy_reward + 5*forward_reward - 5*control_loss 
        # print(net_reward)
        
    return net_reward