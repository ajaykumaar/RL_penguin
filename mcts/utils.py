import numpy as np

#only for 2 controls

class discretize_action_space():
    def __init__(self, action_bins=5, action_min= -.078, action_max = .078):

        self.action_list= np.linspace(action_min,action_max,action_bins)
        self.action_dict = {} 

        key_count=0
        for i in self.action_list:
            for j in self.action_list:
                self.action_dict[key_count] = [i,j]

                key_count+=1

    def get_discretized_action(self,actions):

        dis_action=[]

        for action in actions:

            diff_list= [np.abs(d_a - action) for d_a in self.action_list]
            bin_num = np.argmin(diff_list)

            dis_act= self.action_list[bin_num]

            dis_action.append(dis_act)

        return dis_action


    def get_action_in_bin(self,bin_num):

        return self.action_dict.get(bin_num)














# action_bins = 5
# num_actions = 2

# action_min, action_max = -.078, .078
# action_list= np.linspace(action_min,action_max,action_bins)

# action_dict = {} 

# key_count=0
# for i in action_list:
#     for j in action_list:
#         action_dict[key_count] = [i,j]

#         key_count+=1
    
# def discretize_action_space(actions):

#     dis_action=[]

#     for action in actions:

#         diff_list= [np.abs(d_a - action) for d_a in action_list]
#         bin_num = np.argmin(diff_list)

#         dis_act= action_list[bin_num]

#         dis_action.append(dis_act)

#     return dis_action

# # print(action_list)
# # print(rand_act)
# # print(discretize_action_space(rand_act))



# def get_action_in_bin(bin_num):

#     return action_dict.get(bin_num)





# # class discretize_action_space():
# #     def __init__(self, action_bins=10, action_min= -2.0, action_max = 2.0):
# #         self.action_bins= action_bins
# #         self.action_min = action_min
# #         self.action_max = action_max

# #     def get_discretized_action(self,action):

# #         action_bin_size = (self.action_max - self.action_min) / self.action_bins

# #         action_bin = int((action - self.action_min) / action_bin_size)

# #         dis_action = (action * action_bin_size) + self.action_min + (action_bin_size / 2)

# #         return np.clip(dis_action, self.action_min, self.action_max)


