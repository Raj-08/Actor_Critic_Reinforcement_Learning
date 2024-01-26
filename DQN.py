import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from mpl_toolkits import mplot3d
from matplotlib import cm
import pandas as pd
import smart_grid
from TestEnv import Electric_Car 
env=Electric_Car('./train.xlsx')
import seaborn as sns
fname = 'train.xlsx'

train_frame = pd.read_excel(fname)
train_frame.loc[:, train_frame.columns != 'PRICES'] /= 1000
# class DQN(nn.Module):
#     def __init__(self,env,learning_rate):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=8, hidden_size=50, num_layers=1, batch_first=True)
#         self.linear = nn.Linear(50, 3)
#         self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.linear(x)
#         return x



class DQN(nn.Module):

    def __init__(self, env, learning_rate):

        super(DQN,self).__init__()
        input_features = 8
        action_space = 3


        #Solution:
        self.dense1=nn.Linear(in_features=input_features,out_features=512)
        self.dense2=nn.Linear(in_features=512,out_features=256)
        self.dense3=nn.Linear(in_features=256,out_features=128)
        self.dense4=nn.Linear(in_features=128,out_features=64)
        self.dense5=nn.Linear(in_features=64,out_features=32)
        self.dense6=nn.Linear(in_features=32,out_features=action_space)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

    def forward(self, x):

        x=torch.tanh(self.dense1(x))
        x=torch.tanh(self.dense2(x))
        x=torch.tanh(self.dense3(x))
        x=torch.tanh(self.dense4(x))
        x=torch.tanh(self.dense5(x))
        x=self.dense6(x)
        return x



class experience_replay:
    def __init__(self, env, buffer_size, min_replay_size = 1000, seed = 123):
        self.env = env
        self.min_replay_size = min_replay_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque([-200.0], maxlen = 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        observation_old= self.env.state
        # observation_old, info = env.reset()
        for _ in range(0,self.min_replay_size):
            action = random.choice([-1,0,1])
            observation_new, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = (observation_new, action, reward, done, observation_old)
            self.replay_buffer.append(transition)
            observation_old = observation_new
            # break


    def add_data(self, data):
        self.replay_buffer.append(data)
    def add_reward(self, reward):
        self.reward_buffer.append(reward)

    def sample(self, batch_size):
        transitions = random.sample(self.replay_buffer, batch_size)
        # print('len',np.asarray(transitions).shape)
        observations=[]
        actions_t=[]
        rewards_t=[]
        dones_t=[]
        new_observations_t=[]
        for i in range(0,batch_size):
            observations.append(transitions[i][0])
            actions_t.append(transitions[i][1])
            rewards_t.append(transitions[i][2])
            dones_t.append(transitions[i][3])
            new_observations_t.append(transitions[i][4])
        for j in range(0,batch_size):
            len1=len(new_observations_t[j])
            if len1!=8:
                l1=list(new_observations_t[j])
                l1.append(0.0)
                new_observations_t[j]=np.asarray(l1)

        observations=np.asarray(observations)
        actions_t=np.asarray(actions_t)
        rewards_t=np.asarray(rewards_t)
        dones_t=np.asarray(dones_t)
        # for t in range(0,len(new_observations_t)):
        #   print('s ' ,len((new_observations_t[t])))
        new_observations_t=np.asarray(new_observations_t,dtype=np.float32)

        # observations, actions_t, rewards_t, dones_t, new_observations_t= transitions
        observations_t1 = torch.as_tensor(observations, dtype = torch.float32, device=self.device)
        actions_t1 = torch.as_tensor(actions_t, dtype = torch.int64, device=self.device).unsqueeze(-1)
        rewards_t1 = torch.as_tensor(rewards_t, dtype = torch.float32, device=self.device).unsqueeze(-1)
        dones_t1 = torch.as_tensor(dones_t, dtype = torch.float32, device=self.device).unsqueeze(-1)
        new_observations_t1 = torch.as_tensor(new_observations_t, dtype = torch.float32, device=self.device)

        return observations_t1, actions_t1, rewards_t1, dones_t1, new_observations_t1

        # return observations, actions_t, rewards_t, dones_t, new_observations_t

class DDQNAgent:
    def __init__(self, env, device, epsilon_decay,
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed = 123):
        self.env=env
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size

        self.replay_memory = experience_replay(self.env, self.buffer_size, seed = seed)
        self.online_network = DQN(self.env, self.learning_rate).to(self.device)
        self.target_network = DQN(self.env, self.learning_rate).to(self.device)

    def choose_action(self,step, observation, greedy = False):
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        random_sample = random.random()

        if (random_sample >= epsilon):
        # if eps>epsilon:
            observation = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
            # print(observation.shape)
            observation=observation[None,:]
            q_values=self.online_network(observation)
            action=torch.argmax(q_values)
            action=action.cpu().detach()
            action=action-1

            # return action,epsilon
            ## take 
        else :
            action=random.choice([-1,0,1])
        return action,epsilon
    def learn(self,batch_size,update_agent):
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)
        # observations_t=torch.as_tensor(observations_t)
        # new_observations_t=torch.as_tensor(new_observations_t)

        target_q_values=self.target_network (new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards_t + self.discount_rate * (1-dones_t) * max_target_q_values  
        # print(observations_t.shape)
  
        q_values = self.online_network(observations_t)
        actions_t=actions_t+1
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        output = F.smooth_l1_loss(action_q_values, targets)

        # loss = nn.MSELoss()
        # output = loss(action_q_values, targets)
        output.backward()
        optimizer = torch.optim.SGD(params=self.online_network.parameters(), lr=0.001)
        optimizer.step()
        if update_agent==True:
            self.target_network.load_state_dict(self.online_network.state_dict())





class DQNAgent:
    def __init__(self, env, device, epsilon_decay,
                 epsilon_start, epsilon_end, discount_rate, lr, buffer_size, seed = 123):
        self.env = env
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.discount_rate = discount_rate
        self.learning_rate = lr
        self.buffer_size = buffer_size

        self.replay_memory = experience_replay(self.env, self.buffer_size, seed = seed)
        self.online_network = DQN(self.env, self.learning_rate).to(self.device)

    


    def choose_action(self,step, observation, greedy = False):
        epsilon = np.interp(step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        random_sample = random.random()

        if (random_sample >= epsilon):
        # if eps>epsilon:
            observation = torch.as_tensor(observation, dtype = torch.float32, device=self.device)
            q_values=self.online_network(observation)
            action=torch.argmax(q_values)
            action=action.cpu().detach()
            action=action-1

            # return action,epsilon
            ## take 
        else :
            action=random.choice([-1,0,1])
        return action,epsilon
    def learn(self,batch_size):
        observations_t, actions_t, rewards_t, dones_t, new_observations_t = self.replay_memory.sample(batch_size)
        # observations_t=torch.as_tensor(observations_t)
        # new_observations_t=torch.as_tensor(new_observations_t)

        target_q_values=self.online_network(new_observations_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards_t + self.discount_rate * (1-dones_t) * max_target_q_values    
        # print(observations_t.shape)
        q_values = self.online_network(observations_t)
        actions_t=actions_t+1
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.MSELoss()
        output = loss(action_q_values, targets)
        output.backward()
        optimizer = torch.optim.SGD(params=self.online_network.parameters(), lr=0.001)
        optimizer.step()


#Set the hyperparameters

#Discount rate
discount_rate = 0.99
#That is the sample that we consider to update our algorithm
batch_size = 32
#Maximum number of transitions that we store in the buffer
buffer_size = 50000
#Minimum number of random transitions stored in the replay buffer
min_replay_size = 1000
#Starting value of epsilon
epsilon_start = 1.0
#End value (lowest value) of epsilon
epsilon_end = 0.05
#Decay period until epsilon start -> epsilon end
epsilon_decay = 10000

max_episodes = 500000

#Learning_rate
lr = 5e-4

# env = gym.make("Electric_Car", time_data=train_frame, render_mode=None, delay=0.7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vanilla_agent = DDQNAgent(env, device, epsilon_decay, epsilon_start, epsilon_end, discount_rate, lr, buffer_size)


def training_loop(env,agent, max_episodes, target_ = False, seed=42):

    
    # env.action_space.seed(seed)
    # obs, _ = env.reset(seed=seed)
    average_reward_list = [-200]
    steps=[0.0]
    episode_reward = 0.0
    obs=env.state
    for step in range(max_episodes):

        action, epsilon = agent.choose_action(step, obs)
        new_obs, rew, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        transition = (obs, action, rew, done, new_obs)
        agent.replay_memory.add_data(transition)
        obs = new_obs

        episode_reward += rew
        
        if done:
            obs = env.reset()
            # env.state = np.empty(7)
            # env=Electric_Car('./train.xlsx')

            env.price_values = env.test_data.iloc[:, 1:25].to_numpy()

            # obs, _ = env.reset(seed=seed)
            agent.replay_memory.add_reward(episode_reward)
            #Reinitilize the reward to 0.0 after the game is over
            episode_reward = 0.0
        update_agent=False
        #Learn
        if step%500:
            update_agent=True
        agent.learn(batch_size,update_agent)

        #Calculate after each 100 episodes an average that will be added to the list

        if (step+1) % 100 == 0:

            steps.append(step)
            average_reward_list.append(np.mean(agent.replay_memory.reward_buffer))

        #Update target network, do not bother about it now!
        # if target_:

        #     #Set the target_update_frequency
        #     target_update_frequency = 250
        #     if step % target_update_frequency == 0:
        #         dagent.update_target_network()

        #Print some output
        if (step+1) % 10000 == 0:
            print(20*'--')
            print('Step', step)
            print('Epsilon', epsilon)
            print('Avg Rew', np.mean(agent.replay_memory.reward_buffer))
            print()

    return average_reward_list,steps

average_rewards_vanilla_dqn,steps = training_loop(env, vanilla_agent, max_episodes)
xpoints = np.array(steps)
ypoints = np.array(average_rewards_vanilla_dqn)

plt.plot(xpoints, ypoints)
# plt.show()
plt.savefig('./agentL1.png')