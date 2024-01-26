## Actor-Critic Algorithm

In Actor-Critic algorithms, we simultaneously train an actor (policy) and a critic (value function). The actor guides the agent's actions, and the critic evaluates the state or state-action values.


### Mathematical Formulation


```math
\mathbb{E}_{\pi_{\theta}} \left[ \left( \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot A_{\pi_{\theta}}(s_t, a_t) \right)^2 \leq \left( \sum_{t=0}^{T} (\nabla_{\theta} \log \pi_{\theta}(a_t|s_t))^2 \right) \cdot \mathbb{E}_{\pi_{\theta}} \left[ \sum_{t=0}^{T} (A_{\pi_{\theta}}(s_t, a_t))^2 \right] \right]
```
The model uses Torch and OpenAIGym  - 
```
pip install -r requirements.txt
```

To run vanilla DDQN - 

```
python DQN.py
```

To run Actor Crtic DDQN - 

```
python DQN-AC.py
```
```
#Set the hyperparameters

#Discount rate
discount_rate = 0.9
#That is the sample that we consider to update our algorithm
batch_size = 64
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

max_episodes = 20000000

#Learning_rate
lr = 5e-6
```
### Cumulative Rewards over time
![plot](agentL1.png)
