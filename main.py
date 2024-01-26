import time
import pandas as pd
import gymnasium as gym
import smart_grid
print(smart_grid.__file__)
from net import *
fname = 'train.xlsx'
csv_fname='train.csv'

train_frame = pd.read_excel(fname)
train_frame.loc[:, train_frame.columns != 'PRICES'] /= 1000

timeseries=parse_data(csv_fname)
timeseries_or=timeseries.copy()
# print('len',len(timeseries))
train,test,train_size,test_size=split_data(timeseries)
lookback = 4
X_full, y_full = create_dataset(timeseries, lookback=lookback)

X_train, y_train = create_dataset(train, lookback=lookback)


X_test, y_test = create_dataset(test, lookback=lookback)
# print('y_train',len(y_train))
# print('y_test',len(y_test))
model = LSTMModel()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
n_epochs = 20
model=train_LSTM(n_epochs,model,loader,loss_fn,optimizer,X_train,X_test,y_train,y_test)
y_pred = model(X_full)
y_pred=y_pred.cpu().detach().numpy()
y_pred = y_pred[:, -1, :]
print('pred',len(y_pred))
# print(y_pred)
# print(X_train)
plot(model,X_train,X_test,timeseries,lookback,train_size,test_size,timeseries_or)
actions=[]
for p in y_pred:
	# print(p)
	if p[0]<1.0:
		actions.append(1)
	elif 1.0<p[0]<28:
		actions.append(0)
	else:
		actions.append(2)
env = gym.make("smart_grid/DumbGrid-v0", time_data=train_frame, render_mode=None, delay=0.7)
# print(actions)
# print(type(env))
observation, info = env.reset()
# print(observation)
count=0
rewards=[]
# for _ in range(480):

#     action = env.action_space.sample()
#     # print(action)  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     # print(observation)
#     print(reward)
#     rewards.append(reward)
#     if terminated or truncated:
#         observation, info = env.reset()
# count, bins_count = np.histogram(rewards, bins=10) 
# pdf = count / sum(count) 

# cdf = np.cumsum(pdf) 
  
# # plotting PDF and CDF 
# plt.plot(bins_count[1:], pdf, color="red", label="PDF") 
# plt.plot(bins_count[1:], cdf, label="CDF") 
# plt.legend() 
# plt.savefig('./cumulative.png')
actions.append(0)
actions.append(0)
actions.append(0)
actions.append(0)
print(len(actions))
reward=0
for i in range(0,480):
	action=actions[i]
	observation, reward1, terminated, truncated, info = env.step(action)
	# print(observation)
	# print(reward1)
	reward=reward+reward1
	if terminated or truncated:
		observation, info = env.reset()
# print(reward)
time.sleep(3)
env.close()



