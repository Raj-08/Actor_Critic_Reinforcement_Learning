import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from matplotlib import pyplot as plt


def parse_data(csv_file):
    timeseries=[]
    df = pd.read_csv(csv_file)
    all_rows=df[["PRICES"]].values
    hours=['Hour 01',  'Hour 02', 'Hour 03', 'Hour 04' ,'Hour 05', 'Hour 06', 'Hour 07', 'Hour 08' ,'Hour 09', 'Hour 10', 'Hour 11' ,'Hour 12' ,'Hour 13', 'Hour 14', 'Hour 15', 'Hour 16', 'Hour 17', 'Hour 18' ,'Hour 19' ,'Hour 20', 'Hour 21' ,'Hour 22', 'Hour 23' ,'Hour 24']
    for i in range(0,1096):
        for hour in hours:
            timeseries.append([df[[hour][0]].values[i].astype('float32')])
    return timeseries
#

def split_data(timeseries):
    print('time',len(timeseries))
    train_size = int(len(timeseries) * 0.70)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]
    return train,test,train_size,test_size

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)
 

 
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def train_LSTM(n_epochs,model,loader,loss_fn,optimizer,X_train,X_test,y_train,y_test): 
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            print(X_batch.shape)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    return model

def plot(model,X_train,X_test,timeseries,lookback,train_size,test_size,timeseries_or): 
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
    t_series=[]
    for t  in timeseries_or:
        t_series.append(t[0])

    # print('time',t_series)
    n = 4
    lists = [[] for _ in range(n)]
    start=0
    for i in range(0,6):
        for l in lists:
            t=t_series[start]
            l.append(t)
            start=start+1
            if start==4032:
                continue
    # print(lists)
    l_num=np.asarray(lists)
    print(l_num.shape)
    sum_all=np.mean(l_num,axis=1)
    print(sum_all.shape)
    # plot
    # plt.plot(t_series)
    hours_per_day = 24
    total_hours = len(t_series)
    custom_x_axis = np.tile(np.arange(hours_per_day), total_hours // hours_per_day + 1)[:total_hours]
    # print('len',len(custom_x_axis))
    # plot
    new_custom_x_axis=[]
    count=0
    for k in custom_x_axis:
        new_custom_x_axis.append(str(k))
        count=count+1
    # print(new_custom_x_axis)
    plt.plot(sum_all,label='Fluctuation of prices per 6 months')
    plt.ylabel("Price")
    plt.xlabel("Averaged 6 months")

    # plt.xticks(t_series, new_custom_x_axis)
    # plt.plot(custom_x_axis, t_series, label='Original Time Series')
    # plt.plot(train_plot, c='r')
    # plt.plot(test_plot, c='g')
    plt.savefig('./per 6 months.png')
# plt.show()