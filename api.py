import yfinance as yf
import pandas as pd
import numpy as np
import datetime

import calendar

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.multiprocessing

from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image

np.set_printoptions(precision=4,suppress=True)
torch.multiprocessing.set_sharing_strategy('file_system')

#check tick name
# tick = '6758.T' #sony 'SONY' #sony adr '^IXIC' #NAS '^DJI' #DAW 'JPY=X' #JPY 'NIY=F' #NIkkei

#TODO::Nikkei Sakimono
tickers = ['6758.T','SONY','^IXIC','^DJI','JPY=X','^N225','NIY=F']
tickerColmns = ['Open','High','Low','Close']

if False : # re download
    for tick in tickers:
        print(tick)
        data = yf.download(tick, period='1y', interval = "1d")
        data.to_csv(tick+'.csv', encoding='utf8')

if False : # data check
    tickerList = []
    for tick in tickers:
        #print(tick)
        tickerList.append(pd.read_csv(tick+'.csv'))
        #print(tickerList[-1])

    dt = tickerList[0]

    #Date,Open,High,Low,Close,Volume
    print("print data",type(dt))

    ####date conversion
    print(dt.loc[:,'Date']) #get colmon
    print(dt.loc[:,'Date'][0]) #get colmon
    dates = pd.to_datetime(dt.loc[:,'Date'])
    print("print date")
    print(dates)
    print(dates[10])
    t = dates[10].date()
    print(t.year)#int
    print(t.month)#int
    print(t.day)#int
    ####data conversion
    for col in tickerColmns :
        #print("print data",type(dt))
        #print(dt.loc[:,col]) #get colmon
        #print(dt.loc[:,col][0]) #get colmon
        datas = dt.loc[:,col].to_numpy()
        #print("print data")
        #print(datas)
        print(datas[10])

    #1)create "date" array
    #2)put (ratio - 1.0)
    # colmns3D [5d][tick][high/low] --> [high/open]/[low/open]


######data format clean up######
if False:
    tickerList = []
    for tick in tickers:
        #print(tick)
        tickerList.append(pd.read_csv(tick+'.csv'))
        #print(tickerList[-1])

    year_month_list = [
        [2023, 7],
        [2023, 8],
        [2023, 9],
        [2023, 10],
        [2023, 11],
        [2023, 12],
        [2024, 1],
        [2024, 2],
        [2024, 3],
        [2024, 4],
        [2024, 5]
    ]

    week_day_list = [] # working day list

    for year,month in year_month_list :
        for date in range(1, calendar.monthrange(year, month)[1] + 1): #calendar.monthrange(year, month)[1] is number of day in the month
            day = datetime.date(year, month, date)
            if day.weekday() < 5 : #Monday -to-Fri
                week_day_list.append(day)
                #print(day)

    all_data = [[-1.0 for j in range(len(tickerColmns) * len(tickers) + 4)] for i in range(len(week_day_list))] #(week_days, tickers_num x 4 + (year,month,day,weekday))
    all_data_np = np.array(all_data,dtype=float)

    for day in range(0,len(week_day_list)):
        #print(week_day_list[day].year,week_day_list[day].month,week_day_list[day].day)
        for tick_num in range(len(tickerList)):
            tick = tickerList[tick_num]
            for tick_row in range(len(tick.loc[:,'Open'])):
                tick_date = pd.to_datetime(tick.loc[:,'Date'][tick_row]).date()
                if(tick_date == week_day_list[day]):
                    #print(tick_date, week_day_list[day])
                    for col_num in range(len(tickerColmns)) :
                        col = tickerColmns[col_num]
                        all_data_np[day][len(tickerColmns)*tick_num + col_num] = tick.loc[:,col][tick_row]

    for day in range(0,len(week_day_list)):
        all_data_np[day][-4] = week_day_list[day].year
        all_data_np[day][-3] = week_day_list[day].month
        all_data_np[day][-2] = week_day_list[day].day
        all_data_np[day][-1] = week_day_list[day].weekday()

    #fill -1 place to previous day data
    for day in range(1,len(all_data_np)):
        for col in range(0,len(all_data_np[0])):
            if(all_data_np[day][col] < 0.0):
                all_data_np[day][col] = all_data_np[day - 1][col]

    np.savetxt('./np_savetxt.csv', all_data_np, delimiter=',', fmt='%f')

    #nyu-ryoku
    #test_data[allday-5][tick=4][day=5+1][3=open/pre_close,high/open,low/open,close/open,weekday (last=open/lose x 3)]
    test_data_raw = [[[[0.0 for l in range(5)] for k in range(6)] for j in range(len(tickers))] for i in range(len(all_data_np) - 6)] #(week_days-6, tickers_num x 4 + (year,month,day,weekday))
    test_target_high_raw = [0.0 for k in range(len(all_data_np) - 5)]
    test_target_low_raw = [0.0 for k in range(len(all_data_np) - 5)]
    
    test_data = np.array(test_data_raw,dtype=float)
    test_target_high = np.array(test_target_high_raw,dtype=float)
    test_target_low = np.array(test_target_low_raw,dtype=float)

    test_data_ori = np.array(test_data_raw,dtype=float)
    test_target_high_ori = np.array(test_target_high_raw,dtype=float)
    test_target_low_ori = np.array(test_target_low_raw,dtype=float)

    for day_bak in range(len(test_data)):
        day_now = day_bak  + 6
        test_target_high[day_bak] = ((all_data_np[day_now][1] / all_data_np[day_now][0]) - 1.0) * 10.0 #high / open
        test_target_high_ori[day_bak] = all_data_np[day_now][1]
        test_target_low[day_bak] = ((all_data_np[day_now][2] / all_data_np[day_now][0]) - 1.0) * 10.0 #low / open
        test_target_low_ori[day_bak] = all_data_np[day_now][2]
        for tick in range(len(tickers)):
            tick_offset = tick * 4
            for sub_day in range(5):
                day = day_bak + 1 + sub_day
                test_data[day_bak][tick][sub_day][0] = ((all_data_np[day][tick_offset+0] / all_data_np[day-1][tick_offset+3])-1.0)*1000.0#open/pre_close
                test_data[day_bak][tick][sub_day][1] = ((all_data_np[day][tick_offset+1] / all_data_np[day][tick_offset])-1.0)*1000.0#high/open
                test_data[day_bak][tick][sub_day][2] = ((all_data_np[day][tick_offset+2] / all_data_np[day][tick_offset])-1.0)*1000.0#low/open
                test_data[day_bak][tick][sub_day][3] = ((all_data_np[day][tick_offset+3] / all_data_np[day][tick_offset])-1.0)*1000.0#close/open
                test_data[day_bak][tick][sub_day][4] = (all_data_np[day][-1])/10.0#weekday

                test_data_ori[day_bak][tick][sub_day][0] = all_data_np[day][tick_offset+0]
                test_data_ori[day_bak][tick][sub_day][1] = all_data_np[day][tick_offset+1]
                test_data_ori[day_bak][tick][sub_day][2] = all_data_np[day][tick_offset+2]
                test_data_ori[day_bak][tick][sub_day][3] = all_data_np[day][tick_offset+3]
                test_data_ori[day_bak][tick][sub_day][4] = all_data_np[day][-1]

            test_data[day_bak][tick][5][0] = ((all_data_np[day_now][0] / all_data_np[day_now-1][3]) - 1.0) * 1000.0
            test_data[day_bak][tick][5][1] = ((all_data_np[day_now][0] / all_data_np[day_now-1][3]) - 1.0) * 1000.0
            test_data[day_bak][tick][5][2] = ((all_data_np[day_now][0] / all_data_np[day_now-1][3]) - 1.0) * 1000.0
            test_data[day_bak][tick][5][3] = ((all_data_np[day_now][0] / all_data_np[day_now-1][3]) - 1.0) * 1000.0
            test_data[day_bak][tick][5][4] = (all_data_np[day_now][-1]) / 10.0
    
            test_data_ori[day_bak][tick][5][0] = all_data_np[day_now][0]
            test_data_ori[day_bak][tick][5][1] = all_data_np[day_now][0]
            test_data_ori[day_bak][tick][5][2] = all_data_np[day_now][0]
            test_data_ori[day_bak][tick][5][3] = all_data_np[day_now][0]
            test_data_ori[day_bak][tick][5][4] = all_data_np[day_now][-1]

    print(test_data.shape)
    print(test_data[0])
    print(test_target_high.shape)
    print(test_target_high[0])
    print(test_target_low.shape)
    print(test_target_low[0])

    np.save('test_data',test_data)
    np.save('test_target_high',test_target_high)
    np.save('test_target_low',test_target_low)

    np.save('test_data_ori',test_data_ori)
    np.save('test_target_high_ori',test_target_high_ori)
    np.save('test_target_low_ori',test_target_low_ori)

test_data_ld = np.load('test_data.npy')
test_target_high_ld = np.load('test_target_high.npy')
test_target_low_ld = np.load('test_target_low.npy')

test_data_ld_ori = np.load('test_data_ori.npy')
test_target_high_ld_ori = np.load('test_target_high_ori.npy')
test_target_low_ld_ori = np.load('test_target_low_ori.npy')

test_target_ld = np.concatenate([test_target_high_ld.reshape(len(test_target_high_ld),1), test_target_low_ld.reshape(len(test_target_low_ld),1)], 1)
test_target_ld_ori = np.concatenate([test_target_high_ld_ori.reshape(len(test_target_high_ld_ori),1), test_target_low_ld_ori.reshape(len(test_target_low_ld_ori),1)], 1)

test_data_ld_f = test_data_ld.astype('float32') 
test_target_f = test_target_ld.astype('float32') 

test_data_ld_f_ori = test_data_ld_ori.astype('float32') 
test_target_f_ori = test_target_ld_ori.astype('float32') 


##hiper parameters
BATCH_SIZE = 20
#WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.001
EPOCH = 300
PATH = "./"
model_path = 'model.pth'
device = torch.device("cpu")

class userData(Dataset):

    def __init__(
        self,
        data : torch.Tensor,
        target : torch.Tensor,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        if target_transform is not None:
            self.target_transform = transform
        else:
            self.target_transform = None
        
        self.data = data
        self.targets = target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        #data.requires_grad = True
        #target.requires_grad = True

        return data, target

    def __len__(self) -> int:
        return len(self.data)


userDt = userData(data = torch.from_numpy(test_data_ld_f),target = torch.from_numpy(test_target_f))
userDt_ori = userData(data = torch.from_numpy(test_data_ld_f_ori),target = torch.from_numpy(test_target_f_ori))

n_samples = len(userDt) # n_samples is 60000
train_size = int(len(userDt) * 0.8) # train_size is 48000
val_size = n_samples - train_size # val_size is 48000

# shuffleしてから分割してくれる.
#train_dataset, val_dataset = torch.utils.data.random_split(userDt, [train_size, val_size])

#zengo de wakeru
subset1_indices = list(range(0,train_size)) # [0,1,.....47999]
subset2_indices = list(range(train_size,n_samples)) # [48000,48001,.....59999]

train_dataset = Subset(userDt, subset1_indices)
val_dataset   = Subset(userDt, subset2_indices)

train_dataset_ori = Subset(userDt_ori, subset1_indices)
val_dataset_ori   = Subset(userDt_ori, subset2_indices)

#nets & training
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1)
testloader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(7, 64, (1, 5), 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)#128*7*6=768
        x = self.fc1(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

net = Net()
net = net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

train_loss_value=[]      #trainingのlossを保持するlist
test_loss_value=[]       #testのlossを保持するlist

if False :
    for epoch in range(EPOCH):
        print('epoch', epoch+1)    #epoch数の出力
        for (inputs, labels) in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        sum_loss = 0.0          #lossの合計
        sum_total = 0           #dataの数の合計

        #train dataを使ってテストをする(パラメータ更新がないようになっている)
        for (inputs, labels) in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss                            #lossを足していく
        print("train mean loss={}"
                .format(sum_loss*BATCH_SIZE/len(trainloader.dataset)))  #lossとaccuracy出力
        train_loss_value.append(sum_loss.to('cpu').detach().numpy()*BATCH_SIZE/len(trainloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持

        sum_loss = 0.0
        sum_total = 0

        #test dataを使ってテストをする
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss
        print("test  mean loss={},"
                .format(sum_loss*BATCH_SIZE/len(testloader.dataset)))
        test_loss_value.append(sum_loss.to('cpu').detach().numpy()*BATCH_SIZE/len(testloader.dataset))

    torch.save(net.to('cpu').state_dict(), model_path)
    
    #以下グラフ描画
    plt.figure(figsize=(6,6))      #グラフ描画用
    plt.plot(range(EPOCH), train_loss_value)
    plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.ylim(0, 2.5)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'test loss'])
    plt.title('loss')
    plt.savefig("loss_image.png")
    plt.clf()
else :
    net.load_state_dict(torch.load(model_path))

val_data, val_target = val_dataset[-10:]

net.eval()

result = net.forward(val_data.view(10,7,6,5))#shold be batch
print("data")
print(val_data.to('cpu').detach().numpy())
print("result")
print(result.to('cpu').detach().numpy()/10.0+1.0)


val_data_ori, val_target_ori = val_dataset_ori[-1]
print("data")
print(val_data_ori.to('cpu').detach().numpy())
print("result")
print(val_target_ori.to('cpu').detach().numpy())

#target[10(high/open)] 
