from platform import python_version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading data
df_data = pd.read_csv("../data/04_cricket_1999to2011.csv")
# Data preprocessing
df = df_data[df_data["Innings"] == 1]
df = df.drop(['Run.Rate',"Innings",'Date', 'Outs.Remaining', 'Total.Out',
       'Innings.Run.Rate', 'Run.Rate.Required', 'Initial.Run.Rate.Required',
       'Target.Score', 'Day-night', 'At.Bat', 'Fielding', 'Home.Team',
       'Away.Team', 'Stadium', 'Country', 'Total.Overs', 'Winning.Team',
       'Toss.Winner', 'at.bat.wins', 'at.bat.won.toss', 'at.bat.at.home',
       'at.bat.bat.first', 'chose_bat_1st', 'chose_bat_2nd', 'forced_bat_1st',
       'forced_bat_2nd', 'new.game', 'Error.In.Data', 'common.support'], axis=1)

df_group = df.groupby(['Wickets.in.Hand', 'Over'])

data = []
list_group_data = []
for name, group in df_group:
    data.append([len(group),name[0], 50-name[1], group.mean(axis=0)["Runs.Remaining"]])
    list_group_data.append((group))

X_train = []
for i in range(len(list_group_data)):
    for j,ele in enumerate(list(list_group_data[i]["Runs.Remaining"])):
        X_train.append([data[i][1], data[i][2], ele])
X_train = np.array(X_train)

new_data = []
for i in X_train:
    if i[0]!=0:
        new_data.append(i)
new_data = np.array(new_data)

x_train = new_data[:,0:2]
y_train = new_data[:,2]
x_train = x_train.astype(int)

# Optimization 
def loss_fun(p,t,y):
    return p[t[:,0]]*(1-np.exp(-1*p[0]*t[:,1]/p[t[:,0]]))-y

from scipy.optimize import least_squares
p0 = [5, 10,20,40,80,110,140,180,200,230, 270]
res_1 = least_squares(loss_fun, p0, args=(x_train,y_train))

print("L is ",res_1.x[0])
print("Z(0) is ",res_1.x[1:])
print("Loss is ",2*res_1.cost/len(y_train))

optimal_parameter = res_1.x
def z(over_remain,z0,l):
    return z0*(1-np.exp(-1*l*over_remain/z0))

over_remain2 = np.arange(start=0, stop=51, step=1)
wicket_remain = np.arange(start=1, stop=11, step=1)
plt.figure(figsize=(6,4), dpi=150)
for i in range(10,0,-1):
    plt.plot(50-over_remain2, z(over_remain2, optimal_parameter[i], optimal_parameter[0]))
right_top  =z(50,optimal_parameter[10],optimal_parameter[0])
plt.xlabel("Overs Used")
plt.ylabel("Average Runs Obtainable")
plt.legend(wicket_remain[::-1])
plt.title("Over used plot")
plt.show()
