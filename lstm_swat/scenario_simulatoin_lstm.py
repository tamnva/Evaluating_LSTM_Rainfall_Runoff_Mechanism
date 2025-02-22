# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:17:52 2025

@author: nguyenta
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.create_model import create_model
from hydroecolstm.utility.evaluation_function import EvaluationFunction
import torch

from hydroecolstm.feat_importance.perm_feat_importance import pfib
import seaborn as sns

# Setworking directory
os.chdir('C:/Users/nguyenta/Documents/Manuscript/lstm_swat')
plt.rcParams["font.family"] = "Times New Roman"

q_mm_to_m3s = 86.4/2176.453

#-----------------------------------------------------------------------------#
#                        Plot SWAT + observed streamflow                      #
#-----------------------------------------------------------------------------#
obs = pd.read_csv(Path("data/SWAT/obs_var_1.txt"), sep='\t')
swat_sim = pd.read_csv(Path("data/SWAT/workingFolder_best/TxtInOut_1/watout.dat"),
                       skiprows=6, header=None, delimiter="\s+")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5))
ax.plot(pd.to_datetime(obs["Date Time"]), obs["Qobs"], "+", markersize=2,
        alpha=0.95, label="Observed")
ax.plot(pd.to_datetime(obs["Date Time"]), swat_sim.iloc[:,3][-3652:], 
        alpha=0.65,label="Simulated (SWAT)")
ax.set_ylabel(r'Streamflow (m$^3$/s)')
plt.legend()
plt.show()


#-----------------------------------------------------------------------------#
#                                         Load models                         #
#-----------------------------------------------------------------------------#
config_lstm_q = read_config("data/LSTM/LSTM_Q/best_config.yml")
model_lstm_q = create_model(config_lstm_q)
model_lstm_q.load_state_dict(torch.load(Path(config_lstm_q["output_directory"][0],
                                      "model_state_dict.pt")))                
data_lstm_q = torch.load(Path(config_lstm_q["output_directory"][0], "data.pt"))
objective = EvaluationFunction("NSE", config_lstm_q['warmup_length'], 
                               data_lstm_q['y_column_name'])
objective(data_lstm_q['y_train'], data_lstm_q['y_train_simulated'])
objective(data_lstm_q['y_valid'], data_lstm_q['y_valid_simulated'])
nse_q_lstmq = objective(data_lstm_q['y_test'], data_lstm_q['y_test_simulated']).values


config_lstm_q_et = read_config("data/LSTM/LSTM_Q_ET/best_config.yml")
model_lstm_q_et = create_model(config_lstm_q_et)
model_lstm_q_et.load_state_dict(torch.load(Path(config_lstm_q_et["output_directory"][0],
                                      "model_state_dict.pt")))                
data_lstm_q_et = torch.load(Path(config_lstm_q_et["output_directory"][0], "data.pt"))
objective = EvaluationFunction("NSE", config_lstm_q_et['warmup_length'], 
                               data_lstm_q_et['y_column_name'])
objective(data_lstm_q_et['y_train'], data_lstm_q_et['y_train_simulated'])
objective(data_lstm_q_et['y_valid'], data_lstm_q_et['y_valid_simulated'])
nse_q_lstmqet = objective(data_lstm_q_et['y_test'], 
                          data_lstm_q_et['y_test_simulated']).values


#-----------------------------------------------------------------------------#
#                     Plot training and validation loss both models                    #
#-----------------------------------------------------------------------------#

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
# 1st axis
ax[0,0].plot(data_lstm_q["loss_epoch"].index + 1, 
             data_lstm_q["loss_epoch"]["train_loss"], "--",
        alpha=0.5, label=r'Training loss (LSTM$_Q$)')
ax[0,0].plot(data_lstm_q["loss_epoch"].index + 1, 
             data_lstm_q["loss_epoch"]["valid_loss"], 
        alpha=0.5,label=r'Validation loss (LSTM$_Q$)')

ax[0,0].plot(data_lstm_q_et["loss_epoch"].index + 1, 
             data_lstm_q_et["loss_epoch"]["train_loss"], "--",
        alpha=0.5, label=r'Training loss (LSTM$_{QET}$)')
ax[0,0].plot(data_lstm_q_et["loss_epoch"].index + 1, 
             data_lstm_q_et["loss_epoch"]["valid_loss"], 
        alpha=0.5,label=r'Validation loss (LSTM$_{QET}$)')

ax[0,0].set_ylabel("1-NSE")
ax[0,0].set_xlabel("Epoch")
ax[0,0].legend()
ax[0,0].set_title("(a) Training and validation losses")


# 2nd axis
ax[0,1].plot(data_lstm_q["time_test"]["ht_0006"], 
           data_lstm_q['y_test']["ht_0006"] * 1/q_mm_to_m3s, 
           markersize = 2, marker= "*",
           alpha=0.25, label="SWAT")
ax[0,1].plot(data_lstm_q["time_test"]["ht_0006"], 
           data_lstm_q['y_test_simulated']["ht_0006"] * 1/q_mm_to_m3s, 
           alpha=0.65, label=r'LSTM$_Q$')

ax[0,1].legend()
ax[0,1].set_ylabel(r'Streamflow (m$^3$/s)')
ax[0,1].set_title(r'(b) Streamflow SWAT - LSTM$_Q$')

# 3rd axis
ax[1,0].plot(data_lstm_q_et["time_test"]["ht_0006"], 
           data_lstm_q_et['y_test']["ht_0006"][:,0], 
           markersize = 2, marker= "*",
           alpha=0.25, label="SWAT")
ax[1,0].plot(data_lstm_q_et["time_test"]["ht_0006"], 
           data_lstm_q_et['y_test_simulated']["ht_0006"][:,0], alpha=0.65, 
           label=r'LSTM$_{QET}$')

ax[1,0].legend()
ax[1,0].set_ylabel(r'Actual ET (mm/day)')
ax[1,0].set_title(r'(c) Actual ET SWAT - LSTM$_{QET}$')


# 4zh axis
ax[1,1].plot(data_lstm_q_et["time_test"]["ht_0006"], 
           data_lstm_q_et['y_test']["ht_0006"][:,1]* 1/q_mm_to_m3s, 
           markersize = 2, marker= "*",
           alpha=0.25, label="SWAT")
ax[1,1].plot(data_lstm_q_et["time_test"]["ht_0006"], 
           data_lstm_q_et['y_test_simulated']["ht_0006"][:,1]* 1/q_mm_to_m3s, 
           alpha=0.65, label=r'LSTM$_{QET}$')

ax[1,1].legend()
ax[1,1].set_ylabel(r"Streamflow m$^3$/s")
ax[1,1].set_title(r'(d) Streamflow SWAT - LSTM$_{QET}$')

plt.subplots_adjust(hspace=0.3)
plt.subplots_adjust(wspace=0.3)
plt.show()

#-----------------------------------------------------------------------------#
#                     Random permutation both models                          #
#-----------------------------------------------------------------------------#
features_lstm_q = config_lstm_q["input_dynamic_features"]
features_lstm_q_et = config_lstm_q_et["input_dynamic_features"]
num_iter = 100

for iteration in range(num_iter):
    print(iteration)
    temp_lstm_q = pfib(features=features_lstm_q,
                x_test_scale=data_lstm_q["x_test_scale"],
                y_test=data_lstm_q["y_test"],
                y_scaler=data_lstm_q["y_scaler"], 
                trained_model=model_lstm_q, 
                objective_function_name="NSE", 
                y_column_name=data_lstm_q["y_column_name"],
                nskip=config_lstm_q["warmup_length"])
    
    temp_lstm_q_et = pfib(features=features_lstm_q_et,
                x_test_scale=data_lstm_q_et["x_test_scale"],
                y_test=data_lstm_q_et["y_test"],
                y_scaler=data_lstm_q_et["y_scaler"], 
                trained_model=model_lstm_q_et, 
                objective_function_name="NSE", 
                y_column_name=data_lstm_q_et["y_column_name"],
                nskip=config_lstm_q_et["warmup_length"])
    
    if iteration == 0:
        fib_lstm_q = temp_lstm_q
        fib_lstm_q_et = temp_lstm_q_et
    else:
        fib_lstm_q = pd.concat([fib_lstm_q, temp_lstm_q], axis=0)
        fib_lstm_q_et = pd.concat([fib_lstm_q_et, temp_lstm_q_et], axis=0)

#fib_lstm_q.to_csv("data/LSTM/fib_lstm_q.csv", index=False)
#fib_lstm_q_et.to_csv("data/LSTM/fib_lstm_q_et.csv", index=False)

fib_lstm_q = pd.read_csv("data/LSTM/fib_lstm_q.csv")
fib_lstm_q_et = pd.read_csv("data/LSTM/fib_lstm_q_et.csv")

# Need to run SWAT first
nse_swat = pd.read_csv("data/SWAT/workingFolder_best/random_permutation_nse.csv")
nse_swat.columns = ["rainfall_Q", "rainfall_ET", "t_min_Q", 
                    "t_min_ET", "t_max_Q", "t_max_ET"]


fig_q = nse_swat * 0
fig_q["rainfall_Q"] = fib_lstm_q["precip_mm_NSE_q_mm"].values - nse_q_lstmq[0]
fig_q["t_min_Q"] = fib_lstm_q["t_min_NSE_q_mm"].values -  nse_q_lstmq[0]
fig_q["t_max_Q"] = fib_lstm_q["t_max_NSE_q_mm"].values -  nse_q_lstmq[0]


fig_qet = nse_swat * 0
fig_qet["rainfall_Q"] = fib_lstm_q_et["precip_mm_NSE_q_mm"].values - nse_q_lstmqet.flatten()[1]
fig_qet["t_min_Q"] = fib_lstm_q_et["t_min_NSE_q_mm"].values -  nse_q_lstmqet.flatten()[1]
fig_qet["t_max_Q"] = fib_lstm_q_et["t_max_NSE_q_mm"].values -  nse_q_lstmqet.flatten()[1]
fig_qet["rainfall_ET"] = fib_lstm_q_et["precip_mm_NSE_eta_mm"].values - nse_q_lstmqet.flatten()[0]
fig_qet["t_min_ET"] = fib_lstm_q_et["t_min_NSE_eta_mm"].values -  nse_q_lstmqet.flatten()[0]
fig_qet["t_max_ET"] = fib_lstm_q_et["t_max_NSE_eta_mm"].values -  nse_q_lstmqet.flatten()[0]


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,5))
sns.boxplot(data=nse_swat-1, orient="h", palette="pastel", ax=ax[0,0], fliersize=3)
ax[0,0].set_xlim([-2.5,0.25])
ax[0,0].set_xlabel('ΔNSE')
#ax[0,0].axvline(x=0, linewidth=0.5)
ax[0,0].set_title('(a) SWAT')
sns.boxplot(data=fig_q , orient="h", palette="pastel", ax=ax[0,1], fliersize=3)
ax[0,1].set_xlim([-2.5,0.25])
ax[0,1].set_xlabel('ΔNSE')
#ax[0,1].axvline(x=0, linewidth=0.5)
ax[0,1].set_title(r'(b) LSTM$_{Q}$')
sns.boxplot(data=fig_qet , orient="h", palette="pastel", ax=ax[1,1], fliersize=3)
ax[1,1].set_xlim([-2.5,0.25])
ax[1,1].set_xlabel('ΔNSE')
#ax[1,1].axvline(x=0, linewidth=0.5)
ax[1,1].set_title(r'(c) LSTM$_{QET}$')
ax[1,0].set_axis_off()
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.3)
plt.show()


#-----------------------------------------------------------------------------#
#                        Rainfall-runoff lag times                            #
#-----------------------------------------------------------------------------#
key = "ht_0006"
warm_up = config_lstm_q["warmup_length"]
lag_time = 366

x_test = data_lstm_q["x_test"][key].clone()
np.sum(x_test[:,0].numpy() < 0.0001)/x_test[:,0].shape[0]

# Note input is precipitation first so index of precipitation is 0
x_test[:,0] = torch.max(x_test, 0)[0][0]
x_test_scale = data_lstm_q["x_scaler"].transform({key: x_test})

# LSTM_Q
# Get y_test and y_test manipulate
y_test_manipulate = data_lstm_q["y_scaler"].inverse(model_lstm_q.evaluate(x_test_scale))
y_test_true = data_lstm_q["y_test_simulated"][key]
y = data_lstm_q["y_scaler"].inverse(model_lstm_q.evaluate(
    data_lstm_q["x_scaler"].transform({key: data_lstm_q["x_test"][key]})))[key]

for i in list(range(365, 731, 1)):
    print(i)
   
    x = data_lstm_q["x_test_scale"][key].clone()
    data_lstm_q["time_test"][key][365]
    
    x[i,:] = x_test_scale[key][i,:]
    y_mod = data_lstm_q["y_scaler"].inverse(model_lstm_q.evaluate({key: x}))[key]
    temp = y_mod[i:i+lag_time,:] - y[i:i+lag_time]
   
    if i == 365:
        err = temp
    else:
        err = torch.cat([err, temp],1)

err_lstm_q = err * 1/q_mm_to_m3s

# LSTM-QET
# Get y_test and y_test manipulate
y_test_manipulate = data_lstm_q_et["y_scaler"].inverse(model_lstm_q_et.evaluate(x_test_scale))
y_test_true = data_lstm_q_et["y_test_simulated"][key]
y = data_lstm_q_et["y_scaler"].inverse(model_lstm_q_et.evaluate(
    data_lstm_q_et["x_scaler"].transform({key: data_lstm_q_et["x_test"][key]})))[key]

for i in list(range(365, 731, 1)):
    print(i)
   
    x = data_lstm_q_et["x_test_scale"][key].clone()
    
    x[i,:] = x_test_scale[key][i,:]
    y_mod = data_lstm_q_et["y_scaler"].inverse(model_lstm_q_et.evaluate({key: x}))[key]
    temp = y_mod[i:i+lag_time,:] - y[i:i+lag_time, :]
   
    if i == 365:
        err = temp
    else:
        err = torch.cat([err, temp],1)

err_lstm_q_et = err * 1/q_mm_to_m3s
err_lstm_q_et = err_lstm_q_et[:, np.arange(1,200,2)]


# Read from SWAT
swat = pd.read_csv("data/SWAT/workingFolder_best/runoff_timelag.csv")
time = data_lstm_q["time_test"][key][365:731]


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,2))
ax[0].fill_between(list(range(0, 365,1)),swat.min(axis=1), 
                   swat.max(axis=1), alpha = 0.25)
ax[0].plot(swat.median(axis=1), "--", linewidth = 1, label="SWAT")
ax[0].fill_between(list(range(0, 366,1)),torch.min(err_lstm_q, 1)[0].numpy(),
                torch.max(err_lstm_q, 1)[0].numpy(), alpha = 0.25)
ax[0].plot(torch.median(err_lstm_q, 1)[0].numpy(), linewidth = 1, 
           label=r'LSTM$_{Q}$')
ax[0].set_xlim(0,50)
ax[0].set_xlabel("Number of days after changing precipitation (days)")
ax[0].set_ylabel(r'ΔQ (m$^3$/s)')
ax[0].legend()
ax[0].set_title(r"(a) SWAT and LSTM$_{Q}$")

ax[1].fill_between(list(range(0, 365,1)),swat.min(axis=1), 
                   swat.max(axis=1), alpha = 0.25)
ax[1].plot(swat.median(axis=1), "--", linewidth = 1, label="SWAT")
ax[1].fill_between(list(range(0, 366,1)),torch.min(err_lstm_q_et, 1)[0].numpy(),
                torch.max(err_lstm_q_et, 1)[0].numpy(), alpha = 0.25)
ax[1].plot(torch.median(err_lstm_q_et, 1)[0].numpy(), linewidth = 1, 
           label=r'LSTM$_{QET}$')
ax[1].set_xlim(0,50)
ax[1].set_ylabel(r'ΔQ (m$^3$/s)')
ax[1].legend()
ax[1].set_title(r"(b) SWAT and LSTM$_{QET}$")
ax[1].set_xlabel("Number of days after changing precipitation (days)")
plt.subplots_adjust(wspace=0.3)
plt.show()

#-----------------------------------------------------------------------------#
#                        Precipitation is zero                                #
#-----------------------------------------------------------------------------#
key = "ht_0006"
warm_up = config_lstm_q["warmup_length"]

# LSTMQ model
x_test = data_lstm_q["x_test"][key].clone()
x_test[:,0] = 0
x_test_scale = data_lstm_q["x_scaler"].transform({key: x_test})
y_test_manipulate_lstm_q = data_lstm_q["y_scaler"].inverse(
    model_lstm_q.evaluate(x_test_scale))


x_test = data_lstm_q_et["x_test"][key].clone()
x_test[:,0] = 0
x_test_scale = data_lstm_q_et["x_scaler"].transform({key: x_test})
y_test_manipulate_lstm_q_et = data_lstm_q_et["y_scaler"].inverse(
    model_lstm_q_et.evaluate(x_test_scale))


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,2))
ax.plot([np.datetime64("2011-02-01"), 
         np.datetime64("2019-12-31")], [0, 0],
           ls='--', linewidth=2, label="SWAT")
ax.plot(data_lstm_q["time_test"][key][-3257:], 
        y_test_manipulate_lstm_q[key][-3257:] * 1/q_mm_to_m3s, 
        label=r'LSTM$_{Q}$', linewidth=0.75)
ax.plot(data_lstm_q_et["time_test"][key][-3257:], 
        y_test_manipulate_lstm_q_et[key][-3257:][:,1] * 1/q_mm_to_m3s, 
        label=r'LSTM$_{QET}$', linewidth=0.75)
ax.set_ylabel( r'Streamflow (m$^3$/s)')
ax.set_ylim([-5, 30])
ax.legend(ncol=3)
plt.show()


# Correlation between Q and Tmax
np.corrcoef(y_test_manipulate_lstm_q[key][-3257:][:,0].numpy(), 
            data_lstm_q_et["x_test"]["ht_0006"][-3257:,2].numpy())
np.corrcoef(y_test_manipulate_lstm_q_et[key][-3257:][:,1].numpy(), 
            data_lstm_q_et["x_test"]["ht_0006"][-3257:,2].numpy())