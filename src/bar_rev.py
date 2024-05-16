import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel
from darts.models import NBEATSModel
from darts.models import BlockRNNModel
from darts.metrics import mse
from matplotlib.gridspec import GridSpec
import torch
import timeit

H = 20 # horizon
L = 34 # look-back

model_LSTM = BlockRNNModel(model='LSTM',
                           input_chunk_length=L,output_chunk_length=H)
model_NBEATS = NBEATSModel(input_chunk_length=L,output_chunk_length=H)
model_TiDE = TiDEModel(input_chunk_length=L,output_chunk_length=H)

# choice of model
model = model_TiDE

file = '../data/jln.dat' # input

df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
df.iloc[:,:] = df[::-1] # invert time

ts_i = TimeSeries.from_dataframe(df,value_cols='WG5')
ts_o = TimeSeries.from_dataframe(df,value_cols='WG3')

scaler_i, scaler_o = Scaler(), Scaler()
i_s = scaler_i.fit_transform(ts_i)
o_s = scaler_o.fit_transform(ts_o)

i_s_train,i_s_test = i_s.split_after(0.667)
o_s_train,o_s_test = o_s.split_after(0.667)

tic = timeit.default_timer()
model.fit(o_s_train, past_covariates=i_s, epochs=20)
toc = timeit.default_timer()
train = toc-tic

tic = timeit.default_timer()
pred = model.predict(n=len(o_s_test), series=o_s_train, past_covariates=i_s)
toc = timeit.default_timer()
tpred = toc-tic

pred_error = np.sqrt(mse(scaler_o.inverse_transform(o_s_test),
                         scaler_o.inverse_transform(pred)))
print(np.max(np.abs(scaler_o.inverse_transform(o_s_test).values()-scaler_o.inverse_transform(pred).values())))

df.loc[pred.time_index,
       'pred']=scaler_o.inverse_transform(pred).values().flatten()

plt.rcdefaults()
fig = plt.figure(figsize=(10,6))
gs = GridSpec(2,3, figure=fig, hspace=0, width_ratios=[5,1,5])
ax2 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0],sharex=ax2)

ax1.plot(df['t'].values,df['WG5'].values,'k')
ax1.set_ylabel('WG 5 (m)')
ax1.set_title('(a) JLN',y=-.3)

ax2.plot(df['t'].values,df['WG3'].values,'k')
ax2.plot(df['t'].values,df['pred'].values,'r--')
ax1.set_xlabel('t (s)')

plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_xlim([400,430])

ax2.set_ylabel('WG 3 (m)')

print(train,'s, training')
print(tpred,'s, prediction')
print('Error: ',pred_error)

###

ax4 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,2],sharex=ax4)

# choice of model
model = model_TiDE

file = '../data/jlp.dat' # input

df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
df.iloc[:,:] = df[::-1] # invert time

ts_i = TimeSeries.from_dataframe(df,value_cols='WG5')
ts_o = TimeSeries.from_dataframe(df,value_cols='WG3')

scaler_i, scaler_o = Scaler(), Scaler()
i_s = scaler_i.fit_transform(ts_i)
o_s = scaler_o.fit_transform(ts_o)

i_s_train,i_s_test = i_s.split_after(0.667)
o_s_train,o_s_test = o_s.split_after(0.667)

tic = timeit.default_timer()
model.fit(o_s_train, past_covariates=i_s, epochs=20)
toc = timeit.default_timer()
train = toc-tic

tic = timeit.default_timer()
pred = model.predict(n=len(o_s_test), series=o_s_train, past_covariates=i_s)
toc = timeit.default_timer()
tpred = toc-tic

pred_error = np.sqrt(mse(scaler_o.inverse_transform(o_s_test),
                         scaler_o.inverse_transform(pred)))
print(np.max(np.abs(scaler_o.inverse_transform(o_s_test).values()-scaler_o.inverse_transform(pred).values())))

df.loc[pred.time_index,
       'pred']=scaler_o.inverse_transform(pred).values().flatten()

ax3.plot(df['t'].values,df['WG5'].values,'k')
ax3.set_title('(b) JLP',y=-.3)

ax4.plot(df['t'].values,df['WG3'].values,'k')
ax4.plot(df['t'].values,df['pred'].values,'r--')
ax3.set_xlabel('t (s)')

plt.setp(ax4.get_xticklabels(), visible=False)
ax4.set_xlim([400,430])

ax2.legend(['Observed','Prediction'],loc='upper right')

ax1.set_ylim([-0.025,0.055])
ax2.set_ylim([-0.025,0.055])
ax3.set_ylim([-0.025,0.055])
ax4.set_ylim([-0.025,0.055])

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

plt.savefig('../../figs/bar_rev.png')
plt.show()

print(train,'s, training')
print(tpred,'s, prediction')
print('Error: ',pred_error)
