import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel
from darts.models import NBEATSModel
from darts.models import BlockRNNModel
from darts.metrics import mse
import torch
import timeit

fig = plt.figure(figsize=(7,4))
gs = GridSpec(2, 2, figure=fig,hspace=0)

file = '../data/sln.dat'
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m

ax1 = fig.add_subplot(gs[0,0])
ax1.plot(np.array(df['t']),np.array(df.iloc[:,1]),'k')
ax1.text(.15, .85, 'WG1', ha='left', va='top', transform=ax1.transAxes)
ax1.set_xlim([225,233]); ax1.set_ylim([-0.025,0.045])
ax1.set_ylabel(r'$\eta$ (m)'); ax1.grid()

ax2 = fig.add_subplot(gs[0,1])
ax2.plot(np.array(df['t']),np.array(df.iloc[:,3]),'k')
ax2.text(.15, .85, 'WG3', ha='left', va='top', transform=ax2.transAxes)
ax2.set_xlim([225,233]); ax2.set_ylim([-0.025,0.045])
ax2.grid()

ax3 = fig.add_subplot(gs[1,0])
ax3.plot(np.array(df['t']),np.array(df.iloc[:,5]),'k')
ax3.text(.15, .85, 'WG5', ha='left', va='top', transform=ax3.transAxes)
ax3.set_xlim([225,233]); ax3.set_ylim([-0.025,0.045])
ax3.set_ylabel(r'$\eta$ (m)'); ax3.set_xlabel('t (s)') ; ax3.grid()

ax4 = fig.add_subplot(gs[1,1])
ax4.plot(np.array(df['t']),np.array(df.iloc[:,7]),'k')
ax4.text(.15, .85, 'WG7', ha='left', va='top', transform=ax4.transAxes)
ax4.set_xlim([225,233]); ax4.set_ylim([-0.025,0.045])
ax4.set_xlabel('t (s)') ; ax4.grid()

H = 41 ; L = 116
model_TiDE = TiDEModel(input_chunk_length=L,output_chunk_length=H)
model = model_TiDE
file = '../data/jln.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
ts_i = TimeSeries.from_dataframe(df,value_cols='WG1')
ts_o = TimeSeries.from_dataframe(df,value_cols='WG3')
scaler_i, scaler_o = Scaler(), Scaler()
i_s = scaler_i.fit_transform(ts_i)
o_s = scaler_o.fit_transform(ts_o)
i_s_train,i_s_test = i_s.split_after(0.667)
o_s_train,o_s_test = o_s.split_after(0.667)
model.fit(o_s_train, past_covariates=i_s, epochs=20)

file = '../data/sln.dat' # input
df2 = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df2 = df2.rename(columns = {0:'t',
                            1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                            5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df2.iloc[:,1:] = 0.01*df2.iloc[:,1:] # convert cm to m
ts_i2 = TimeSeries.from_dataframe(df2,value_cols='WG1')
ts_o2 = TimeSeries.from_dataframe(df2,value_cols='WG3')*0
i2_s = scaler_i.transform(ts_i2)
o2_s = scaler_o.transform(ts_o2)
i2_s_train,i2_s_test = i2_s.split_after(0.667)
o2_s_train,o2_s_test = o2_s.split_after(0.667)
pred = model.predict(n=len(o2_s_test), series=o2_s_train, past_covariates=i2_s)
df2.loc[pred.time_index,
        'pred']=scaler_o.inverse_transform(pred).values().flatten()

ax2.plot(np.array(df2['t']),np.array(df2['pred']),'r--')
        
H = 61 ; L = 150
model_TiDE = TiDEModel(input_chunk_length=L,output_chunk_length=H)
model = model_TiDE
ts_i = TimeSeries.from_dataframe(df,value_cols='WG1')
ts_o = TimeSeries.from_dataframe(df,value_cols='WG5')
scaler_i, scaler_o = Scaler(), Scaler()
i_s = scaler_i.fit_transform(ts_i)
o_s = scaler_o.fit_transform(ts_o)
i_s_train,i_s_test = i_s.split_after(0.667)
o_s_train,o_s_test = o_s.split_after(0.667)
model.fit(o_s_train, past_covariates=i_s, epochs=20)

ts_i2 = TimeSeries.from_dataframe(df2,value_cols='WG1')
ts_o2 = TimeSeries.from_dataframe(df2,value_cols='WG5')*0
i2_s = scaler_i.transform(ts_i2)
o2_s = scaler_o.transform(ts_o2)
i2_s_train,i2_s_test = i2_s.split_after(0.667)
o2_s_train,o2_s_test = o2_s.split_after(0.667)
pred = model.predict(n=len(o2_s_test), series=o2_s_train, past_covariates=i2_s)
df2.loc[pred.time_index,
        'pred']=scaler_o.inverse_transform(pred).values().flatten()

ax3.plot(np.array(df2['t']),np.array(df2['pred']),'r--')
        
H = 76 ; L = 187
model_TiDE = TiDEModel(input_chunk_length=L,output_chunk_length=H)
model = model_TiDE
ts_i = TimeSeries.from_dataframe(df,value_cols='WG1')
ts_o = TimeSeries.from_dataframe(df,value_cols='WG7')
scaler_i, scaler_o = Scaler(), Scaler()
i_s = scaler_i.fit_transform(ts_i)
o_s = scaler_o.fit_transform(ts_o)
i_s_train,i_s_test = i_s.split_after(0.667)
o_s_train,o_s_test = o_s.split_after(0.667)
model.fit(o_s_train, past_covariates=i_s, epochs=20)

ts_i2 = TimeSeries.from_dataframe(df2,value_cols='WG1')
ts_o2 = TimeSeries.from_dataframe(df2,value_cols='WG7')*0
i2_s = scaler_i.transform(ts_i2)
o2_s = scaler_o.transform(ts_o2)
i2_s_train,i2_s_test = i2_s.split_after(0.667)
o2_s_train,o2_s_test = o2_s.split_after(0.667)
pred = model.predict(n=len(o2_s_test), series=o2_s_train, past_covariates=i2_s)
df2.loc[pred.time_index,
        'pred']=scaler_o.inverse_transform(pred).values().flatten()

ax4.plot(np.array(df2['t']),np.array(df2['pred']),'r--')
plt.rcdefaults()
plt.savefig('../../figs/bar_irr_reg.png')
plt.show()
