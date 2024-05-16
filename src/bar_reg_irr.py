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

scaler_i, scaler_o = Scaler(), Scaler()

file = '../data/slp.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
i_s1 = scaler_i.fit_transform(TimeSeries.from_dataframe(df,value_cols='WG3'))
o_s1 = scaler_o.fit_transform(TimeSeries.from_dataframe(df,value_cols='WG5'))

file = '../data/sls.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
i_s2 = scaler_i.transform(TimeSeries.from_dataframe(df,value_cols='WG3'))
o_s2 = scaler_o.transform(TimeSeries.from_dataframe(df,value_cols='WG5'))

file = '../data/sln.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
i_s3 = scaler_i.transform(TimeSeries.from_dataframe(df,value_cols='WG3'))
o_s3 = scaler_o.transform(TimeSeries.from_dataframe(df,value_cols='WG5'))

file = '../data/ssp.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
i_s4 = scaler_i.transform(TimeSeries.from_dataframe(df,value_cols='WG3'))
o_s4 = scaler_o.transform(TimeSeries.from_dataframe(df,value_cols='WG5'))

file = '../data/sss.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
i_s5 = scaler_i.transform(TimeSeries.from_dataframe(df,value_cols='WG3'))
o_s5 = scaler_o.transform(TimeSeries.from_dataframe(df,value_cols='WG5'))

file = '../data/ssn.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
i_s6 = scaler_i.transform(TimeSeries.from_dataframe(df,value_cols='WG3'))
o_s6 = scaler_o.transform(TimeSeries.from_dataframe(df,value_cols='WG5'))

tic = timeit.default_timer()
model.fit([o_s1,o_s2,o_s3,o_s4,o_s5,o_s6],
          past_covariates=[i_s1,i_s2,i_s3,i_s4,i_s5,i_s6], epochs=20)
toc = timeit.default_timer()
train = toc-tic

file = '../data/jln.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
i_s = scaler_i.transform(TimeSeries.from_dataframe(df,value_cols='WG3'))
o_s = scaler_o.transform(TimeSeries.from_dataframe(df,value_cols='WG5'))

i_s_train,i_s_test = i_s.split_after(0.01)
o_s_train,o_s_test = i_s.split_after(0.01)

tic = timeit.default_timer()
pred = model.predict(n=len(o_s_test), series=o_s_train, past_covariates=i_s)
toc = timeit.default_timer()
tpred = toc-tic

pred_error = mse(o_s_test,pred)

df.loc[pred.time_index,
       'pred']=scaler_o.inverse_transform(pred).values().flatten()

plt.rcdefaults()
fig = plt.figure(figsize=(5,5))
gs = GridSpec(2,1, figure=fig,hspace=0)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0],sharex=ax1)

ax1.plot(df['t'].values,df['WG3'].values,'k')
ax1.set_ylabel('WG 3 (m)')

plt.setp(ax1.get_xticklabels(), visible=False)

ax2.plot(df['t'].values,df['WG5'].values,'k')
ax2.plot(df['t'].values,df['pred'].values,'r--')
ax2.set_xlabel('t (s)')

ax2.set_xlim([2000,2030])

ax2.set_ylabel('WG 5 (m)')
ax2.legend(['Observed','Prediction'])

ax1.set_ylim([-0.025,0.055])
ax2.set_ylim([-0.025,0.055])

ax1.grid()
ax2.grid()

fig.set_tight_layout(True)
plt.savefig('../../figs/bar_reg_irr.png')
plt.show()

print(train,'s, training')
print(tpred,'s, prediction')
print('Error: ',pred_error)
