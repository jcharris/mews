import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel
from darts.models import NBEATSModel
from darts.models import BlockRNNModel
from darts.metrics import mse,mae
from matplotlib.gridspec import GridSpec
import torch
import timeit

TT = [10,30,60,300,600,1200,1800]

NT = len(TT)

E = np.zeros((NT,)) # error
T = np.zeros((NT,)) # training time
P = np.zeros((NT,)) # prediction time
MAE = np.zeros((NT,))
MSE = np.zeros((NT,))

H = 20 # horizon
L = 34 # look-back

file = '../data/jln.dat' # input
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})
df.iloc[:,1:] = 0.01*df.iloc[:,1:] # convert cm to m
ts_i = TimeSeries.from_dataframe(df,value_cols='WG3')
ts_o = TimeSeries.from_dataframe(df,value_cols='WG5')
scaler_i, scaler_o = Scaler(), Scaler()
i_s = scaler_i.fit_transform(ts_i)
o_s = scaler_o.fit_transform(ts_o)
i_s_train,i_s_test = i_s.split_after(0.667)
o_s_train,o_s_test = o_s.split_after(0.667)

model_TiDE = TiDEModel(input_chunk_length=L,output_chunk_length=H)
model = model_TiDE

for i in range(NT):
    tt = TT[i]
    i_s_train0,i_s_train1 = i_s_train.split_after(tt/1800)
    o_s_train0,o_s_train1 = o_s_train.split_after(tt/1800)
    tic = timeit.default_timer()
    model.fit(o_s_train0, past_covariates=i_s, epochs=20)
    toc = timeit.default_timer()
    T[i] = toc-tic
    tic = timeit.default_timer()
    pred = model.predict(n=len(o_s_test),
                         series=o_s_train, past_covariates=i_s)
    toc = timeit.default_timer()
    P[i] = toc-tic
    E[i] = np.sqrt(mse(scaler_o.inverse_transform(o_s_test),
                       scaler_o.inverse_transform(pred)))

plt.rcdefaults()
fig = plt.figure(figsize=(10,3))
gs = GridSpec(1,5, figure=fig, width_ratios=[3,1,3,1,3])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[0,4])

ax1.loglog(np.array(TT)/2.5,T,'k.-')
ax1.set_xlabel(r'$t_{train} / T_p$')
ax1.set_ylabel('Training time (s)')
ax1.axis([1,1000,.2,200])

ax2.loglog(np.array(TT)/2.5,P,'k.-')
ax2.set_xlabel(r'$t_{train} / T_p$')
ax2.set_ylabel('Prediction time (s)')
ax2.axis([1,1000,.1,5])

ax3.loglog(np.array(TT)/2.5,E,'k.-')
ax3.set_xlabel(r'$t_{train} / T_p$')
ax3.set_ylabel('RMS error (m)')
ax3.axis([1,1000,1e-4,1e-2])

ax1.grid()
ax2.grid()
ax3.grid()
fig.set_tight_layout(True)
plt.savefig('../../figs/bar_train.png')
plt.show()
