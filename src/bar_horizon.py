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

Hi = [1,5,20]
Li = [5,15,25,45,65]

NH = len(Hi)
NL = len(Li)

E = np.zeros((NL,NH)) # error
T = np.zeros((NL,NH)) # training time
P = np.zeros((NL,NH)) # prediction time
MAE = np.zeros((NL,NH))
MSE = np.zeros((NL,NH))

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

for i in range(NH):
    for j in range(NL):
        H = Hi[i]
        L = Li[j]
        model_TiDE = TiDEModel(input_chunk_length=L,output_chunk_length=H)
        model = model_TiDE
        tic = timeit.default_timer()
        model.fit(o_s_train, past_covariates=i_s, epochs=20)
        toc = timeit.default_timer()
        T[j,i] = toc-tic
        tic = timeit.default_timer()
        pred = model.predict(n=len(o_s_test),
                             series=o_s_train, past_covariates=i_s)
        toc = timeit.default_timer()
        P[j,i] = toc-tic
        E[j,i] = np.sqrt(mse(scaler_o.inverse_transform(o_s_test),
                             scaler_o.inverse_transform(pred)))

plt.rcdefaults()
fig = plt.figure(figsize=(10,3))
gs = GridSpec(1,5, figure=fig, width_ratios=[5,1,5,1,5])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[0,4])

ax1.plot(Li,T[:,0],'k.-')
ax1.plot(Li,T[:,1],'k.--')
ax1.plot(Li,T[:,2],'k.-.')
ax1.set_xlabel('L')
ax1.set_ylabel('Training time (s)')
ax1.axis([0,80,0,100])

ax2.plot(Li,P[:,0],'k.-')
ax2.plot(Li,P[:,1],'k.--')
ax2.plot(Li,P[:,2],'k.-.')
ax2.set_xlabel('L')
ax2.set_ylabel('Prediction time (s)')
ax2.axis([0,80,0,5])

ax3.plot(Li,E[:,0],'k.-')
ax3.plot(Li,E[:,1],'k.--')
ax3.plot(Li,E[:,2],'k.-.')
ax3.set_xlabel('L')
ax3.set_ylabel('RMS error (m)')
ax3.axis([0,80,0,0.01])

ax3.legend(['H=1','H=10','H=20'])

ax1.grid()
ax2.grid()
ax3.grid()
fig.set_tight_layout(True)
plt.savefig('../../figs/bar_horizon.png')
plt.show()

