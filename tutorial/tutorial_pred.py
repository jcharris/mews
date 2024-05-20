import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.optimize import fsolve
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel

H = 14 # horizon
L = 38 # look-back
dx = 2

# choice of model
model_TiDE = TiDEModel(input_chunk_length=L,output_chunk_length=H)
model = model_TiDE

# read saved data
df = pd.read_csv('../data/tutorial_data.csv')

# convert to "TimeSeries"
ts_i = TimeSeries.from_dataframe(df,value_cols='z0')
ts_o = TimeSeries.from_dataframe(df,value_cols='z1')

# scale inputs and outputs to 0-1 range
scaler_i, scaler_o = Scaler(), Scaler()
i_s = scaler_i.fit_transform(ts_i)
o_s = scaler_o.fit_transform(ts_o)

# split data into training and test
i_s_train,i_s_test = i_s.split_after(0.667)
o_s_train,o_s_test = o_s.split_after(0.667)

# train model
model.fit(o_s_train, past_covariates=i_s, epochs=20)

# get prediction
#o_s_train = o_s_train * 0 # zero out history
pred = model.predict(n=len(o_s_test), series=o_s_train, past_covariates=i_s)

# store result in dataframe
df.loc[pred.time_index,
       'pred']=scaler_o.inverse_transform(pred).values().flatten()

# plot results
plt.rcdefaults()
fig = plt.figure(figsize=(10,6))
gs = GridSpec(2,1, figure=fig)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0],sharex=ax1)

ax1.plot(df['t'].values,df['z0'].values,'k')
ax1.set_ylabel('z (m)')
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.plot(df['t'].values,df['z1'].values,'k')
ax2.set_ylabel('z (m)')
ax2.plot(df['t'].values,df['pred'].values,'r--')
ax2.set_xlabel('t (s)')
ax2.legend(['Observed','Prediction'],loc='upper left')

ax1.grid()
ax2.grid()
ax1.set_title('x = 0m')
ax2.set_title('x = '+str(dx)+'m')
ax2.set_xlim([np.min(df['t'].values),np.max(df['t'].values)])

plt.show()
