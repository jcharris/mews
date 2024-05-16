import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel
from darts.models import NBEATSModel
from darts.models import BlockRNNModel
from darts.metrics import mse
import torch
import timeit
from darts import concatenate
from matplotlib.gridspec import GridSpec

H = 10 # horizon
L = 400 # lookback

#model = BlockRNNModel(model='LSTM',input_chunk_length=L,output_chunk_length=H)
#model = NBEATSModel(input_chunk_length = L,output_chunk_length = H,kwargs={'loss':HH_loss})
model = TiDEModel(input_chunk_length=L,output_chunk_length=H)

#fn = '../data/FH01_05.dat'
#fn = '../data/FH02_04.dat'
fn = '../data/FH03_03.dat'
df0 = pd.read_csv(fn, delimiter='\s+', header=None,
                 usecols=[0,3,4,6,8], names=['t', 'up', 'eta', 'F1', 'F2'])
#                 usecols=[0,4,6,8], names=['t', 'eta', 'F1', 'F2'])

df = df0

df['up']=0.0282*df.loc[:,'up'] # storm 3
df['up']=df.loc[:,'up']-df['up'].mean()

df['eta']=0.0282*df.loc[:,'eta'] # storm 3
df['eta']=df.loc[:,'eta']-df['eta'].mean()

df['F1']=1.254*df['F1']
df['F2']=1.101*df['F2']
df['F1']=df['F1']-df['F1'].mean()
df['F2']=df['F2']-df['F2'].mean()

df['Fx']=df['F1']+df['F2']
df['My']=df['F1']*0.850

df.loc[:,'eta_shift']=df['eta'].shift(-200)
df = df[:-200]

#df['Fx'] = df['Fx'] + np.gradient(np.gradient(np.array(df['Fx']),.005),.005).reshape(len(df),)/3e4

#ts_i0 = TimeSeries.from_dataframe(df,value_cols='up')
#ts_i1 = TimeSeries.from_dataframe(df,value_cols='eta_shift')
#ts_i = concatenate([ts_i0,ts_i1],axis=1)

#ts_i = TimeSeries.from_dataframe(df,value_cols='up')
ts_i = TimeSeries.from_dataframe(df,value_cols='eta_shift')

ts_o0 = TimeSeries.from_dataframe(df,value_cols='Fx')
ts_o1 = TimeSeries.from_dataframe(df,value_cols='My')
ts_o = concatenate([ts_o0,ts_o1],axis=1)

#ts_o = TimeSeries.from_dataframe(df,value_cols='Fx')

scaler_i, scaler_o = Scaler(), Scaler()
i_s = scaler_i.fit_transform(ts_i)
o_s = scaler_o.fit_transform(ts_o)

i_s_train,i_s_test = i_s.split_after(0.2)
o_s_train,o_s_test = o_s.split_after(0.2)

tic = timeit.default_timer()
model.fit(o_s_train, past_covariates=i_s, epochs=3)
toc = timeit.default_timer()
train = toc-tic

#from pickle import dump
#dump([scaler_i,scaler_o], open('../models/wsi_scalers_TiDE.pkl','wb'))
#model.save('../models/wsi_TiDE.pt')

tic = timeit.default_timer()
pred = model.predict(n=len(o_s_test), series=o_s_train, past_covariates=i_s)
toc = timeit.default_timer()
tpred = toc-tic

pred_error = np.sqrt(mse(scaler_o.inverse_transform(o_s_test),
                         scaler_o.inverse_transform(pred)))
print("Error: ",pred_error)

df['pred']=np.NaN
df['pred'][pred.time_index]=scaler_o.inverse_transform(pred).values()[:,0]

# had to split OpenFAST runs to save memory
dfa = pd.read_csv('driver_3a.out',delimiter='\s+',skiprows=6)
dfa = dfa[1:] ; dfa = dfa.astype(np.float64)
dfb = pd.read_csv('driver_3b.out',delimiter='\s+',skiprows=6)
dfb = dfb[1:] ; dfb = dfb.astype(np.float64)
dfc = pd.read_csv('driver_3c.out',delimiter='\s+',skiprows=6)
dfc = dfc[1:] ; dfc = dfc.astype(np.float64)

dfo = pd.DataFrame(index=np.arange(1,4320001),columns=['Time','Wave1Elev','HydroFxi','HydroMyi'])

dfo.loc[0:1440000,'Time'] = dfa.loc[0:1440000,'Time'].values
dfo.loc[0:1440000,'Wave1Elev'] = dfa.loc[0:1440000,'Wave1Elev'].values
dfo.loc[0:1440000,'HydroFxi'] = dfa.loc[0:1440000,'HydroFxi'].values
dfo.loc[0:1440000,'HydroMyi'] = dfa.loc[0:1440000,'HydroMyi'].values

dfo.loc[1440000:2880000,'Time'] = dfb.loc[60000:1500000,'Time'].values+6900
dfo.loc[1440000:2880000,'Wave1Elev'] = dfb.loc[60000:1500000,'Wave1Elev'].values
dfo.loc[1440000:2880000,'HydroFxi'] = dfb.loc[60000:1500000,'HydroFxi'].values
dfo.loc[1440000:2880000,'HydroMyi'] = dfb.loc[60000:1500000,'HydroMyi'].values

dfo.loc[2880000:4320000,'Time'] = dfc.loc[60000:1500000,'Time'].values+14100
dfo.loc[2880000:4320000,'Wave1Elev'] = dfc.loc[60000:1500000,'Wave1Elev'].values
dfo.loc[2880000:4320000,'HydroFxi'] = dfc.loc[60000:1500000,'HydroFxi'].values
dfo.loc[2880000:4320000,'HydroMyi'] = dfc.loc[60000:1500000,'HydroMyi'].values

plt.rcdefaults()
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(2,1, figure=fig,hspace=0)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0],sharex=ax1)

ax1.plot(df['t'].values,df['eta'].values,'k')
ax1.set_ylabel(r'$\eta$ (m)')
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.plot(df['t'].values,df['Fx'].values,'k')
ax2.plot(dfo['Time'].values,dfo['HydroFxi'].values,'b--')
ax2.plot(df['t'].values,df['pred'].values,'r:')
ax2.set_ylabel('Force (N)')
ax2.legend(['Observed','OpenFAST','TiDE'],loc='upper left')
ax2.annotate('Obs.', xy=(9420.3, 1.6), xytext=(9419, 1.6),
             arrowprops=dict(arrowstyle="->"),
             horizontalalignment='right',
             verticalalignment='center')
ax2.annotate('TiDE', xy=(9420.3, 1.4), xytext=(9419, 1.4),
             arrowprops=dict(arrowstyle="->"),
             horizontalalignment='right',
             verticalalignment='center')
ax2.annotate('OpenFAST', xy=(9420.3, 0.7), xytext=(9419, 0.7),
             arrowprops=dict(arrowstyle="->"),
             horizontalalignment='right',
             verticalalignment='center')
ax2.set_xlim([9414,9428])
ax2.set_xlabel('t (s)')

ax1.grid()
ax2.grid()
plt.savefig('../../figs/wsi_force.png',dpi=300)
plt.show()

fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1,3, figure=fig,width_ratios=[5,1,5])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,2])

fstd = np.std(df['Fx'])
zc = np.where(np.diff(np.sign(df['eta']))<0)[0]

fmax = np.array([np.max(df['Fx'][x[0]:x[1]])
                 for x in list(np.array([zc[:-1],zc[1:]]).transpose())])
fmin = np.array([np.min(df['Fx'][x[0]:x[1]])
                 for x in list(np.array([zc[:-1],zc[1:]]).transpose())])

omax = np.array([np.max(dfo['HydroFxi'][x[0]:x[1]])
                 for x in list(np.array([zc[:-1],zc[1:]]).transpose())])
omin = np.array([np.min(dfo['HydroFxi'][x[0]:x[1]])
                 for x in list(np.array([zc[:-1],zc[1:]]).transpose())])
ax1.plot([-10,10],[-15,15],'-k')
ax1.plot([-15,15],[-10,10],'-k')
ax1.set_xlim([-5,10]) ; ax1.set_ylim([-5,10])
ax1.plot(fmax/fstd,omax/fstd,'.k');
ax1.plot(fmin/fstd,omin/fstd,'.k');
ax1.set_xlabel(r'Measured force peaks ($F_{meas}/F_{std}$)')
ax1.set_ylabel(r'Predicted force peaks ($F_{pred}/F_{std}$)')
ax1.set_title('OpenFAST')
ax1.grid()

pmax = np.array([np.max(df['pred'][x[0]:x[1]])
                 for x in list(np.array([zc[:-1],zc[1:]]).transpose())])
pmin = np.array([np.min(df['pred'][x[0]:x[1]])
                 for x in list(np.array([zc[:-1],zc[1:]]).transpose())])
ax2.plot([-10,10],[-15,15],'-k')
ax2.plot([-15,15],[-10,10],'-k')
ax2.set_xlim([-5,10]) ; ax2.set_ylim([-5,10])
ax2.plot(fmax/fstd,pmax/fstd,'.k');
ax2.plot(fmin/fstd,pmin/fstd,'.k');
ax2.set_xlabel(r'Measured force peaks ($F_{meas}/F_{std}$)')
ax2.set_ylabel(r'Predicted force peaks ($F_{pred}/F_{std}$)')
ax2.set_title('TiDE')
ax2.grid()
ax1.set_aspect('equal','box')
ax2.set_aspect('equal','box')
ax1.set_xticks([-4,-2,0,2,4,6,8,10])
ax1.set_yticks([-4,-2,0,2,4,6,8,10])
ax2.set_xticks([-4,-2,0,2,4,6,8,10])
ax2.set_yticks([-4,-2,0,2,4,6,8,10])

plt.savefig('../../figs/wsi_peaks.png')
plt.show()

ik = ~np.isnan(fmax) ; fmax = fmax[ik] ; omax = omax[ik] ; pmax = pmax[ik]
ik = ~np.isnan(omax) ; fmax = fmax[ik] ; omax = omax[ik] ; pmax = pmax[ik]
ik = ~np.isnan(pmax) ; fmax = fmax[ik] ; omax = omax[ik] ; pmax = pmax[ik]

fig = plt.figure(figsize=(10,3))
gs = GridSpec(1,5, figure=fig, width_ratios=[5,1,5,1,5])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[0,4])

#print(len(fmax))
#print(np.max(fmax)/np.std(fmax)) # see Brouwers and Verbeek, 1983
#Fstd[i]=np.std(fmax)
ax1.semilogy(np.sort(fmax)/np.std(fmax),
             np.linspace(1,1/len(fmax),len(fmax)),'.-k')
ax2.semilogy(np.sort(omax)/np.std(fmax),
             np.linspace(1,1/len(omax),len(omax)),'.-k')
ax3.semilogy(np.sort(pmax)/np.std(fmax),
             np.linspace(1,1/len(pmax),len(pmax)),'.-k')
ax1.set_ylim([1e-5,1])
ax2.set_ylim([1e-5,1])
ax3.set_ylim([1e-5,1])
ax1.set_xlim([0,12])
ax2.set_xlim([0,12])
ax3.set_xlim([0,12])
ax1.set_title('Observed')
ax2.set_title('OpenFAST')
ax3.set_title('TiDE')
ax1.set_xlabel(r'$F_{meas}/F_{std}$')
ax2.set_xlabel(r'$F_{pred}/F_{std}$')
ax3.set_xlabel(r'$F_{pred}/F_{std}$')
ax1.set_ylabel(r'$P_{exceedance}$')
ax2.set_ylabel(r'$P_{exceedance}$')
ax3.set_ylabel(r'$P_{exceedance}$')
ax1.grid()
ax2.grid()
ax3.grid()
fig.set_tight_layout(True)
plt.savefig('../../figs/wsi_prob.png')
plt.show()
