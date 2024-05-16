import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fdir = '../data/'
fn = ['FH01_05.dat','FH02_04.dat','FH03_03.dat','FH04_02.dat','FH05_02.dat',
      'FH06_03.dat','FH07_03.dat','FH08_02.dat','FH09_02.dat','FH10_04.dat']

df = pd.read_csv(fdir+fn[2], delimiter='\s+', header=None,
                 usecols=[0,4], names=['t', 'eta'])

# 0.0859 / 4 / np.std(df['eta'].values)

df['eta']=0.0282*df['eta'] # storm 3
df['eta']=df['eta']-df['eta'].mean()

df_a = df[0:1560001]
df_a.to_csv('storm3a.Elev',sep=' ',index=False)

df_a = df[1380000:2940001]
df.loc[:,'t'] = df['t']-1380000*.005
df_a.to_csv('storm3b.Elev',sep=' ',index=False)

df_a = df[2820000:4380001]
df.loc[:,'t'] = df['t']-(2820000-1380000)*.005
df_a.to_csv('storm3c.Elev',sep=' ',index=False)

Tp = 1.014 # storm 3
WvLowCOff = 2*np.pi/Tp/4
WvHiCOff = 2*np.pi/Tp*4
