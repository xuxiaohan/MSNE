from scipy import stats
import pandas as pd
import numpy as np

def norm2distance(df):
    temp = (df ** 2).sum(axis=1, keepdims=True).repeat(df.shape[0], axis=1)
    distance = temp + temp.T - 2 * df @ df.T
    return distance

def similarity(df,k=20,sig=0.5,ktop=True,high=True):
    assert isinstance(df, pd.DataFrame)
    samples=df.index
    df=df.values
    distance=norm2distance(df)
    distance=(distance+distance.T)/2
    knear = distance.argsort()[:,1:k+1]
    x=np.repeat(np.arange(distance.shape[0]).reshape(-1,1),k,axis=1)
    delta0=distance[x,knear].mean(axis=1)
    delta=delta0.reshape(-1,1).repeat(distance.shape[0],axis=1)
    delta=sig*(delta+delta.T+distance)/3
    s=stats.norm(0,delta).pdf(distance)
    if(ktop==True):
        ag = s.argsort()[:, :-(k + 1)]
        x = np.repeat(np.arange(ag.shape[0]).reshape(-1, 1), ag.shape[1], axis=1)
        s[x, ag] = 0
    s = (s + s.T) / 2
    s=normalize(s)
    if(high):
        s=high_order(s)
    s=pd.DataFrame(s,index=samples,columns=samples)
    return s

def normalize(s):
    ispd=isinstance(s,pd.DataFrame)
    if(ispd):
        index=s.index
        col=s.columns
        s=s.values
    s=s/s.sum(axis=1).reshape(-1,1)
    if(ispd):
        s=pd.DataFrame(s,index=index,columns=col)
    return (s+s.T)/2

def high_order(s,n=5):
    for i in range(n):
        s=s@s
    s=(s+s.T)/2
    return normalize(s)

def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
