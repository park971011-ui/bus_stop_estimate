import pandas as pd, numpy as np
L=pd.read_parquet('experiments/E73_loro.parquet')
pt=pd.read_csv('data/practice/test.csv',low_memory=False); ptr=pd.read_csv('data/practice/train.csv',low_memory=False)
SEGK=['route_id','direction','from_stop','to_stop']
# turn segments from practice train (non-Sunday, like turn_of on fit data)
med=ptr[ptr.direction==0].groupby(SEGK).tt_sec.median().reset_index()
turn=med.loc[med.groupby('route_id').tt_sec.idxmax(),SEGK].assign(is_turn2=1)
d=L.merge(pt,on='row_id').merge(turn,on=SEGK,how='left'); d['is_turn2']=d.is_turn2.fillna(0)
print(d[d.is_turn2==1].groupby(['route','stop_seq']).size())
def l1k(y,p):  # best multiplier under L1: weighted median of y/p with weights p
    r=y/p; o=np.argsort(r); c=np.cumsum(p[o]); return r[o][np.searchsorted(c,c[-1]/2)]
base=(d.y-d.PX).abs().mean()
for shrink in [1.0,0.75,0.5]:
    out=d.PX.copy(); ks={}
    for r in d.route.unique():
        o=d[(d.route!=r)&(d.is_turn2==1)]
        k=1+shrink*(l1k(o.y.values,o.PX.values)-1); ks[r]=round(k,3)
        m=(d.route==r)&(d.is_turn2==1); out[m]=d.PX[m]*k
    new=(d.y-out).abs()
    per={r:round((d.y-d.PX).abs()[d.route==r].mean()-new[d.route==r].mean(),3) for r in sorted(d.route.unique())}
    print('shrink',shrink,'k',ks,'Sunday MAE',round(base,3),'->',round(new.mean(),3),'gain per route',per)
print('--- non-turn rows by predicted size: L1 multiplier, per route')
nt=d[d.is_turn2==0].copy(); nt['b']=pd.cut(nt.PX,[0,60,120,200,300,2000])
print(nt.groupby(['b','route']).apply(lambda g: pd.Series({'n':len(g),'k':l1k(g.y.values,g.PX.values)})).unstack('route').round(3))
# second-largest-median segment per route (other long waits)
med2=ptr[ptr.direction==0].groupby(SEGK).tt_sec.median().reset_index().sort_values('tt_sec',ascending=False)
print(med2.groupby('route_id').head(3).merge(ptr[['route_id','route_nm']].drop_duplicates()).sort_values('route_nm'))
