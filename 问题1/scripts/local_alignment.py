"""Reference-complete semi-global edit alignment; only outer hypothesis words are free."""
import numpy as np

def reference_in_hypothesis(reference,hypothesis):
    n,m=len(reference),len(hypothesis)
    if not n:return {'status':'empty_reference','error_rate':None,'pairs':[],'hypothesis_span':None,'tied_endpoints':[]}
    d=np.zeros((n+1,m+1),dtype=np.int32);d[:,0]=np.arange(n+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i,j]=min(d[i-1,j-1]+(reference[i-1]!=hypothesis[j-1]),d[i-1,j]+1,d[i,j-1]+1)
    ends=np.flatnonzero(d[n]==d[n].min()).tolist();j=ends[0];end=j;i=n;pairs=[]
    while i:
        if j and d[i,j]==d[i-1,j-1]+(reference[i-1]!=hypothesis[j-1]):
            if reference[i-1]==hypothesis[j-1]:pairs.append((i-1,j-1))
            i-=1;j-=1
        elif d[i,j]==d[i-1,j]+1:i-=1
        else:j-=1
    return {'status':'candidate','error_rate':float(d[n,end]/n),'pairs':list(reversed(pairs)),
            'hypothesis_span':[j,end],'tied_endpoints':ends,'note':'Earliest optimal end, diagonal-first traceback; ties are not resolved speech identity.'}
