import re,numpy as np
def tok(s):return re.findall(r"[a-z0-9]+(?:'[a-z]+)*",s.lower().replace('’',"'").replace('‘',"'"))
def match(ref,hyp):
 m,n=len(ref),len(hyp);d=np.zeros((m+1,n+1),dtype=np.int32);d[:,0]=np.arange(m+1)
 for i in range(1,m+1):
  for j in range(1,n+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(ref[i-1]!=hyp[j-1]))
 j=int(np.argmin(d[-1]));cost=int(d[-1,j]);end=j;i=m;pairs=[]
 while i:
  if j and d[i,j]==d[i-1,j-1]+(ref[i-1]!=hyp[j-1]):
   if ref[i-1]==hyp[j-1]:pairs.append((i-1,j-1))
   i-=1;j-=1
  elif d[i,j]==d[i-1,j]+1:i-=1
  else:j-=1
 return cost,list(reversed(pairs)),j,end
