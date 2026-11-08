exec(open('pertype.py').read().split("types=sorted")[0])
B=load(M+'blend_val.csv'); BB=np.array([B[q] for q in ids])
allm=H>=0
def mix(pw,size): P=np.column_stack([1-pw,pw[:,None]*size]); return P/P.sum(1,keepdims=True)
tw=1-TT[:,0]; ts=TT[:,1:]/np.maximum(TT[:,1:].sum(1,keepdims=True),1e-12)
bw=1-BB[:,0]; bs=BB[:,1:]/BB[:,1:].sum(1,keepdims=True)
print('blend          %.3f'%pts(BB,allm))
print('true win+our size %.3f'%pts(mix(tw,bs),allm))
print('our win+true size %.3f'%pts(mix(bw,ts),allm))
for c in sorted(set(C)):
  m=C==c
  print(c,'win-part %.5f size-part %.5f'%(((mix(tw,bs)-TT)[m]**2).sum(1).mean(),((mix(bw,ts)-TT)[m]**2).sum(1).mean()))
