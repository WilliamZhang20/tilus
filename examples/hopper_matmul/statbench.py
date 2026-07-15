"""Rigorous: N back-to-back cublas-vs-tilus ratio samples. DCE-safe (real stores)."""
import sys, math
import numpy as np, torch, tilus, importlib
from tilus.utils import benchmark_func
tilus.option.cache_dir('./cache')
VCLS={"v5":"MatmulWGMMAV5","v6":"MatmulWGMMAV6","v7":"MatmulWGMMAV7"}
v=sys.argv[1] if len(sys.argv)>1 else "v6"
N=int(sys.argv[2]) if len(sys.argv)>2 else 15
m=n=k=8192
a=(torch.rand(m,k,dtype=torch.float16,device='cuda')-0.5)/math.sqrt(k)
b=(torch.rand(n,k,dtype=torch.float16,device='cuda')-0.5)/math.sqrt(k)
c=torch.empty(m,n,dtype=torch.float16,device='cuda'); cref=a@b.T
mm=getattr(importlib.import_module(f"matmul_{v}"),VCLS[v])()
mm(m,n,k,a,b,c); torch.cuda.synchronize()
assert (c.float()-cref.float()).abs().max().item()<0.5, "correctness fail"
pcts=[]
for i in range(N):
    cl=benchmark_func(lambda: torch.matmul(a,b.T,out=cref),warmup=3,repeat=25)
    t=benchmark_func(lambda: mm(m,n,k,a,b,c),warmup=3,repeat=25)
    pcts.append(cl/t*100)
pcts=np.array(pcts)
print(f"{v} vs cuBLAS over {N} back-to-back samples:")
print(f"  mean={pcts.mean():.2f}%  median={np.median(pcts):.2f}%  max={pcts.max():.2f}%  min={pcts.min():.2f}%  std={pcts.std():.2f}")
print(f"  samples > 100%: {(pcts>100).sum()}/{N}")
print(f"  all: {', '.join(f'{p:.1f}' for p in pcts)}")
