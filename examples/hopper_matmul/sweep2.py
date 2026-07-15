"""Contention-robust schedule sweep: measure cublas right before each config (ratio)."""
import math
import torch, tilus
from tilus.utils import benchmark_func
tilus.option.cache_dir('./cache_sweep2')
import matmul_v6 as base
m=n=k=8192
a=(torch.rand(m,k,dtype=torch.float16,device='cuda')-0.5)/math.sqrt(k)
b=(torch.rand(n,k,dtype=torch.float16,device='cuda')-0.5)/math.sqrt(k)
c=torch.empty(m,n,dtype=torch.float16,device='cuda')
cref=a@b.T
def fits(bm,bn,bk,st):
    return (st*2*(bm//2)*bk*2 + st*bn*bk*2) <= 232448
configs=[]
for bm,bn in [(128,256),(256,128),(256,256),(128,128)]:
    for bk in [64]:
        for st in [3,4,5,6]:
            for sw in [4,8]:
                if fits(bm,bn,bk,st): configs.append((st,bm,bn,bk,sw))
print(f"{len(configs)} configs")
res=[]
for cfg in configs:
    st,bm,bn,bk,sw=cfg
    base.MatmulWGMMAV6.debug_schedule=dict(num_stages=st,block_m=bm,block_n=bn,block_k=bk,swizzle_size=sw)
    try:
        mm=base.MatmulWGMMAV6(); mm(m,n,k,a,b,c); torch.cuda.synchronize()
        if (c.float()-cref.float()).abs().max().item()>0.5: print(f"{cfg}: BAD"); continue
        cl=benchmark_func(lambda: torch.matmul(a,b.T,out=cref),warmup=3,repeat=20)  # cublas NOW
        t=benchmark_func(lambda: mm(m,n,k,a,b,c),warmup=3,repeat=20)
        pct=cl/t*100
        res.append((pct,cfg)); print(f"{cfg}: {pct:.1f}% of cublas")
    except Exception as e: print(f"{cfg}: ERR {str(e)[:50]}")
res.sort(reverse=True)
print("\nTOP 5:"); [print(f"  {c}: {p:.1f}%") for p,c in res[:5]]
