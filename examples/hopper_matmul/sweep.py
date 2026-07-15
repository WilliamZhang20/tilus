"""Sweep pinned schedules for a v6-clone, find best vs cuBLAS. Fast local."""
import math, itertools
import torch, tilus
from tilus.utils import benchmark_func
tilus.option.cache_dir('./cache_sweep')
import matmul_v6 as base

m=n=k=8192
a=(torch.rand(m,k,dtype=torch.float16,device='cuda')-0.5)/math.sqrt(k)
b=(torch.rand(n,k,dtype=torch.float16,device='cuda')-0.5)/math.sqrt(k)
c=torch.empty(m,n,dtype=torch.float16,device='cuda')
cref=a@b.T
cl=benchmark_func(lambda: torch.matmul(a,b.T,out=cref),warmup=5,repeat=50)
ctf=2*m*n*k/cl*1e-9
print(f"cublas baseline: {ctf:.1f} TF")

# SMEM budget check (232KB)
def fits(bm,bn,bk,st):
    sa=st*2*(bm//2)*bk*2; sb=st*bn*bk*2
    return (sa+sb) <= 232448

configs=[]
for bm,bn in [(128,256),(256,128),(128,128),(256,256)]:
    for bk in [32,64]:
        for st in [3,4,5,6]:
            for sw in [4,8]:
                if fits(bm,bn,bk,st):
                    configs.append((st,bm,bn,bk,sw))

print(f"testing {len(configs)} configs that fit SMEM")
results=[]
for (st,bm,bn,bk,sw) in configs:
    base.MatmulWGMMAV6.debug_schedule=dict(num_stages=st,block_m=bm,block_n=bn,block_k=bk,swizzle_size=sw)
    try:
        mm=base.MatmulWGMMAV6()
        mm(m,n,k,a,b,c); torch.cuda.synchronize()
        err=(c.float()-cref.float()).abs().max().item()
        if err>0.5:
            print(f"  ({st},{bm},{bn},{bk},sw{sw}): BAD err={err:.2f}"); continue
        t=benchmark_func(lambda: mm(m,n,k,a,b,c),warmup=5,repeat=50)
        tf=2*m*n*k/t*1e-9
        pct=tf/ctf*100
        results.append((pct,tf,(st,bm,bn,bk,sw)))
        print(f"  ({st},{bm},{bn},{bk},sw{sw}): {tf:.1f} TF  {pct:.1f}%")
    except Exception as e:
        print(f"  ({st},{bm},{bn},{bk},sw{sw}): ERR {str(e)[:60]}")
results.sort(reverse=True)
print("\n=== TOP 5 ===")
for pct,tf,cfg in results[:5]:
    print(f"  {cfg}: {tf:.1f} TF  {pct:.1f}%")
