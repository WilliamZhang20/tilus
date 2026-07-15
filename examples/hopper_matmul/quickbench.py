"""Fast local benchmark: one tilus version vs cuBLAS. Usage: python quickbench.py v6 v7 ..."""
import sys, math, importlib
import torch, tilus
from tilus.utils import benchmark_func
tilus.option.cache_dir('./cache')
VCLS = {"v5":"MatmulWGMMAV5","v6":"MatmulWGMMAV6","v7":"MatmulWGMMAV7"}
m=n=k=8192
a=(torch.rand(m,k,dtype=torch.float16,device='cuda')-0.5)/math.sqrt(k)
b=(torch.rand(n,k,dtype=torch.float16,device='cuda')-0.5)/math.sqrt(k)
c=torch.empty(m,n,dtype=torch.float16,device='cuda')
cref=a@b.T
cl=benchmark_func(lambda: torch.matmul(a,b.T,out=cref),warmup=5,repeat=30)
ctf=2*m*n*k/cl*1e-9
print(f"cublas: {cl*1e3:.3f}us  {ctf:.1f} TF  100.0%")
for v in sys.argv[1:]:
    try:
        mod=importlib.import_module(f"matmul_{v}")
        mm=getattr(mod,VCLS[v])()
        mm(m,n,k,a,b,c); torch.cuda.synchronize()
        err=(c.float()-cref.float()).abs().max().item()
        ok="OK" if err<0.5 else f"BADerr={err:.3f}"
        t=benchmark_func(lambda: mm(m,n,k,a,b,c),warmup=5,repeat=30)
        tf=2*m*n*k/t*1e-9
        print(f"{v}: {t*1e3:.3f}us  {tf:.1f} TF  {tf/ctf*100:.1f}%  [{ok}]")
    except Exception as e:
        print(f"{v}: ERROR {str(e)[:120]}")
