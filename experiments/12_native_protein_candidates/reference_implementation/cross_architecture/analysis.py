"""Development-only fitting and application to held-out disjoint protein groups."""
import hashlib
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from cohorts import ROOT, OUT, read_tsv, sequences

AA = "ACDEFGHIKLMNPQRSTVWYX^"
PENALTIES = (.1,.01,.001,.0001)
SALT = 20261118

class Sites:
    def __init__(self, split):
        self.seq = sequences(OUT/f"{split}_sequences.jsonl")
        self.pairs = {t:list(read_tsv(OUT/f"{split}_{t}.tsv")) for t in ("contact","exposure")}
        self.grid = list(read_tsv(OUT/"D_grid.tsv")) if split=="D" else []
        keys = {(r["accession"],int(r[k])) for data in self.pairs.values() for r in data for k in ("early","late")}
        keys.update((r["accession"],int(r["pos"])) for r in self.grid)
        self.keys = sorted(keys)
        self.index = {k:i for i,k in enumerate(self.keys)}
        self.groups = np.array([self.seq[a]["group"] for a,p in self.keys])
        self.byacc = {}
        for i,(a,p) in enumerate(self.keys): self.byacc.setdefault(a,[]).append((i,p))
        self.grid_idx = np.array([self.index[r["accession"],int(r["pos"])] for r in self.grid],dtype=int)
        self.x = local_features(self.keys,self.seq)
        self.pair_idx = {}
        for task,rs in self.pairs.items():
            self.pair_idx[task] = np.array([(self.index[r["accession"],int(r["early"])],self.index[r["accession"],int(r["late"])]) for r in rs],dtype=int).reshape(-1,2)
    def pair_difference(self,values,task):
        ix = self.pair_idx[task]
        return values[ix[:,1]]-values[ix[:,0]]

def local_features(keys, seq):
    width=31*len(AA)
    slot={a:i for i,a in enumerate(AA)}
    rows,cols,vals=[],[],[]
    for i,(acc,pos) in enumerate(keys):
        s=seq[acc]["sequence"]
        for k in range(31):
            site=pos-16+k
            residue=s[site] if 0<=site<len(s) else "^"
            rows.append(i);cols.append(k*len(AA)+slot.get(residue,slot["X"]));vals.append(1.)
        rel=(pos-1)/max(1,len(s)-1)
        rows.extend([i]*4);cols.extend(range(width,width+4));vals.extend((rel,rel**2,len(s)/256,1.))
    return sparse.csr_matrix((vals,(rows,cols)),shape=(len(keys),width+4))

def folds(groups):
    unique=np.unique(groups)
    if len(unique)<4: raise ValueError("Fewer than four independent development groups")
    return {str(g):i%4 for i,g in enumerate(np.random.default_rng(SALT).permutation(unique))}

def groupweights(groups):
    _,ix,n=np.unique(groups,return_inverse=True,return_counts=True)
    return 1./(len(n)*n[ix])

def solve_ridge(train, cross, y, penalty):
    trace=np.mean(np.diag(train))
    if trace<=0: raise ValueError("Zero pairwise kernel")
    reg=train/trace
    reg.flat[::1+len(train)] += len(train)*penalty
    return np.clip((1+(cross/trace)@cho_solve(cho_factor(reg,lower=True),2*y-1))/2,0,1)

class PairModel:
    def __init__(self, data, task):
        self.task=task
        self.x=data.pair_difference(data.x,task)
        self.k=(self.x@self.x.T).toarray()
        self.y=np.array([int(r["target"]) for r in data.pairs[task]],dtype=float)
        groups=[r["group"] for r in data.pairs[task]]
        foldmap=folds(groups)
        self.f=np.array([foldmap[g] for g in groups])
        self.indices=np.arange(len(groups))
        self.inner=[]
        best=np.inf
        for penalty in PENALTIES:
            out=self.crossfit(self.indices,self.f,penalty)
            loss=np.mean((self.y-out)**2)
            if loss<best:
                self.oof,self.penalty,best=out,penalty,loss
        for mask in range(4):
            tr=self.indices[self.f!=mask]; te=self.indices[self.f==mask]
            inner_prediction=self.crossfit(tr,self.f[tr],self.penalty)
            self.inner.append((tr,te,self.y[tr]-inner_prediction))
    def crossfit(self,ii,f,penalty):
        out=np.zeros(len(ii))
        for j in np.unique(f):
            tr=ii[f!=j];te=ii[f==j]
            out[f==j]=solve_ridge(self.k[np.ix_(tr,tr)],self.k[np.ix_(te,tr)],self.y[tr],penalty)
        return out
    def base_for(self, data):
        front=data.pair_difference(data.x,self.task)
        return solve_ridge(self.k,(front@self.x.T).toarray(),self.y,self.penalty)
    def select_axis(self,delta):
        z=np.asarray(delta,dtype=np.float64)
        losses=np.zeros(z.shape[1])
        for tr,te,r in self.inner:
            s=slope(z[tr],r)
            p=np.clip(self.oof[te,None]+z[te]*s,0,1)
            losses+=((self.y[te,None]-p)**2).sum(0)
        best=int(np.argmin(losses))
        return best,float(slope(z[:,best], self.y-self.oof)),losses/len(z)

def slope(z,residual):
    return (z.T@residual)/(1.01*np.maximum(np.sum(z*z,axis=0),1e-20))

class GridRegression:
    """Fast linear maps for each split with per-group weighted ridge and an unpenalized intercept."""
    def __init__(self,data):
        grid=data.grid_idx
        self.grid=grid
        self.w=groupweights(data.groups[grid])
        gf=folds(data.groups[grid])
        fo=np.array([gf[str(g)] for g in data.groups])
        self.maps=[]; self.folds=fo
        for hold in (0,1,2,3,None):
            train=grid[fo[grid]!=hold] if hold is not None else grid
            w=groupweights(data.groups[train])
            x=data.x[train]
            weighted=x.multiply(w[:,None])
            gram=(x.T@weighted).toarray()
            gram.flat[::len(gram)+1]+=.01
            gram[-1,-1]-=.01
            dual=cho_solve(cho_factor(gram,lower=True),weighted.T.toarray())
            self.maps.append((train,dual))
    def fit(self,values):
        """Crossfit at all dev grid sites and at any dev pairs; final map for deployment."""
        z=values.reshape(len(values),-1)
        # Geometry-only contact selection never needs this high-dimensional operation.
        out=np.empty_like(z,dtype=np.float64)
        for j,(train,map_) in enumerate(self.maps[:4]):
            test=np.flatnonzero(self.folds==j)
            coeff=map_@np.asarray(z[train],dtype=np.float64)
            out[test]=z[test]-self.design[test]@coeff
        tr,map_=self.maps[-1]
        return out, map_@np.asarray(z[tr],dtype=np.float64)
    @classmethod
    def from_data(cls,data):
        obj=cls(data); obj.design=data.x; return obj

def match_channel(teacher,student,w):
    y=teacher-np.sum(w*teacher)
    x=student-np.sum(w[:,None]*student,axis=0)
    cov=(w*y)@x
    var=w@(x*x)
    corr=cov/np.sqrt(np.maximum(1e-25,var*(w@(y*y))))
    idx=int(np.argmax(np.abs(corr)))
    return idx,float(corr[idx]),float(cov[idx]/max(var[idx],1e-25))

def rotation(width,idx):
    import torch
    gen=torch.Generator(device="cpu").manual_seed(202611920+idx)
    g=torch.randn(width,width,generator=gen,dtype=torch.float64)
    # Same CPU float64 QR for development and holding a fixed direction in external data.
    q,r=torch.linalg.qr(g)
    s=torch.sign(torch.diag(r));s[s==0]=1
    return (q*s).numpy()
