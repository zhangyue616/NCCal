"""The archived Gaussian-QR, diagonal-sign-corrected Haar basis; Z@Q."""
from prepare import *
import numpy as np,time
from threadpoolctl import threadpool_limits
BASE=ROOT/'protein_extension'
EXISTING={1280:BASE/'calibration/rotations',1920:BASE/'review_followup/rotation/35m/rotations',5120:ROOT/'scale/trained_650m/rotations'}
def matrix_path(width,index):
    return (EXISTING[width] if width in EXISTING else OUT/'haar'/f'width_{width}')/f'haar_{index:02d}.npy'
def matrix(width,seed):
    index=seed-2026111800;assert 1<=index<=9
    path=matrix_path(width,index)
    if not path.exists():raise FileNotFoundError('Generate/freeze the Haar source before evaluation: '+str(path))
    q=np.load(path,mmap_mode='r',allow_pickle=False);assert q.shape==(width,width) and q.dtype==np.float64;return q
def prepare():
    records=[]
    for width in [1280,1920,2304,3072,5120]:
        for index in range(1,10):
            path=matrix_path(width,index);seed=2026111800+index;created=False;elapsed=None
            if not path.exists():
                assert width not in EXISTING;path.parent.mkdir(parents=True,exist_ok=True);start=time.time()
                with threadpool_limits(limits=8):
                    rng=np.random.default_rng(seed);q,r=np.linalg.qr(rng.standard_normal((width,width)));q*=np.where(np.diag(r)>=0,1.,-1.)[None,:];error=float(np.max(np.abs(q.T@q-np.eye(width))))
                assert error<1e-12;np.save(path,q,allow_pickle=False);created=True;elapsed=time.time()-start
                write(str(path.with_suffix('.json').relative_to(OUT)),dict(seed=seed,width=width,sha256=sha(path),max_orthogonality_error=error,generation_seconds=elapsed,convention='Gaussian QR with diagonal sign correction; Z@Q; same Q shared by all blocks of this width'))
                del q,r
            original_meta=js(path.with_suffix('.json')) if path.with_suffix('.json').exists() else {}
            observed=np.load(path,mmap_mode='r',allow_pickle=False);assert observed.shape==(width,width) and observed.dtype==np.float64;del observed
            digest=sha(path)
            if original_meta.get('sha256'):assert original_meta['sha256']==digest
            records.append(dict(width=width,index=index,seed=seed,path=str(path),sha256=digest,shape=[width,width],dtype='float64',existing_archive=width in EXISTING,new_generation_seconds=elapsed,orthogonality_error_from_bound_receipt=original_meta.get('max_orthogonality_error'),convention='Z@Q; one shared matrix per width/index across all blocks'))
            print(width,index,'ready',elapsed,flush=True)
    write('HAAR_MANIFEST.json',dict(utc=datetime.now(timezone.utc).isoformat(),records=records,source_code=str(BASE/'review_followup/rotation/run_rotation.py'),source_code_sha256=sha(BASE/'review_followup/rotation/run_rotation.py'),selection_before_evaluation=True,prior_fast_DCT_not_used=True))
if __name__=='__main__':prepare()
