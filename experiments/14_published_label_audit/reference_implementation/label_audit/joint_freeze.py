"""Bind accepted inputs, statistics and GPU implementation before scoring."""
from prepare import *
def main():
    assert not (OUT/'FREEZE.json').exists()
    inp=js(OUT/'INPUT_FREEZE.json')
    for rel,h in inp['input_files'].items():assert sha(OUT/rel)==h,rel
    for rel,h in inp['GPU_code_files'].items():assert sha(OUT/rel)==h,rel
    write('FREEZE.json',dict(status='FROZEN_AUTHORIZED_EXECUTION',utc=datetime.now(timezone.utc).isoformat(),input_freeze_sha256=sha(OUT/'INPUT_FREEZE.json'),protocol_sha256=sha(OUT/'PROTOCOL.md'),CPU_PROTOCOL_FREEZE_sha256=sha(OUT/'CPU_PROTOCOL_FREEZE.json'),CPU_IMPLEMENTATION_CORRECTION_001_sha256=sha(OUT/'CPU_IMPLEMENTATION_CORRECTION_001.json'),GPU_code_files=inp['GPU_code_files'],random_networks=199,rotations=9,COMPUTE_GO_sha256=sha(OUT.parent/'COMPUTE_GO.json'),all_new_evaluation_scores_before_this_freeze=0,eligibility_sha256=sha(OUT/'inputs/UNIT_ELIGIBILITY.tsv'),permission='Complete all four authorized experiments; no user continue stop; guideSection6 hard stops remain'))
    print('JOINT FREEZE',sha(OUT/'FREEZE.json'),flush=True)
if __name__=='__main__':main()
