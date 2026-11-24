"""Static source-mapping audit only. Does not construct a model or run inference."""
from pathlib import Path
import argparse,ast,hashlib,json,datetime
P=Path(__file__).parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--transform-source',type=Path,required=True)
parser.add_argument('--fastplms-source',type=Path,required=True)
parser.add_argument('--esm-block-source',type=Path,required=True)
parser.add_argument('--author-code-dir',type=Path,required=True)
args=parser.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hit(path,needle):
 lines=Path(path).read_text(encoding='utf8').splitlines()
 found=[i+1 for i,l in enumerate(lines)if needle in l]
 assert found,(path,needle)
 return found
transform=args.transform_source
tree=ast.parse(transform.read_text(encoding='utf8'))
# Execute only the name mapper, which has no torch dependency or network/I/O.
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_esmc_key')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(transform),'exec'),ns)
keymap=ns['_esmc_key']
checks=[]
for source,target in [
 ('esmc.transformer.blocks.29.ffn.fc1_weight','transformer.blocks.29.ffn.1.weight'),
 ('esmc.transformer.blocks.29.ffn.fc2_weight','transformer.blocks.29.ffn.3.weight'),
 ('transformer.blocks.29.ffn.1.weight','transformer.blocks.29.ffn.1.weight'),
 ('transformer.blocks.29.ffn.3.weight','transformer.blocks.29.ffn.3.weight'),
]:
 actual=keymap(source);assert actual==target
 checks.append(dict(source_key=source,mapped_key=actual,expected_key=target,PASS=True))
fast=args.fastplms_source;up=args.esm_block_source
assert 'x1, x2 = x.chunk(2, dim=-1)'in fast.read_text(encoding='utf8')
assert 'return F.silu(x1) * x2'in fast.read_text(encoding='utf8')
assert 'x1, x2 = x.chunk(2, dim=-1)'in up.read_text(encoding='utf8')
assert 'return F.silu(x1) * x2'in up.read_text(encoding='utf8')
att=json.loads((P/'esmc634_runtime_attestation.json').read_text(encoding='utf8'))
assert sha(fast)==att['files']['fastplms/models/esm_plusplus/modeling_esm_plusplus.py'].split(':')[1]
files=[transform,fast,P/'esmc634_models.toml',P/'esmc634_runtime_attestation.json',P/'esmc634_hf_config.json',up,args.author_code_dir/'capture.py',args.author_code_dir/'poles.py',args.author_code_dir/'bases.py']
metadata=json.loads((P.parent/'scan/ESMC_preflight.json').read_text(encoding='utf8'))
assert metadata['layers']==30 and metadata['units_per_layer']==2560
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_SOURCE_MAPPING_ONLY',inference_or_model_construction=False,
 scope='Pinned deterministic conversion plus runtime SwiGLU source; no remote weight-value parity or historical author checkpoint hash remeasurement.',
 conversion_revision='e6dd397a9ad368c998d714f6bd64d40b533d1ed1',fast_weights_revision='46c5f7d562e47d4c14165b424c71ab7db008e6fb',biohub_source_revision='a59b831785f907e96e6a246b1d142bfb76df31ee',
 checks=checks,block0=29,unit0=634,ffn_width=2560,gate_projection_row0=634,up_projection_row0=3194,down_projection_column0=634,
 author_mlp_positive='ReLU(SiLU(gate634)*up634)',author_mlp_native='Gate-open tagged poles; differs from mlp_positive where gate<=0 and product>0',
 tensors_are_cloned_without_axis_operations=True,
 source_evidence={str(transform):{'clone':hit(transform,'return value.detach().cpu().clone()'),'map':hit(transform,'transformed[mapped] = _clone_tensor'),'fc1':hit(transform,'(".ffn.fc1_weight", ".ffn.1.weight")'),'fc2':hit(transform,'(".ffn.fc2_weight", ".ffn.3.weight")'),'esmc_transform':hit(transform,'return _map_state(state, _esmc_key, expected_keys)'),'transform_dispatch':hit(transform,'"esmc_to_fastplms_v1": _esmc')},str(fast):{'chunk':hit(fast,'x1, x2 = x.chunk(2, dim=-1)'),'product':hit(fast,'return F.silu(x1) * x2'),'ffn_expansion':hit(fast,'expansion_ratio: float = 8 / 3')},str(up):{'chunk':hit(up,'x1, x2 = x.chunk(2, dim=-1)'),'product':hit(up,'return F.silu(x1) * x2')}},
 files_sha256={str(p):sha(p)for p in files})
(P/'esmc634_checkpoint_mapping_checks.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({k:receipt[k]for k in('status','block0','unit0','gate_projection_row0','up_projection_row0','down_projection_column0','inference_or_model_construction')},indent=2))
