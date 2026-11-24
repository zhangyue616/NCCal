from pathlib import Path
import json, time, hashlib, traceback
import e1_runtime as rt
import torch
from safetensors.torch import load_file

root = rt.ROOT
torch.set_num_threads(6)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
record = json.loads(Path('validation/repeat_structure/groups/P69905.json').read_text())
sequence = record['chosen']['entity']['sequence']
result = {'sequence_source': 'existing natural human hemoglobin alpha P69905/1ird A; no label scoring',
          'length': len(sequence), 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
          'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
          'software': {'transformers': __import__('transformers').__version__, 'kernels': __import__('importlib.metadata').metadata.version('kernels')},
          'weight_source': 'Profluent-Bio/E1-150m', 'weight_revision': 'c4dbfe827e4aa6ed7f95eaef50dc1e084f4d77dc',
          'source_revision': 'bfd2620a602248499f3d2583d85a7ecddf0b6e02'}
config = rt.E1Config.from_pretrained(str(root/'weights'))
model = rt.modeling.E1ForMaskedLM(config)
load = model.load_state_dict(load_file(str(root/'weights/model.safetensors')), strict=True)
result['strict_load'] = str(load)
result['parameters'] = sum(p.numel() for p in model.parameters())
result['token_ids'] = rt.get_batch(sequence, device='cpu')['input_ids'][0].tolist()

captures = {}
handles = rt.attach_unit_capture(model, lambda l, a: captures.__setitem__(l, a.detach().float().cpu()))
model = model.eval().cuda()
try:
    with torch.inference_mode():
        model(**rt.get_batch(sequence))
    result['original_gpu_kernel'] = 'success'
except Exception as e:
    result['original_gpu_kernel'] = {'error': type(e).__name__, 'message': str(e)[:3000]}
    print('ORIGINAL_GPU_FAILED', str(e)[:300], flush=True)

model = model.cpu()
import E1.model.flex_attention as fa
import E1.model.varlen_flex_attention as vfa
from torch.nn.attention.flex_attention import flex_attention
fa.flex_attention = flex_attention
vfa.flex_attention = flex_attention
captures.clear()
native_units = None
try:
    with torch.inference_mode():
        native = model(**rt.get_batch(sequence, device='cpu')).embeddings.detach()
    native_units = dict(captures)
    result['original_cpu_eager_flex'] = 'success (only disabled torch.compile wrapper)'
    print('ORIGINAL_CPU_EAGER_SUCCESS', flush=True)
except Exception as e:
    result['original_cpu_eager_flex'] = {'error': type(e).__name__, 'message': str(e)[:3000]}
    print('ORIGINAL_CPU_EAGER_FAILED', str(e)[:300], flush=True)

rt.install_sdpa(model)
captures.clear()
with torch.inference_mode():
    adapted = model(**rt.get_batch(sequence, device='cpu')).embeddings.detach()
adapted_units = dict(captures)
if native_units:
    result['cpu_flex_vs_sdpa_embedding_max_abs'] = float((native - adapted).abs().max())
    result['cpu_flex_vs_sdpa_unit_max_abs_per_layer'] = [float((native_units[l] - adapted_units[l]).abs().max()) for l in range(20)]
    result['cpu_flex_vs_sdpa_unit_rel_l2_per_layer'] = [float((native_units[l] - adapted_units[l]).norm() / native_units[l].norm()) for l in range(20)]
    assert max(result['cpu_flex_vs_sdpa_unit_rel_l2_per_layer']) < 0.0001

model.cuda()
captures.clear()
with torch.inference_mode():
    gpu = model(**rt.get_batch(sequence)).embeddings.cpu()
result['cpu_sdpa_vs_gpu_sdpa_embedding_max_abs'] = float((adapted - gpu).abs().max())
result['cpu_sdpa_vs_gpu_sdpa_unit_max_abs_per_layer'] = [float((adapted_units[l] - captures[l]).abs().max()) for l in range(20)]
result['cpu_sdpa_vs_gpu_sdpa_unit_rel_l2_per_layer'] = [float((adapted_units[l] - captures[l]).norm() / adapted_units[l].norm()) for l in range(20)]
assert max(result['cpu_sdpa_vs_gpu_sdpa_unit_rel_l2_per_layer']) < 0.0001
assert all(torch.isfinite(captures[l]).all() for l in range(20))
result['hook_shapes'] = {str(l): list(a.shape) for l, a in captures.items()}
result['weight_sha256'] = hashlib.file_digest(open(root/'weights/model.safetensors','rb'), 'sha256').hexdigest()
result['success'] = bool(native_units)
(root/'preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
