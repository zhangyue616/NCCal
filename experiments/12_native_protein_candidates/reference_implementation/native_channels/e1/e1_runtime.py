"""Original E1 modules with a bounded, single-sequence SDPA kernel adapter."""
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'python_deps'), str(ROOT / 'upstream/E1-main/src')]
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from E1.batch_preparer import E1BatchPreparer
from E1.config import E1Config
import E1.modeling as modeling


def sdpa_single(self, attention_type, query_states, key_states, val_states,
                sequence_ids, attention_args=None, output_attentions=False):
    # This adapter is deliberately limited to one unpadded, uncached sequence.
    assert not output_attentions
    assert sequence_ids.ndim == 2 and sequence_ids.shape[0] == 1
    assert bool((sequence_ids == 0).all())
    assert query_states.shape[1] == key_states.shape[1]
    q, k, v = [x.transpose(1, 2) for x in (query_states, key_states, val_states)]
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return y.transpose(1, 2).reshape(1, query_states.shape[1], self.hidden_size).contiguous(), None


def install_sdpa(model):
    for block in model.model.layers:
        attn = block.norm_attn_norm.self_attn
        attn._attn = types.MethodType(sdpa_single, attn)
    # For one unpadded sequence both original within-seq equality masks and
    # global sequence-id block-causal masks are all True. No dense mask needed.
    def single_mask(sequence_ids):
        assert sequence_ids.shape[0] == 1 and bool((sequence_ids == 0).all())
        return None
    modeling.create_block_causal_mask_optimized = single_mask


def get_model(trained=True, seed=None, device='cuda'):
    config = E1Config.from_pretrained(str(ROOT / 'weights'))
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    model = modeling.E1ForMaskedLM(config)
    if trained:
        state = load_file(str(ROOT / 'weights/model.safetensors'))
        load_result = model.load_state_dict(state, strict=True)
        assert not load_result.missing_keys and not load_result.unexpected_keys
        del state
    install_sdpa(model)
    model = model.eval().to(device=device, dtype=torch.float32)
    return model


def get_batch(sequence, device='cuda'):
    bp = E1BatchPreparer(device=torch.device(device))
    batch = bp.get_batch_kwargs([sequence], device=torch.device(device))
    batch = {k: batch[k] for k in ['input_ids', 'within_seq_position_ids', 'global_position_ids', 'sequence_ids']}
    assert batch['input_ids'].shape == (1, len(sequence) + 4)
    assert batch['input_ids'][0, 2:-2].tolist() == [bp.vocab[x] for x in sequence]
    return batch


def attach_unit_capture(model, callback):
    """callback(layer0, full-token post-SwiGLU input to w2)."""
    handles = []
    for layer, block in enumerate(model.model.layers):
        handles.append(block.ffn.mlp.w2.register_forward_pre_hook(
            lambda mod, args, layer=layer: callback(layer, args[0])))
    return handles
