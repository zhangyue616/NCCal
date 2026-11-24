"""Offline, explicit model inputs for frozen protein forward subsets."""
from __future__ import annotations
import gc,sys,types
from pathlib import Path
import numpy as np

AA="ACDEFGHIKLMNPQRSTVWY"

class ProteinModel:
    def __init__(self,name,weights,dependency_dirs=(),device="cuda"):
        import torch
        self.torch=torch;self.name=name;self.device=device
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for directory in dependency_dirs:sys.path.insert(0,str(directory))
        if name.startswith("esm2_"):
            from transformers import AutoTokenizer,EsmForMaskedLM
            self.tokenizer=AutoTokenizer.from_pretrained(weights,local_files_only=True)
            self.model=EsmForMaskedLM.from_pretrained(weights,local_files_only=True,attn_implementation="sdpa").eval().float().to(device)
            self.blocks=self.model.esm.encoder.layer;self.kind="esm2"
        elif name=="e1_150m":
            from E1.batch_preparer import E1BatchPreparer
            from E1.config import E1Config
            import E1.modeling as modeling
            from safetensors.torch import load_file
            config=E1Config.from_pretrained(str(weights));self.model=modeling.E1ForMaskedLM(config)
            self.model.load_state_dict(load_file(str(Path(weights)/"model.safetensors")),strict=True)
            def single_mask(sequence_ids):
                assert sequence_ids.shape[0]==1 and bool((sequence_ids==0).all())
                return None
            modeling.create_block_causal_mask_optimized=single_mask
            def single_sdpa(module,attention_type,query_states,key_states,val_states,sequence_ids,attention_args=None,output_attentions=False):
                assert not output_attentions and sequence_ids.shape[0]==1 and bool((sequence_ids==0).all())
                assert query_states.shape[1]==key_states.shape[1]
                q,k,v=[x.transpose(1,2) for x in [query_states,key_states,val_states]]
                y=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=False)
                return y.transpose(1,2).reshape(1,query_states.shape[1],module.hidden_size).contiguous(),None
            for block in self.model.model.layers:
                attention=block.norm_attn_norm.self_attn;attention._attn=types.MethodType(single_sdpa,attention)
            self.model=self.model.eval().float().to(device);self.blocks=self.model.model.layers
            self.batch_preparer=E1BatchPreparer(device=torch.device(device));self.kind="e1"
        else:raise ValueError("Unsupported model for this forward subset: "+name)

    def inputs(self,fixture):
        if self.kind=="esm2":
            result={k:v.to(self.device)for k,v in self.tokenizer(fixture["sequence"],return_tensors="pt").items()}
            for p in fixture.get("mask_positions1",[]):result["input_ids"][0,p]=self.tokenizer.mask_token_id
            return result
        batch=self.batch_preparer.get_batch_kwargs([fixture["sequence"]],device=self.torch.device(self.device))
        return {k:batch[k]for k in ["input_ids","within_seq_position_ids","global_position_ids","sequence_ids"]}

    def run(self,fixture,replacement=None,basis=False):
        torch=self.torch;block=self.blocks[fixture["block0"]];values={};handles=[]
        positions=[p+(1 if self.kind=="e1"else 0)for p in fixture["positions1"]]
        def capture(output):
            value=output[0,positions,fixture["unit0"]]
            if fixture["readpoint"]=="pre_pole":value=torch.relu(value if fixture.get("pole")=="positive"else -value)
            values["activation"]=value.detach().cpu().numpy().astype(float)
        def hook(module,args,output):
            capture(output)
            if replacement is not None:
                changed=output.clone();changed[0,positions[0],fixture["unit0"]]=replacement;return changed
            if basis:
                changed=output.clone();changed[...,subset]=output[...,subset]@rotation;return changed
        original_weight=None
        if self.kind=="e1":
            handles.append(block.ffn.mlp.w2.register_forward_pre_hook(lambda module,args:capture(args[0])))
        else:
            module=block.intermediate.dense if fixture["readpoint"]=="pre_pole"else block.intermediate
            if basis:
                subset=[i for i in range(15)if i!=fixture["unit0"]]+[fixture["unit0"]]
                rotation=np.linalg.qr(np.random.default_rng(2026111801).normal(size=(16,16)))[0]
                rotation=torch.as_tensor(rotation,dtype=torch.float32,device=self.device)
                original_weight=block.output.dense.weight[:,subset].detach().clone()
                with torch.no_grad():block.output.dense.weight[:,subset]=original_weight@rotation
            handles.append(module.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                output=(self.model(**self.inputs(fixture)) if self.kind=="esm2"else self.model.model(**self.inputs(fixture)))
            if self.kind=="esm2":
                values["logits"]=output.logits.detach().cpu().numpy()
                if fixture.get("mask_positions1"):
                    ids=self.tokenizer.convert_tokens_to_ids(list(AA));p=fixture["mask_positions1"][0]
                    values["logits20"]=values["logits"][0,p,ids]
            return values
        finally:
            for handle in handles:handle.remove()
            if original_weight is not None:
                with torch.no_grad():block.output.dense.weight[:,subset]=original_weight

    def close(self):
        self.model.cpu();del self.model;gc.collect()
        if self.torch.cuda.is_available():self.torch.cuda.empty_cache()

def check_fixture(model,fixture,tolerance=2e-5):
    actual=model.run(fixture);checks=[]
    if "expected_values"in fixture:
        error=float(np.max(np.abs(actual["activation"]-fixture["expected_values"])));checks.append(dict(quantity="native_activation",max_absolute_error=error,passed=error<=tolerance))
    if "expected_difference"in fixture:
        error=abs(float(actual["activation"][0]-actual["activation"][1])-fixture["expected_difference"]);checks.append(dict(quantity="native_activation_difference",max_absolute_error=error,passed=error<=tolerance))
    if "expected_logits20"in fixture:
        error=float(np.max(np.abs(actual["logits20"]-fixture["expected_logits20"])));checks.append(dict(quantity="masked_neighbour_logits",max_absolute_error=error,passed=error<=tolerance))
    if "replacement"in fixture:
        replaced=model.run(fixture,replacement=fixture["replacement"])
        checks.append(dict(quantity="replacement_forward_finite",passed=bool(np.isfinite(replaced["logits"]).all()),max_absolute_logit_change=float(np.max(np.abs(replaced["logits"]-actual["logits"])))))
    if fixture.get("check_basis_equivalence"):
        rotated=model.run(fixture,basis=True);error=float(np.max(np.abs(rotated["logits"]-actual["logits"])))
        checks.append(dict(quantity="compensated_basis_full_forward",max_absolute_error=error,passed=error<=tolerance))
    return dict(experiment=fixture["experiment"],model=fixture["model"],accession=fixture["accession"],residues=len(fixture["sequence"]),checks=checks,status="PASS"if all(x["passed"]for x in checks)else"FAIL")
