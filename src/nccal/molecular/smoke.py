"""Small real-checkpoint replays with public-source derived input fixtures."""
from pathlib import Path
import json,hashlib
import numpy as np

def run(root,experiment,model_dir,device='cpu'):
    import torch
    from torch import nn
    from transformers import RobertaConfig,RobertaModel
    from .reproduce import compare
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=RobertaConfig.from_json_file(str(model_dir/'config.json'));config._attn_implementation='eager'
    if (config.num_hidden_layers,config.hidden_size,config.intermediate_size,config.num_labels)!=(3,384,464,199):raise ValueError('The supplied model is not the fixed checkpoint architecture.')
    class Head(nn.Module):
        def __init__(self):
            super().__init__();self.dense=nn.Linear(384,384);self.dropout=nn.Dropout(config.hidden_dropout_prob);self.out_proj=nn.Linear(384,199)
        def forward(self,x):return self.out_proj(self.dropout(torch.relu(self.dense(self.dropout(x[:,0])))))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.roberta=RobertaModel(config,add_pooling_layer=False);self.regression=Head();self.register_buffer('norm_mean',torch.tensor(config.norm_mean));self.register_buffer('norm_std',torch.tensor(config.norm_std))
        def forward(self,ids,mask):return self.regression(self.roberta(input_ids=ids,attention_mask=mask,return_dict=True).last_hidden_state)
    model=Model()
    state=torch.load(model_dir/'pytorch_model.bin',map_location='cpu',weights_only=True)
    position=state.pop('roberta.embeddings.position_ids',None)
    if position is not None and not torch.equal(position,torch.arange(config.max_position_embeddings).expand(1,-1)):raise ValueError('Unexpected fixed positional buffer.')
    model.load_state_dict(state,strict=True);model.to(device).eval().requires_grad_(False)
    d=root/'data/molecular';data=json.loads((d/'smoke_inputs.json').read_text());expected=np.load(d/'smoke_expected.npz');checks=[]
    def forward(view,layer=1,replacement=None,direction=None,delta=None,pre=False):
        ids=torch.tensor([view['input_ids']],device=device);mask=torch.tensor([view['attention_mask']],device=device);anchors=torch.tensor(view['anchors'],device=device);captured={}
        module=model.roberta.encoder.layer[layer].intermediate
        def hook(m,a,z):
            captured['a']=z[0,anchors].detach().cpu().numpy().copy()
            if replacement is None and direction is None:return z
            z=z.clone()
            if replacement is not None:z[0,anchors]=torch.as_tensor(np.array(replacement,copy=True),dtype=z.dtype,device=device)
            else:z[0,anchors]+=torch.as_tensor(delta,dtype=z.dtype,device=device)[:,None]*torch.as_tensor(direction,dtype=z.dtype,device=device)[None,:]
            return z
        handle=(module.dense if pre else module).register_forward_hook(hook)
        try:
            with torch.inference_mode():output=model(ids,mask)[0].cpu().numpy()
        finally:handle.remove()
        return captured['a'],output
    native,baseline=forward(data['views'][0])
    checks.append(compare('archived_post_GELU_activation_max_error',float(np.max(np.abs(native-expected['native'][0]))),0,2e-5))
    checks.append(compare('archived_199_output_max_error',float(np.max(np.abs(baseline-expected['baseline'][0]))),0,2e-5))
    if experiment=='01_molecular_calibration':
        pre,_=forward(data['views'][0],layer=2,pre=True)
        checks.append(compare('calibration_pre_GELU_width',int(pre.shape[1]),464));checks.append(compare('finite_calibration_activation',bool(np.isfinite(pre).all()),True))
    elif experiment=='02_molecular_reconstruction':
        replacement=native.copy();replacement[:,293]=expected['rule'][:,293];_,out=forward(data['views'][0],replacement=replacement)
        checks.extend([compare('replacement_output_is_finite',bool(np.isfinite(out).all()),True),compare('rule_replaces_only_selected_channel',int(np.count_nonzero((replacement-native)[:,np.arange(464)!=293])),0)])
        _,sham=forward(data['views'][0],replacement=native);checks.append(compare('sham_output_max_error',float(np.max(np.abs(sham-baseline))),0,2e-5))
    elif experiment=='03_whole_layer':
        _,out=forward(data['views'][0],replacement=expected['rule']);checks.append(compare('whole_layer_rule_output_is_finite',bool(np.isfinite(out).all()),True))
        mean=np.load(d/'development_reference.npz')['mean'];_,out=forward(data['views'][0],replacement=np.broadcast_to(mean,native.shape));checks.append(compare('whole_layer_mean_output_is_finite',bool(np.isfinite(out).all()),True))
    elif experiment=='04_sparse_dictionary':
        pre,_=forward(data['views'][0],layer=2,pre=True);z=np.load(d/'calibration_dictionary_encoder.npz');x=((pre.astype(float)-z['mean'])/float(z['scale'])).astype(np.float32)-z['decoder_bias'];scores=x@z['encoder_weight'].T+z['encoder_bias'];scores=np.maximum(scores,0);keep=np.argsort(scores,axis=1)[:,-16:];codes=np.zeros_like(scores);np.put_along_axis(codes,keep,np.take_along_axis(scores,keep,axis=1),axis=1);reconstructed=codes@z['decoder_weight'].T+z['decoder_bias']
        checks.extend([compare('calibration_dictionary_width',int(codes.shape[1]),928),compare('max_active_features',bool(((codes>0).sum(1)<=16).all()),True),compare('finite_dictionary_reconstruction',bool(np.isfinite(reconstructed).all()),True)])
    elif experiment=='05_molecular_rewrites':
        for v in range(1,6):
            actual,out=forward(data['views'][v]);checks.append(compare('rewrite_'+str(v)+'_activation_max_error',float(np.max(np.abs(actual-expected['native'][v]))),0,2e-5));checks.append(compare('rewrite_'+str(v)+'_output_max_error',float(np.max(np.abs(out-expected['baseline'][v]))),0,2e-5))
    elif experiment=='06_molecular_edits':
        pair=data['edit'];z=np.load(d/'edit_smoke_expected.npz');a,_=forward(pair['source']);b,_=forward(pair['product']);checks.append(compare('edited_parent_readout_error',float(np.max(np.abs(a-z['source']))),0,2e-5));checks.append(compare('edited_product_readout_error',float(np.max(np.abs(b-z['product']))),0,2e-5));checks.append(compare('matched_edit_activation_delta_error',float(np.max(np.abs((b-a)-(z['product']-z['source'])))),0,4e-5))
    elif experiment=='07_basis_comparison':
        for kind in ['rot_30000','pca','sae_rep0']:
            z=np.load(d/(kind+'_encoder.npz'));x=((torch.tensor(native,device=device).double()-torch.tensor(z['input_center'],device=device))/float(z['input_scale'])).float();bias=torch.tensor(z['encoder_input_bias'],device=device) if 'encoder_input_bias' in z.files else torch.zeros(464,device=device);scores=torch.nn.functional.linear(x-bias,torch.tensor(z['encoder_weights'],dtype=torch.float32,device=device),torch.tensor(z['encoder_bias'],dtype=torch.float32,device=device))
            if kind.startswith('sae'):
                values,indices=torch.topk(scores.relu(),int(z['k']),dim=-1);scores=torch.zeros_like(scores).scatter_(-1,indices,values)
            actual=scores.cpu().numpy();stored=np.load(d/(kind+'_smoke_expected.npz'))['coefficients'][0];checks.append(compare(kind+'_coefficient_error',float(np.max(np.abs(actual-stored))),0,3e-5));direction=z['directions'][0];delta=np.full(len(native),.01);_,out=forward(data['views'][0],direction=direction,delta=delta);checks.append(compare(kind+'_writeback_finite',bool(np.isfinite(out).all()),True))
    return checks
