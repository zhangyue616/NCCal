"""Bind original unit identities before readout extraction."""
from prepare import *
from inventory import table
from collections import Counter
import numpy as np
def main():
    units=[]
    for r in table(OUT/'PUBLIC_SELECTED.tsv'):
        model=r['canonical_model'];concept=r['concept'];units.append(dict(unit_id=f"Hallee|{model}|b{r['block0']}|u{r['unit0']}|{r['pole']}|"+hashlib.sha256(concept.encode()).hexdigest()[:10],source='Hallee',model=model,block0=int(r['block0']),unit0=int(r['unit0']),pole=r['pole'],family='pre_pole' if model.startswith('esm2') else 'gate_tagged',sign=1,concept=concept,annotation_category=r['annotation_category'],masked=False,status='IDENTITY_MAPPED_AUTHOR_CODE',author_score=float(r['best_test_f1'])))
    for model,block,unit in [('esm2_8m',2,801),('esm2_35m',5,814),('esm2_650m',15,4566)]:units.append(dict(unit_id=f'NCCal|{model}|b{block}|u{unit}|OWN_EXPOSURE',source='NCCal',model=model,block0=block,unit0=unit,pole='signed_positive',family='native_signed',sign=1,concept='OWN_EXPOSURE',annotation_category='exposure',masked=False,status='ARCHIVED_ORIGINAL_IDENTITY'))
    units.append(dict(unit_id='NCCal|esm2_8m|b2|u1246|masked|OWN_ALA_HELIX_VS_BETA',source='NCCal',model='esm2_8m',block0=2,unit0=1246,pole='signed_negative',family='native_signed',sign=-1,concept='OWN_ALA_HELIX_VS_BETA',annotation_category='alanine_helix_vs_beta',masked=True,status='ARCHIVED_MASKED_ALANINE_CENTRE_IDENTITY'))
    identities=js(OUT/'E1_F_IDENTITY_REPLAY.json')['identities']
    for unit in [2192,735,2062,1900]:
        identity=identities[str(unit)];block=identity['block0'];original='archived D-signed same-layer comparison'
        for concept,sign in [('OWN_HELIX_START',identity['sign_start']),('OWN_HELIX_VS_BETA',identity['sign_helix_vs_beta'])]:
            units.append(dict(unit_id=f'NCCal|e1_150m|b{block}|u{unit}|{concept}',source='NCCal',model='e1_150m',block0=block,unit0=unit,pole='signed_positive' if sign==1 else 'signed_negative',family='native_signed',sign=sign,concept=concept,annotation_category='helix_start' if concept=='OWN_HELIX_START' else 'helix_vs_beta',masked=False,status='ORIGINAL_IDENTITY_CROSS_DEFINITION_PRESET',original_label=original))
    unresolved=[]
    for unit in []:
        for concept in ['OWN_HELIX_START','OWN_HELIX_VS_BETA']:unresolved.append(dict(unit_id=f'NCCal|e1_150m|u{unit}|{concept}',unit0=unit,concept=concept,status='IDENTITY_UNRESOLVED',reason='Guide lists E1 unit number without layer/polarity; multiple E1 blocks have this coordinate; no selection from current D rankings permitted'))
    old=np.load(ROOT/'results/revision/protein_D/FINAL_MODELS.npz');cov=float(np.mean((old['train_dj']-old['train_dj'].mean())*(old['train_y_a']-old['train_y_a'].mean())))
    assert cov<0
    write('UNIT_MANIFEST.json',{'utc':datetime.now(timezone.utc).isoformat(),'units':units,'unresolved':unresolved,'label_rows':len(units),'public_label_rows':sum(u['source']=='Hallee' for u in units),'public_distinct_physical_signed_units':len({(u['model'],u['block0'],u['unit0'],u['pole']) for u in units if u['source']=='Hallee'}),'own_known_rows':sum(u['source']=='NCCal' for u in units),'1246_original_D_covariance':cov,'1246_D_source':str(ROOT/'results/revision/protein_D/FINAL_MODELS.npz'),'1246_D_source_sha256':sha(ROOT/'results/revision/protein_D/FINAL_MODELS.npz'),'E1_cross_definition_note':'All four start=-raw; Helix/Beta signs2192+,735-,2062+,1900+; literal interior_vs_start proven START-positive from archived pair annotations; archived F20 already used by all four.','E1_identity_replay_sha256':sha(OUT/'E1_F_IDENTITY_REPLAY.json'),'activation_outputs_read':'existing D identity direction and old F replay only; no new evaluation outputs'})
    print(len(units),len(unresolved),flush=True)
if __name__=='__main__':main()
