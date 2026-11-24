"""Frozen chemical editing, dual-tokenizer mapping and controlled-string eligibility."""
import dataclasses,hashlib,json
import numpy as np
from rdkit import Chem
from .standardization import standardize,parsed
from .semantic_mapper import SemanticMapper,lex_smiles
from .input_features import input_keys
from .mapping import AtomMapper
def h(text):return hashlib.sha256(text.encode()).hexdigest()
OH=Chem.MolFromSmarts('[CX4;+0:1]-[OX2H1;+0:2]')
KETONE=Chem.MolFromSmarts('[#6:3]-[CX3H0;+0:1](=[OX1;+0:2])-[#6:4]')
ACID=Chem.MolFromSmarts('[CX3;+0:1](=[OX1;+0:2])-[OX2H1;+0:3]')
def safe(m,ids):
 for i in ids:
  a=m.GetAtomWithIdx(i)
  if a.GetIsotope() or a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED:return False
 for b in m.GetBonds():
  if b.GetBeginAtomIdx()in ids and b.GetEndAtomIdx()in ids and b.GetStereo()!=Chem.BondStereo.STEREONONE:return False
 return True
def sites(m):
 oh=sorted(set((x[0],x[1])for x in m.GetSubstructMatches(OH)))
 ket=sorted(set((x[1],x[2])for x in m.GetSubstructMatches(KETONE)))
 acid=sorted(set(tuple(x)for x in m.GetSubstructMatches(ACID)))
 target=set(i for site in oh+ket+acid for i in site)
 out={1:[x for x in oh if safe(m,set(x))],2:[x for x in oh if safe(m,set(x))],3:[x for x in ket if safe(m,set(x))],4:[x for x in acid if safe(m,set(x))],5:[]}
 if target:
  dist=Chem.GetDistanceMatrix(m)
  out[5]=[(a.GetIdx(),)for a in m.GetAtoms()if a.GetAtomicNum()==6 and a.GetIsAromatic()and a.GetTotalNumHs()==1 and min(dist[a.GetIdx(),i]for i in target)>=4 and safe(m,{a.GetIdx()})]
 return out
def edit(m,kind,site):
 rw=Chem.RWMol(m)
 for a in rw.GetAtoms():a.SetIntProp('_source_atom',a.GetIdx())
 removed=[];added=[]
 if kind in (1,3):
  removed=[site[1]]
  # Removing the terminal O and its bond supplies H/CH2 through implicit valence.
  rw.GetAtomWithIdx(site[0]).SetNoImplicit(False)
  rw.RemoveAtom(site[1])
 elif kind in (2,4):
  oxygen=site[1]if kind==2 else site[2]
  a=rw.GetAtomWithIdx(oxygen);a.SetNumExplicitHs(0);a.SetNoImplicit(False)
  new=Chem.Atom(6);new.SetIntProp('_source_atom',-1);index=rw.AddAtom(new);rw.AddBond(oxygen,index,Chem.BondType.SINGLE);added=[index]
 elif kind==5:
  a=rw.GetAtomWithIdx(site[0]);a.SetNumExplicitHs(0);a.SetNoImplicit(False)
  new=Chem.Atom(9);new.SetIntProp('_source_atom',-1);index=rw.AddAtom(new);rw.AddBond(site[0],index,Chem.BondType.SINGLE);added=[index]
 else:raise ValueError('edit kind')
 product=rw.GetMol();Chem.SanitizeMol(product);Chem.AssignStereochemistry(product,cleanIt=True,force=True)
 # Existing unrelated tetrahedral and bond-stereo annotations are checked under lineage.
 mapping=[a.GetIntProp('_source_atom')for a in product.GetAtoms()]
 for a in product.GetAtoms():
  old=mapping[a.GetIdx()]
  if old>=0 and old not in site:
   oa=m.GetAtomWithIdx(old)
   assert (a.GetAtomicNum(),a.GetIsotope(),a.GetFormalCharge())==(oa.GetAtomicNum(),oa.GetIsotope(),oa.GetFormalCharge()),'unrelated_atom_identity'
   assert a.GetChiralTag()==oa.GetChiralTag(),'unrelated_chiral_tag'
 oldedges={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))):(str(b.GetBondType()),str(b.GetStereo()))for b in m.GetBonds()}
 for b in product.GetBonds():
  a,c=mapping[b.GetBeginAtomIdx()],mapping[b.GetEndAtomIdx()]
  if a>=0 and c>=0 and not({a,c}<=set(site)):
   assert oldedges[tuple(sorted((a,c)))]==(str(b.GetBondType()),str(b.GetStereo())),'unrelated_bond_stereo'
 return product,mapping,removed,added
def serialize(product,mapping):
 text=Chem.MolToSmiles(product,canonical=False,isomericSmiles=True)
 order=json.loads(product.GetProp('_smilesAtomOutputOrder'))
 lineage=[mapping[i]for i in order]
 back=parsed(text)
 assert Chem.MolToSmiles(back,True)==Chem.MolToSmiles(product,True),'serialized_stereo_identity'
 return text,lineage
def audit(text,cb,mf,name):
 a=cb.audit_molecule(text,molecule_id=name,require_canonical=False,max_tokens=512)
 if not a['accepted']:raise ValueError('chemberta:'+repr(a['exclusion_reasons']))
 mr=mf.audit(text,name)
 return dict(smiles=text,chemberta=a,molformer=dataclasses.asdict(mr),keys=input_keys([text],[[x['anchor_token_id']for x in a['atoms']]]))
def syntax_outside_local(text,lineage,local):
 atoms,syntax,_=lex_smiles(text);events=[]
 for s in syntax:
  left=[i for i,a in enumerate(atoms)if a['span'][1]<=s['span'][0]]
  right=[i for i,a in enumerate(atoms)if a['span'][0]>=s['span'][1]]
  prev=lineage[left[-1]]if left else None;nxt=lineage[right[0]]if right else None
  if prev not in local and nxt not in local:events.append((s['kind'],s['text'],prev,nxt))
 return events
def controlled(original,product_text,lineage,original_keys,product_keys,site):
 shared=[i for i in lineage if i>=0]
 if shared!=sorted(shared):return False,['shared_atom_traversal_changed']
 local=set(site)|{-1}
 for old,new in enumerate(lineage):
  if new>=0 and original_keys[new]['r1']!=product_keys[old]['r1']:local.add(new)
 oa,_,_=lex_smiles(original);pa,_,_=lex_smiles(product_text);why=[]
 for i,src in enumerate(lineage):
  if src>=0 and src not in local and oa[src]['text']!=pa[i]['text']:why.append('nonlocal_atom_token_spelling_changed')
 if syntax_outside_local(original,list(range(len(oa))),local)!=syntax_outside_local(product_text,lineage,local):why.append('nonlocal_bond_branch_ring_syntax_changed')
 return not why,why
def prepare_one(cid,can,cb,mf):
 m=parsed(can);choices=sites(m);classes=[k for k,v in choices.items()if v]
 if not classes:return None,dict(cid=cid,accepted=False,reason='no_eligible_edit_site',candidate_counts={str(k):len(v)for k,v in choices.items()})
 kind=min(classes,key=lambda k:h(f'20261123/edit-class/{cid}/{k}'))
 site=min(choices[kind],key=lambda x:h(f'20261123/edit-site/{cid}/'+','.join(map(str,x))))
 state=dict(cid=cid,edit_class=kind,site=list(site),candidate_counts={str(k):len(v)for k,v in choices.items()},accepted=False)
 try:
  product,mapping,removed,added=edit(m,kind,site)
  canproduct=Chem.MolToSmiles(product,canonical=True,isomericSmiles=True)
  check,why=standardize(canproduct)
  if why or check!=canproduct:raise ValueError('product_qualification:'+(why or 'standardization_changes_target'))
  main,lineage=serialize(product,mapping)
  source=audit(can,cb,mf,str(cid));target=audit(main,cb,mf,str(cid)+'_edited')
  ok,reasons=controlled(can,main,lineage,source['keys'],target['keys'],site)
  # Canonical products carry graph lineage from writing the same edited graph.
  canorder=json.loads(product.GetProp('_smilesAtomOutputOrder')) if False else None
  cantext=Chem.MolToSmiles(product,canonical=True,isomericSmiles=True)
  canon_lineage=[mapping[i]for i in json.loads(product.GetProp('_smilesAtomOutputOrder'))]
  canrecord=audit(cantext,cb,mf,str(cid)+'_edited_canonical')
  shared_source=sorted(x for x in lineage if x>=0)
  product_inverse={old:new for new,old in enumerate(lineage)if old>=0}
  changed=[i for i in shared_source if source['keys'][i]['r1']!=target['keys'][product_inverse[i]]['r1']]
  unchanged=[i for i in shared_source if i not in changed]
  roles={str(i):('center'if i in site else 'direct_neighbor'if any(m.GetBondBetweenAtoms(i,j)is not None for j in site)else 'other')for i in shared_source}
  result=dict(cid=cid,edit_class=kind,site=list(site),source=source,controlled_product=target,canonical_product=canrecord,product_to_source=lineage,canonical_product_to_source=canon_lineage,source_to_product=product_inverse,shared_source_atoms=shared_source,changed_environment_atoms=changed,unchanged_environment_atoms=unchanged,atom_roles=roles,removed_source_atoms=removed,added_product_atoms=[i for i,x in enumerate(lineage)if x<0],main_string_control_pass=ok,main_exclusion_reasons=reasons,canonical_product_smiles=canproduct)
  state.update(accepted=True,main_string_control_pass=ok,main_exclusion_reasons=reasons,shared_atoms=len(shared_source),changed_atoms=len(changed),unchanged_atoms=len(unchanged),removed_atoms=len(removed),added_atoms=sum(x<0 for x in lineage))
  return result,state
 except Exception as exc:state['reason']=str(exc);return None,state
