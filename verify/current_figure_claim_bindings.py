"""Scientific bindings for the current six-panel molecular and protein figures."""
from pathlib import Path
import ast,csv,gzip,json
import numpy as np


def bindings(root):
    root=Path(root);out={}
    def rows(path,delimiter='\t'):
        with(root/path).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f,delimiter=delimiter))
    def load(path):return json.loads((root/path).read_text('utf8'))
    def put(ids,value,source,operation):
        for identifier in ids.split():out[identifier]=dict(value=float(value),source=source,operation=operation)
    def common(values):
        values=list(values)
        if not values or len(set(values))!=1:raise ValueError('Expected one common design quantity')
        return values[0]
    def one(records,**criteria):
        match=[r for r in records if all(r[k]==v for k,v in criteria.items())]
        if len(match)!=1:raise ValueError('Nonunique semantic selector '+repr(criteria))
        return match[0]
    art='results/artwork/use/'
    example5='results/artwork/examples/Fig5d_CID66277342_three_SMILES.json'
    example6='results/artwork/examples/Fig6d_Q9I398_E1_1900.json'
    j5=load(example5);j6=load(example6)

    source=art+'5a_and_s_r1_primary_metrics.tsv';metrics=rows(source)
    controlled=[r for r in metrics if r['representation']=='controlled']
    edit4=[r for r in controlled if int(r['edit_class'])in(1,2,3,4)]
    if len({(r['unit'],r['edit_class'])for r in edit4})!=len(edit4):raise ValueError('Repeated neuron-edit combination')
    units={r['unit']for r in edit4};edits={r['edit_class']for r in edit4}
    passed=[r for r in edit4 if float(r['mean_advantage_lo'])>0 and float(r['swap_advantage_lo'])>0 and float(r['unchanged_hi'])<.2]
    if len(passed)!=sum(r['better_than_both']=='True'and r['unchanged_small']=='True'for r in edit4):raise ValueError('Complete edit criterion and archived flags disagree')
    put('MS-2244 MS-2252',len(units),source,'Current Figure5c: distinct selected panel neurons in controlled edit classes1-4; fluorination is separate.')
    put('MS-2245',len(passed),source,'Current Figure5c: count controlled neuron/edit cells with both reference-advantage lower95% bounds>0 and unchanged-environment upper bound<0.2 development SD; this is the complete criterion.')
    put('MS-2246',len(edit4),source,'Current Figure5c: selected panel neurons times four functional-group edit classes, one controlled row per neuron/edit.')
    put('MS-2253',len(edits),source,'Current Figure5c: distinct functional-group edits1-4; exclude the distant fluorination control5.')
    hydroxyl=one(controlled,unit='293',edit_class='1')
    put('MS-0548',hydroxyl['changed_atoms'],source,'Current Figure5b: shared atoms with changed radius1 environment for controlled hydroxyl-to-hydrogen edits at neuron293.')
    put('MS-0549',hydroxyl['molecules'],source,'Current Figure5b: controlled hydroxyl-to-hydrogen mother/product pairs contributing those changed atoms.')

    drawing='figures/native/main_5.py';tree=ast.parse((root/drawing).read_text('utf8'))
    unit,=[ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='U'for t in n.targets)]
    if unit not in j5['measurement']['neuron_indices_0']:raise ValueError('Illustrative unit absent from the true per-atom export')
    put('MS-2249 MS-2256',unit,drawing,'Current Figure5a/d: renderer U selects the zero-based hydroxyl neuron, also present in the true per-atom JSON measurement identity.')
    pick=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='pick')
    atom_limit,=[ast.literal_eval(n.comparators[0])for n in ast.walk(pick)if isinstance(n,ast.Compare)and len(n.ops)==1 and isinstance(n.ops[0],ast.LtE)and isinstance(n.left,ast.Call)and isinstance(n.left.func,ast.Attribute)and n.left.func.attr=='GetNumAtoms']
    put('MS-2250',atom_limit,drawing,'Current Figure5a illustrative selection: source heavy-atom ceiling in pick(), applied before choosing the example closest to the all-pair median error; this is an illustrative selection rule.')
    raw_source='results/molecular/edits/raw_pairs.npz'
    with np.load(root/raw_source,allow_pickle=False)as z:
        offsets=z['offsets'];classes=z['edit_classes'];main=z['main_pass'];cids=z['cids'];changed=z['changed'];indices=z['source_atom_index']
        source_values=z['source'][:,unit];product_values=z['controlled_product'][:,unit];predicted=z['predicted_delta'][:,unit]
    inputs='data/artwork/examples/hydroxyl_pair_inputs.jsonl.gz'
    with gzip.open(root/inputs,'rt',encoding='utf8')as f:records={(int(r['cid']),int(r['edit_class'])):r for r in map(json.loads,f)}
    candidates=[]
    for p in range(len(classes)):
        if classes[p]!=1 or not main[p]:continue
        s=slice(offsets[p],offsets[p+1]);mask=changed[s]
        if not mask.any():continue
        k=int(np.argmax(np.abs(predicted[s])*mask));error=abs(float(product_values[s][k]-source_values[s][k]-predicted[s][k]))
        record=records[(int(cids[p]),1)];shared=set(map(int,indices[s]));removed=set(map(int,record['removed_source_atoms']))
        if shared&removed or len(shared)!=offsets[p+1]-offsets[p]or shared|removed!=set(range(len(shared)+len(removed))):raise ValueError('Shared/removed atom indices do not cover the parent graph')
        candidates.append(dict(error=error,pair=p,atom=int(indices[s][k]),cid=int(cids[p]),parent_atoms=len(shared)+len(removed),product_atoms=len(shared)))
    median=float(np.median([r['error']for r in candidates]));small=[r for r in candidates if max(r['parent_atoms'],r['product_atoms'])<=atom_limit]
    selected=min(small,key=lambda r:abs(r['error']-median))
    if selected['cid']!=int(j5['cid'])or selected['atom']!=j5['highlighted_site_from_figure_5a_edit']['canonical_carbon_atom_index_0']:raise ValueError('Current drawing no longer selects the exported molecule/marked atom')
    put('MS-2251',len(candidates),raw_source,'Current Figure5a: all controlled main-pass hydroxyl-to-hydrogen pairs used for the error median; for each, use the changed shared atom with largest absolute rule delta; verify the selected small molecule and carbon against the real export.')
    if not all(r['lineage_graph_properties_equal_canonical']for r in j5['checks']):raise ValueError('Exported SMILES graph identity checks failed')
    put('MS-2254',len(j5['encodings']),example5,'Current Figure5d: actual exported canonical and two frozen alternative encodings, without new sampling or model inference.')
    put('MS-2255',len({j5['cid']}),example5,'Current Figure5d: one exported CID; all supplied encodings pass the recorded canonical-lineage graph-identity checks.')

    source=art+'5b_and_s_q1_census.tsv';census=rows(source)
    put('MS-0551',sum(r['canonical_usable']=='True'for r in census),source,'Current Figure5e: count canonically usable native neurons before grouping rewritten outcomes.')
    put('MS-0553',len(j5['cohort']['available_frozen_rewrite_ids']),example5,'Current Figure5e: number of frozen additional notations in the underlying rewriting census, distinct from the three illustrative notations displayed in5d.')
    source=art+'5c_and_s_s1_basis_summary.tsv';basis=rows(source)
    plotted=[r for r in basis if not r['basis'].startswith('sae_')or r['subset']!='full']
    put('MS-0554',common(int(r['physical_n'])for r in plotted),source,'Current Figure5f: common physical width of all plotted native/rotation/PCA and variance-selected dictionary subsets; full928-feature dictionaries are not the plotted subsets.')
    put('MS-0555',len([r for r in plotted if r['basis'].startswith('rot_')]),source,'Current Figure5f: distinct stored rotated bases, one line per basis.')
    put('MS-0556',len({r['basis']for r in plotted if r['basis'].startswith('sae_')}),source,'Current Figure5f: distinct dictionary fits represented by their top464 variance-contribution subsets.')
    source='src/nccal/molecular/reproduce.py';basis_tree=ast.parse((root/source).read_text('utf8'))
    cutoff,=[n.comparators[0].value for n in ast.walk(basis_tree)if isinstance(n,ast.Compare)and len(n.ops)==1 and isinstance(n.ops[0],ast.GtE)and isinstance(n.left,ast.Name)and n.left.id=='r2'and isinstance(n.comparators[0],ast.Constant)and n.comparators[0].value==.5]
    put('MS-2258',cutoff,source,'Current Figure5f: the per-direction r2 comparison used to recompute R2_ge05_n in basis verification; no histogram cutoff is substituted.')

    source='data/artwork/generated/protein_exposure_pair_manifest.json';geometry=load(source);geometry=geometry[0]if isinstance(geometry,list)else geometry
    if geometry['pdb_id'].lower()!='2xmx'or geometry['chain']!='A':raise ValueError('The displayed exposure structure identity changed')
    sites=geometry['sites']
    put('MS-0630',len(sites),source,'Current Figure6a: actual selected same-amino-acid sites in the displayed2XMX/A structural rendering.')
    put('MS-0632',sum(k=='buried'for k in sites),source,'Current Figure6a: selected buried site in the actual image/site manifest.')
    put('MS-0633',sum(k=='exposed'for k in sites),source,'Current Figure6a: selected exposed site in the actual image/site manifest.')
    source=art+'6b_and_s_t1_public_labels.tsv';public=rows(source)
    publicT=[r for r in public if r['source']=='Hallee'and r['status']=='EVALUABLE'and r['frame']=='T']
    put('MS-2263',len(publicT),source,'Current Figure6b/c: all evaluable public confirmation label rows, irrespective of their test flags; one audit-matrix column per unit-label identity.')
    put('MS-2264',len({'same_AA_flag','strong_increment_flag'}),source,'Current Figure6c: the two separately recorded same-amino-acid discrimination and strong-sequence-increment test flags; accuracy is a separate descriptive row.')
    source='data/protein/sequence_identifiers.jsonl'
    population=[json.loads(s)for s in(root/source).read_text('utf8').splitlines()if s.strip()]
    put('MS-2265',len({r['group']for r in population if r['split']=='T'}),source,'Current Figure6c: confirmation mother-frame homology groups, not the usable denominator of every individual label.')

    put('MS-0592 MS-2266',j6['unit']['unit0'],example6,'Current Figure6d: the selected native E1 channel identity, retaining its original zero-based index.')
    put('MS-2268',j6['readout_definition']['block_number1'],example6,'Current Figure6d: exported one-based block number corresponding to zero-based model layer19.')
    if len(j6['selected']['sequence'])!=j6['selected']['full_model_input_sequence_length']:raise ValueError('Full-chain input length mismatch')
    put('MS-2271',len(j6['selected']['sequence']),example6,'Current Figure6d: full natural model-input sequence length of the selected Q9I398 chain; not its observed PDB construct length.')
    put('MS-2272',j6['selection_rule']['maximum_full_input_sequence_length'],example6,'Current Figure6d: author-requested post hoc display length ceiling, applied after computing the median over all correct pairs.')
    correct=j6['all_correct_pairs_used_for_median']
    if len(correct)!=j6['counts']['correct']:raise ValueError('Correct-pair illustrative median population mismatch')
    put('MS-2273',len(correct),example6,'Current Figure6d: all strictly correctly ordered frozen NEW pairs used for the illustrative median, before the length restriction.')

    source=art+'6c_and_s_t2_own_units.tsv';own=[r for r in rows(source)if r['model']=='e1_150m'and r['concept']=='OWN_HELIX_VS_BETA'and r['frame']=='NEW']
    put('MS-2274',len({(r['block0'],r['unit0'])for r in own}),source,'Current Figure6e: distinct development-fixed E1 physical channels on the same helix-versus-strand question and NEW frame.')
    put('MS-0637',common(int(r['groups'])for r in own),source,'Current Figure6e: common usable same-amino-acid NEW pair count for the four fixed E1 channels.')
    source=art+'6c_and_s_t2_references_new.tsv'
    references=[r for r in rows(source)if r['model']=='e1_150m'and r['concept']=='OWN_HELIX_VS_BETA'and r['frame']=='NEW'and r['original_family']=='native_signed'and r['masked']=='False']
    random={r['reference']for r in references if r['reference'].startswith('random_')};rotations={r['reference']for r in references if r['reference'].startswith('rotation_')}
    put('MS-0638',len(random),source,'Current Figure6e: distinct untrained networks in the exact E1/OWN_HELIX_VS_BETA/native_signed/unmasked/NEW reference family shared by the four channels.')
    put('MS-0639',len(rotations),source,'Current Figure6e: distinct rotated bases in that same exact E1 reference family; no exposure-network family is substituted.')
    source=art+'6d_network_results.tsv';net=rows(source)
    put('MS-0641',common(len({r['index']for r in net if r['model']==model and r['network_type']=='untrained'})for model in('8m','35m','650m')),source,'Current Figure6f: actual distinct untrained network indices per displayed ESM-2 model in the original exposure comparison; this is not the E1 family.')

    source='src/nccal/metrics.py';tree=ast.parse((root/source).read_text('utf8'))
    cutoffs=[n.comparators[0].value for n in ast.walk(tree)if isinstance(n,ast.Compare)and len(n.ops)==1 and isinstance(n.ops[0],ast.GtE)and isinstance(n.left,ast.Call)and isinstance(n.left.func,ast.Name)and n.left.func.id=='float'and n.left.args and isinstance(n.left.args[0],ast.Subscript)and isinstance(n.left.args[0].slice,ast.Constant)and n.left.args[0].slice.value=='auc_mean'and isinstance(n.comparators[0],ast.Constant)]
    put('MS-0573',common(cutoffs),source,'Manuscript public-label subset cutoff: auc_mean >=0.70 in public.auroc70.eligible; this is a descriptive subset definition, not a line in the current Figure6b or a BH test threshold.')
    # Overview caption: selectors follow the actual fixed examples.
    f1mol=load(example5)
    assert f1mol["cid"]==66277342
    f1geom=load("data/artwork/protein_masked_geometry.json")
    f1pair=load(example6)["selected"]["pair"]
    assert f1pair["accession"]=="Q9I398" and f1pair["amino_acid"]=="R"
    assert f1pair["helix_position1"]!=f1pair["strand_position1"]
    f1exposure=load("results/artwork/main/exposure_example.json")
    assert f1exposure["selection"]["model"]=="35m"
    put('MS-2282',len({f1mol["cid"]}),'results/artwork/examples/Fig5d_CID66277342_three_SMILES.json','Figure 1f: count the single fixed molecule in the three-notation example.')
    put('MS-2283',len(f1mol["encodings"]),'results/artwork/examples/Fig5d_CID66277342_three_SMILES.json','Figure 1f: count the saved SMILES encodings in the fixed molecular example.')
    put('MS-2285',len({f1geom['chain']}),'data/artwork/protein_masked_geometry.json','Figure 1g: count the single structural chain used for the illustrative alanine pair.')
    put('MS-2288',len({f1pair["helix_position1"],f1pair["strand_position1"]}),'results/artwork/examples/Fig6d_Q9I398_E1_1900.json','Figure 1g: count the distinct arginine sites in the exact frozen Q9I398 example.')
    put('MS-2292',f1exposure["selection"]["axis_zero_based"],'results/artwork/main/exposure_example.json','Figure 1g: read the development-selected ESM-2 35M exposure neuron.')
    return out
