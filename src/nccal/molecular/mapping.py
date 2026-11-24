"""Lightweight exact-input atom mapping; no torch import or model forward."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from tokenizers import Tokenizer

from . import semantic_mapper as _semantic


@dataclass(frozen=True)
class AtomRecord:
    molecule_id: str
    exact_smiles: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    special_tokens_mask: tuple[int, ...]
    anchor_token_indices: tuple[int, ...]  # RDKit graph_atom_id order, never token sort
    tokenizer_id: str
    audit: dict


class AtomMapper:
    """MTR's audited ByteLevel mapper and MoLFormer's exact-span WordLevel mapper."""

    def __init__(self, model_name: str, asset_dir=None):
        self.model_name = model_name
        if model_name == "chemberta_mtr":
            self.mapper = _semantic.SemanticMapper(asset_dir)
            self.max_tokens, self.pad_id, self.special_ids = 512, 0, {0, 11, 12, 13, 14, 591, 592}
        elif model_name == "molformer":
            self.tokenizer = Tokenizer.from_file(str(Path(asset_dir) / "tokenizer.json"))
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
            self.max_tokens, self.pad_id, self.special_ids = 202, 2, {0, 1, 2, 3, 2361}
        else:
            raise ValueError("Unknown model name")

    def audit(self, smiles: str, molecule_id: str = "synthetic") -> AtomRecord:
        if self.model_name == "chemberta_mtr":
            audit = self.mapper.audit_molecule(smiles, molecule_id=molecule_id, require_canonical=False)
            if not audit["accepted"]:
                raise ValueError(f"MTR mapping rejected: {audit['exclusion_reasons']}")
            atoms = sorted(audit["atoms"], key=lambda a: a["graph_atom_id"])
            if [a["graph_atom_id"] for a in atoms] != list(range(len(atoms))):
                raise ValueError("MTR graph atom IDs are not a bijection")
            positions = tuple(a["anchor_token_index"] for a in atoms)
            ids, mask, specials = (tuple(audit[k]) for k in ("input_ids", "attention_mask", "special_tokens_mask"))
        else:
            atoms, syntax, edges = _semantic.lex_smiles(smiles)
            graph = _semantic.graph_attribution(smiles, atoms, edges)
            encoding = self.tokenizer.encode(smiles, add_special_tokens=True)
            ids, mask, specials = tuple(encoding.ids), tuple(encoding.attention_mask), tuple(encoding.special_tokens_mask)
            if 2361 in ids:
                raise ValueError("MoLFormer unknown token; no identity repair")
            covered, ordinary, details = set(), {}, []
            last_end = 0
            for i, (token, offset, special) in enumerate(zip(encoding.tokens, encoding.offsets, specials)):
                start, end = offset
                if special:
                    if offset != (0, 0) or ids[i] not in self.special_ids:
                        raise ValueError("Unexpected MoLFormer special-token offset or ID")
                    continue
                if not (0 <= start < end <= len(smiles)) or start < last_end:
                    raise ValueError("Invalid/nonmonotonic token offset")
                if token != smiles[start:end] or self.tokenizer.id_to_token(ids[i]) != token:
                    raise ValueError("Token spelling differs from exact input span")
                span = set(range(start, end))
                if covered & span:
                    raise ValueError("Overlapping token offsets")
                covered |= span
                ordinary[(start, end)] = i
                last_end = end
            if covered != set(range(len(smiles))):
                raise ValueError("MoLFormer tokenizer silently omitted source characters")
            positions_list = [None] * len(atoms)
            for atom in atoms:
                span = tuple(atom["span"])
                if span not in ordinary:
                    raise ValueError("No unique complete-atom token at exact source span")
                token_index = ordinary[span]
                graph_id = atom["graph_atom_id"]
                if positions_list[graph_id] is not None:
                    raise ValueError("Duplicate graph atom ID")
                positions_list[graph_id] = token_index
                details.append({"graph_atom_id": graph_id, "anchor_token_index": token_index,
                                "full_atom_span": list(span), "element_span": atom["element_span"],
                                "source_atom_text": atom["text"], "graph_properties": atom["graph_properties"]})
            if any(p is None for p in positions_list):
                raise ValueError("Missing graph anchor")
            positions = tuple(positions_list)
            audit = {"accepted": True, "gate": "PASS_EXACT_ATOM_SPAN_AND_GRAPH_BIJECTION",
                     "exact_model_input": smiles, "molecule_id": molecule_id,
                     "graph_atom_count": len(atoms), "graph": graph,
                     "atoms": sorted(details, key=lambda x: x["graph_atom_id"]),
                     "offsets": [list(x) for x in encoding.offsets], "tokens": encoding.tokens,
                     "offset_convention": "zero-based half-open original ASCII character offsets"}
        if len(ids) > self.max_tokens:
            raise ValueError("Token window exceeded; truncation forbidden")
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("Empty/nonbijective graph anchors")
        if any(not mask[p] or specials[p] or ids[p] in self.special_ids for p in positions):
            raise ValueError("Anchor points to special/pad token")
        return AtomRecord(str(molecule_id), smiles, ids, mask, specials, positions, self.model_name, audit)


