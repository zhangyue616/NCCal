"""E1: InterPLM's concept test, re-run with the units ESM-2 computes.

InterPLM (Simon and Zou) scored ESM-2's sparse-autoencoder latents and its neurons against Swiss-Prot concepts and found
up to 2,548 latents per layer associated with a concept against 46 neurons; its neurons were the dimensions of the
residual stream, which no nonlinearity privileges. This module scores, on InterPLM's evaluation and with its metric, the
units the model computes: the feed-forward intermediate read as the two poles of its pre-activation (its positive and
negative parts), beside InterPLM's residual dimensions, the residual dimensions read as poles, three Haar rotations of
the intermediate read the same way, the intermediate of a copy of the model whose every weight and bias tensor has its
values shuffled (InterPLM's randomized control), and InterPLM's released SAEs.

The evaluation is the one InterPLM's paper describes and its code at commit 5f4cbf9 implements: concept labels from
`interplm.analysis.concepts` on 50,000 Swiss-Prot proteins in 50 shards, the first 25 for validation and the rest for
test; each feature scaled to [0, 1] by its maximum over the 50,000 proteins (InterPLM's residual dimensions by their
minimum and maximum, as it scaled neurons); thresholds 0, 0.15, 0.5, 0.6 and 0.8; per concept, feature and threshold the
residues where the feature exceeds the threshold, those of them the concept labels, and for a concept spanning domains
the domains with any such residue; precision, recall per domain (per residue for residue-level concepts) and their F1.
Each concept's best feature on validation is scored on test, and a feature is associated with a concept when that F1
exceeds 0.5 on both splits.

The counting, F1 and report here are matrix forms of InterPLM's `calc_metrics_dense`, `calculate_metrics` and
`report_metrics` that give the same numbers (the stage checks them against InterPLM's functions on real data first), so
arms of thousands of dense units fit the budget. Ties go to the lower feature and threshold. The pinned
`compare_activations` reads features without the scaling `normalize.py` stores in a buffer; this module scales every arm
itself.

E13, exploratory and added after H0 to H7 were decided, gives ESMC the same evaluation (`stream_esmc_arms`): at each
block, the stream's dimensions and their poles, the feed-forward units' gate-tagged poles (the atlas's native basis) and
sign poles, the sign poles of rotated units, and the SAEs at the block, every one read as `bases.basis_activations`
reads it for the atlas. E13s adds the units in the standardized coordinates of E8's sparsest-rotation search, read
alone and rotated by each search's Haar start and by the rotation it found (`SearchedBases`).

Symbols: n residues; d model width; m intermediate width; F features; C concepts; T thresholds; K domain instances.
"""

from __future__ import annotations

import copy
import functools
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from torch import Tensor, nn

from .atlas import encode_batch, transformer_units
from .bases import Basis, basis_activations
from .capture import BlockReader, BlockStates
from .poles import sign_poles
from .rotation_search import Standardization
from .rotations import haar_rotation, rotate


THRESHOLDS = (0.0, 0.15, 0.5, 0.6, 0.8)
ASSOCIATION_F1 = 0.5
SPLITS = ("valid", "test")  # InterPLM's names
REPORT_EXCLUDES = ("amino_acid",)  # InterPLM's report leaves out concepts whose name contains these
ROTATION_SEEDS = (0, 1, 2)
SHUFFLE_SEED = 24895
TOKEN_BUDGET = 32_768
FEATURE_CHUNK = 2048  # features per segment maximum over domain instances
MIN_MAX_ARMS = ("residual",)  # scaled by minimum and maximum, as InterPLM scaled neurons; every other arm by its maximum
SUBSETS = {"mlp_positive": ("mlp_poles", 0.5)}  # an arm's first share of features read alone: the poles' GELU side
ESM2 = {  # model -> (Hugging Face repository, layers with a released InterPLM SAE, InterPLM's name for it)
    "esm2_8m": ("facebook/esm2_t6_8M_UR50D", (1, 2, 3, 4, 5, 6), "esm2-8m"),
    "esm2_650m": ("facebook/esm2_t33_650M_UR50D", (1, 9, 18, 24, 30, 33), "esm2-650m"),
}
UNIPROT_FIELDS = ("accession,reviewed,protein_name,length,sequence,ec,ft_act_site,ft_binding,cc_cofactor,ft_disulfid,"
                  "ft_carbohyd,ft_lipid,ft_mod_res,ft_signal,ft_transit,ft_helix,ft_turn,ft_strand,ft_coiled,cc_domain,"
                  "ft_compbias,ft_domain,ft_motif,ft_region,ft_zn_fing,xref_alphafolddb")  # InterPLM's README, all Swiss-Prot
UNIPROT_STREAM = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&query=%28reviewed%3Atrue%29&fields="
                  + UNIPROT_FIELDS.replace(",", "%2C"))
EVAL_PROTEINS = 50_000  # InterPLM's paper: 50,000 Swiss-Prot proteins under 1,024 residues, halved into validation and test
SHARDS = 50
MAX_LENGTH = 1022  # InterPLM's `preprocess_proteins` cap, the longest ESM-2 reads
MIN_INSTANCES = 100  # InterPLM's `extract_annotations` default for a sub-category to become a concept
MIN_RESIDUES, MIN_DOMAINS = 1500, 10  # InterPLM's paper: a concept needs more than 1,500 residues or 10 domains
PILOT_BATCHES = 23  # 3 warmup and 20 timed, as every pilot here
PARITY_FEATURES = 64  # features of the first layer's residual arm checked against InterPLM's own functions


@dataclass(frozen=True)
class RetestShard:
    split: str  # "valid" or "test"
    sequences: tuple[str, ...]  # the shard's proteins, in the order of its label rows
    labels: sp.csr_matrix  # (n, C), InterPLM's per-residue values of the evaluated concepts

    def __post_init__(self) -> None:
        residues = sum(len(sequence) for sequence in self.sequences)
        if self.labels.shape[0] != residues:
            raise ValueError(f"The shard's labels have {self.labels.shape[0]} rows for {residues} residues.")


@dataclass(frozen=True)
class ConceptSet:
    names: tuple[str, ...]
    domain_level: np.ndarray  # (C,) bool: recall counts domains (InterPLM's `not is_aa_level_concept`)
    positives: dict[str, np.ndarray]  # split -> (C,) labeled residues, InterPLM's n_positive_aa_per_concept
    domains: dict[str, np.ndarray]  # split -> (C,) labeled domains, InterPLM's n_positive_domains_per_concept

    @property
    def reported(self) -> np.ndarray:
        """(C,) bool: the concepts InterPLM's report keeps."""
        return np.array([not any(word in name.lower() for word in REPORT_EXCLUDES) for name in self.names])  # (C,)


@dataclass(frozen=True)
class BatchLabels:
    presence: Tensor  # (n, C) float32, 1 where the residue carries the concept
    residues: Tensor  # (q,) the residue of each labeled residue-domain pair
    instances: Tensor  # (q,) its domain instance, numbered within the batch
    owner: Tensor  # (K,) each instance's concept


def batch_labels(labels: sp.csr_matrix, domain_level: np.ndarray, device: str | torch.device = "cpu") -> BatchLabels:
    """A batch's concept labels as residue presence and, for domain-level concepts, domain instances: InterPLM numbers
    a concept's domains within a shard, so a (concept, value) pair is one domain."""
    # labels: (n, C) for the batch's residues; domain_level: (C,)
    entries = labels.tocoo()
    presence = torch.zeros(labels.shape, dtype=torch.float32, device=device)  # (n, C)
    presence[torch.as_tensor(entries.row, device=device), torch.as_tensor(entries.col, device=device)] = 1.0
    spans = domain_level[entries.col] & (entries.data > 0)  # (nnz,)
    rows, columns, values = entries.row[spans], entries.col[spans].astype(np.int64), entries.data[spans].astype(np.int64)
    stride = int(values.max()) + 1 if len(values) else 1
    keys, instances = np.unique(columns * stride + values, return_inverse=True)  # (K,), (q,)
    return BatchLabels(presence, torch.as_tensor(rows, dtype=torch.int64, device=device),
                       torch.as_tensor(instances.reshape(-1), dtype=torch.int64, device=device),
                       torch.as_tensor(keys // stride, dtype=torch.int64, device=device))


class ConceptCounts:
    """Running sums for one arm at one layer and split: per concept, feature and threshold the labeled residues where
    the feature exceeds the threshold (InterPLM's tp) and the domains with any (its tp_per_domain); per feature and
    threshold the residues where it does (its tp + fp)."""

    def __init__(self, concepts: int, features: int, *, thresholds: Sequence[float] = THRESHOLDS,
                 device: str | torch.device = "cpu") -> None:
        self.thresholds = tuple(thresholds)
        shape = (concepts, features, len(self.thresholds))
        self.positive = torch.zeros(shape, dtype=torch.int32, device=device)  # (C, F, T); a split has under 2**31 residues
        self.domains = torch.zeros(shape, dtype=torch.int32, device=device)  # (C, F, T)
        self.fired = torch.zeros((features, len(self.thresholds)), dtype=torch.int64, device=device)  # (F, T)

    def update(self, scaled: Tensor, labels: BatchLabels, *, chunk: int = FEATURE_CHUNK) -> None:
        """Add a batch of residues: each feature's scaled activations and the residues' labels."""
        # scaled: (n, F)
        for index, threshold in enumerate(self.thresholds):
            on = (scaled > threshold).float()  # (n, F)
            self.fired[:, index] += on.sum(dim=0).round().long()
            self.positive[:, :, index] += (labels.presence.T @ on).round().int()  # (C, F); exact below 2**24 residues
        if labels.owner.numel() == 0:
            return
        for start in range(0, scaled.shape[1], chunk):
            gathered = scaled[labels.residues, start:start + chunk]  # (q, c)
            peak = torch.full((labels.owner.numel(), gathered.shape[1]), -1.0, dtype=gathered.dtype, device=gathered.device)
            peak.scatter_reduce_(0, labels.instances[:, None].expand_as(gathered), gathered, reduce="amax")  # (K, c)
            for index, threshold in enumerate(self.thresholds):
                self.domains[:, start:start + chunk, index].index_add_(0, labels.owner, (peak > threshold).int())


def f1_scores(counts: ConceptCounts, concepts: ConceptSet, split: str) -> tuple[Tensor, Tensor, Tensor]:
    """InterPLM's `calculate_metrics` for every concept, feature and threshold at once: whether the entry exists (a
    true positive), its F1 and its F1 per domain."""
    positive = counts.positive.double()  # (C, F, T)
    present = positive > 0  # (C, F, T)
    precision = torch.where(present, positive / counts.fired.double().clamp(min=1)[None], torch.zeros_like(positive))
    labeled = torch.as_tensor(concepts.positives[split], dtype=torch.float64, device=positive.device)[:, None, None]
    recall = torch.where(labeled > 0, positive / labeled.clamp(min=1), torch.zeros_like(positive))  # (C, F, T)
    domains = torch.as_tensor(concepts.domains[split], dtype=torch.float64, device=positive.device)[:, None, None]
    per_domain = torch.where(domains > 0, counts.domains.double() / domains.clamp(min=1), torch.zeros_like(positive))
    spans = torch.as_tensor(concepts.domain_level, device=positive.device)[:, None, None]  # (C, 1, 1)
    recall_domain = torch.where(spans, per_domain, recall)  # (C, F, T)

    def harmonic(first: Tensor, second: Tensor) -> Tensor:
        # first, second: (C, F, T)
        total = first + second
        return torch.where(total > 0, 2 * first * second / total.clamp(min=1e-300), torch.zeros_like(total))  # (C, F, T)

    return present, harmonic(precision, recall), harmonic(precision, recall_domain)  # (C, F, T) each


@dataclass(frozen=True)
class RetestReport:
    features: int  # features scored
    best_feature: np.ndarray  # (C,) each reported concept's best validation feature, -1 where it has none
    best_test_f1: np.ndarray  # (C,) that feature's test F1 per domain, NaN where InterPLM's report drops the concept
    pairs: np.ndarray  # (q, 2) associated (feature, concept) pairs

    @property
    def associated_features(self) -> int:
        return int(len(np.unique(self.pairs[:, 0])))

    @property
    def identified_concepts(self) -> int:
        return int(len(np.unique(self.pairs[:, 1])))


def _best_entries(present: Tensor, f1: Tensor, f1_domain: Tensor) -> tuple[Tensor, Tensor]:
    """Per row of (rows, entries) arrays, the entry with the highest F1 per domain, then F1, then lowest index; and
    whether the row has any entry."""
    # present, f1, f1_domain: (r, e)
    primary = torch.where(present, f1_domain, torch.full_like(f1_domain, -1.0))  # (r, e)
    leaders = present & (primary == primary.amax(dim=1, keepdim=True))  # (r, e)
    secondary = torch.where(leaders, f1, torch.full_like(f1, -1.0))  # (r, e)
    finalists = leaders & (secondary == secondary.amax(dim=1, keepdim=True))  # (r, e)
    return finalists.float().argmax(dim=1), present.any(dim=1)  # (r,), (r,): argmax returns the first maximum


def interplm_report(valid: tuple[Tensor, Tensor, Tensor], test: tuple[Tensor, Tensor, Tensor], reported: np.ndarray, *,
                    features: np.ndarray | None = None, cut: float = ASSOCIATION_F1) -> RetestReport:
    """InterPLM's `report_metrics` on (present, F1, F1 per domain) arrays of both splits, optionally for a subset of
    the features: each reported concept's best validation feature scored on test at its best test threshold, and the
    pairs above `cut` on both splits."""
    # valid, test: (C, F, T) each; reported: (C,) bool; features: (F_subset,)
    if features is not None:
        index = torch.as_tensor(features, dtype=torch.int64, device=valid[0].device)  # (F_subset,)
        valid, test = (tuple(array[:, index] for array in valid), tuple(array[:, index] for array in test))
    concepts, width, thresholds = valid[0].shape
    keep = torch.as_tensor(reported, device=valid[0].device)[:, None, None]  # (C, 1, 1)
    valid = (valid[0] & keep, valid[1], valid[2])
    test = (test[0] & keep, test[1], test[2])
    flat = tuple(array.reshape(concepts, width * thresholds) for array in valid)  # (C, F * T) each
    chosen, found = _best_entries(*flat)  # (C,), (C,)
    feature = chosen // thresholds  # (C,)
    rows = torch.arange(concepts, device=feature.device)  # (C,)
    picked = tuple(array[rows, feature] for array in test)  # (C, T) each
    position, scored = _best_entries(*picked)  # (C,), (C,)
    value = picked[2][rows, position]  # (C,)
    usable = found & scored  # (C,)
    best_test = torch.where(usable, value, torch.full_like(value, float("nan")))  # (C,)
    passed = (valid[0] & (valid[2] > cut)).any(dim=2) & (test[0] & (test[2] > cut)).any(dim=2)  # (C, F)
    concept_index, feature_index = torch.nonzero(passed, as_tuple=True)  # (q,), (q,)
    pairs = torch.stack((feature_index, concept_index), dim=1).cpu().numpy()  # (q, 2)
    if features is not None:
        pairs[:, 0] = np.asarray(features)[pairs[:, 0]]
    best = torch.where(found, feature, torch.full_like(feature, -1)).cpu().numpy()  # (C,)
    if features is not None:
        best = np.where(best >= 0, np.asarray(features)[np.maximum(best, 0)], -1)
    return RetestReport(width, best, best_test.cpu().numpy(), pairs)


def poles(values: Tensor) -> Tensor:
    """Each column's positive and negative parts, side by side."""
    # values: (n, w)
    return torch.cat((values.clamp(min=0), (-values).clamp(min=0)), dim=1)  # (n, 2w)


def shuffled_copy(model: nn.Module, seed: int = SHUFFLE_SEED) -> nn.Module:
    """InterPLM's randomized control: a copy whose every weight and bias tensor has its values permuted."""
    shuffled = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in shuffled.parameters():
            flat = parameter.data.reshape(-1)  # (p,)
            order = torch.randperm(flat.numel(), generator=generator).to(flat.device)  # (p,)
            parameter.data.copy_(flat[order].reshape(parameter.shape))
    return shuffled


class EsmTaps:
    """Hooks on a Hugging Face ESM-2 masked language model that keep, for the real residues of a batch, the residual
    stream leaving each chosen layer (after the final layer norm at the last one, as `hidden_states` reported it when
    InterPLM extracted it) and the pre-activation of the layer's intermediate."""

    def __init__(self, model: nn.Module, layers: Sequence[int]) -> None:
        self.model, self.layers = model, tuple(layers)
        self.encoder = model.esm.encoder
        self.final = len(self.encoder.layer)
        if not all(1 <= layer <= self.final for layer in self.layers):
            raise ValueError(f"Layers {self.layers} fall outside 1 to {self.final}.")
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.captured: dict[tuple[str, int], Tensor] = {}

    def __enter__(self) -> EsmTaps:
        for layer in self.layers:
            block = self.encoder.layer[layer - 1]
            self.handles.append(block.register_forward_hook(self._keep("residual", layer)))
            self.handles.append(block.intermediate.dense.register_forward_hook(self._keep("pre", layer)))
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _keep(self, kind: str, layer: int) -> Callable[[nn.Module, object, object], None]:
        def hook(_module: nn.Module, _inputs: object, output: object) -> None:
            self.captured[kind, layer] = output[0] if isinstance(output, tuple) else output  # (b, l, w)
        return hook

    @torch.no_grad()
    def run(self, input_ids: Tensor, attention_mask: Tensor, residue_mask: Tensor) -> dict[int, tuple[Tensor, Tensor]]:
        """Each chosen layer's (residual, pre-activation) at the batch's residues, in reading order."""
        # input_ids, attention_mask, residue_mask: (b, l)
        self.captured.clear()
        self.model(input_ids=input_ids, attention_mask=attention_mask)
        states = {}
        for layer in self.layers:
            residual = self.captured["residual", layer]  # (b, l, d)
            if layer == self.final and self.encoder.emb_layer_norm_after is not None:
                residual = self.encoder.emb_layer_norm_after(residual)
            states[layer] = (residual[residue_mask].float(), self.captured["pre", layer][residue_mask].float())  # (n, d), (n, m)
        return states  # layer -> ((n, d), (n, m))


def encode_sequences(tokenizer: Callable[..., Mapping[str, Tensor]], sequences: Sequence[str],
                     device: str | torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Token ids, attention mask and the mask of real residues (every position but the start, end and padding)."""
    # tokenizer: (b, l) ids and attention mask from the sequences
    tokens = tokenizer(list(sequences), return_tensors="pt", padding=True)
    input_ids, attention_mask = tokens["input_ids"].to(device), tokens["attention_mask"].to(device)  # (b, l)
    lengths = torch.as_tensor([len(sequence) for sequence in sequences], device=device)  # (b,)
    positions = torch.arange(input_ids.shape[1], device=device)[None, :]  # (1, l)
    residue_mask = (positions >= 1) & (positions <= lengths[:, None])  # (b, l)
    if int(residue_mask.sum()) != int(lengths.sum()):
        raise ValueError("The tokenizer did not give one token per residue.")
    return input_ids, attention_mask, residue_mask  # (b, l) each


@torch.no_grad()
def arm_values(residual: Tensor, pre: Tensor, *, rotations: Mapping[int, Tensor],
               sae: Callable[[Tensor], Tensor] | None) -> dict[str, Tensor]:
    """One layer's arms on a batch: residual dimensions, residual poles, intermediate poles, rotated intermediate
    poles and SAE latents."""
    # residual: (n, d); pre: (n, m); rotations: (m, m) each; sae: (n, d) -> (n, c)
    arms = {"residual": residual, "residual_poles": poles(residual), "mlp_poles": poles(pre)}  # (n, d), (n, 2d), (n, 2m)
    for seed, rotation in rotations.items():
        arms[f"mlp_rotation{seed}"] = poles(pre @ rotation)  # (n, 2m)
    if sae is not None:
        arms["sae"] = sae(residual).float()  # (n, c)
    return arms  # arm -> (n, F)


class FeatureRanges:
    """Each feature's minimum and maximum over every residue it is given, and its activations scaled by them."""

    def __init__(self) -> None:
        self.low: dict[tuple[int, str], Tensor] = {}
        self.high: dict[tuple[int, str], Tensor] = {}

    def update(self, key: tuple[int, str], values: Tensor) -> None:
        # values: (n, F)
        low, high = values.amin(dim=0), values.amax(dim=0)  # (F,), (F,)
        self.low[key] = torch.minimum(self.low[key], low) if key in self.low else low
        self.high[key] = torch.maximum(self.high[key], high) if key in self.high else high

    def scale(self, key: tuple[int, str], values: Tensor) -> Tensor:
        """Activations divided by the feature's maximum, or for the min-max arms placed between its minimum and
        maximum; a feature that never varies scales to zero."""
        # values: (n, F)
        floor = self.low[key] if key[1] in MIN_MAX_ARMS else torch.zeros_like(self.low[key])  # (F,)
        span = self.high[key] - floor  # (F,)
        return torch.where(span > 0, (values - floor) / span.clamp(min=1e-30), torch.zeros_like(values))  # (n, F)


def protein_batches(lengths: Sequence[int], token_budget: int = TOKEN_BUDGET) -> list[np.ndarray]:
    """Proteins grouped longest first so a batch's padded tokens, its start and end included, stay within
    `token_budget`."""
    order = np.argsort(-np.asarray(lengths), kind="stable")  # (p,)
    batches, current = [], []
    for protein in order:
        if current and (lengths[current[0]] + 2) * (len(current) + 1) > token_budget:
            batches.append(np.array(current))
            current = []
        current.append(int(protein))
    if current:
        batches.append(np.array(current))
    return batches  # (p_b,) each


@dataclass(frozen=True)
class RetestBatch:
    shard: int  # index of the shard
    batch: int  # index of the batch within its shard
    split: str
    rows: np.ndarray  # (n,) the batch's residues' label rows, in the order its activations come
    layer: int
    arms: dict[str, Tensor]  # arm -> (n, F)


def stream_arms(
    model: nn.Module,
    shuffled: nn.Module,
    tokenizer: Callable[..., Mapping[str, Tensor]],
    layers: Sequence[int],
    shards: Sequence[RetestShard],
    saes: Mapping[int, Callable[[Tensor], Tensor]],
    *,
    device: str | torch.device,
    token_budget: int = TOKEN_BUDGET,
    rotation_seeds: Sequence[int] = ROTATION_SEEDS,
    batches: Callable[[Iterable[tuple[int, int, np.ndarray]]], Iterable[tuple[int, int, np.ndarray]]] | None = None,
) -> Iterator[RetestBatch]:
    """Every arm of every layer, layer by layer within each batch of the shards' proteins; the shuffled model adds its
    intermediate's poles as the arm `shuffled_mlp_poles`. `batches`, given the (shard, batch, proteins) plan, may
    wrap it, as a pilot's `watch` does."""
    # tokenizer: (b, l) ids and attention mask from the sequences; saes: (n, d) -> (n, c) by layer; batches: (p_b,)
    # proteins in each plan entry
    width = model.esm.encoder.layer[layers[0] - 1].intermediate.dense.out_features
    rotations = {seed: haar_rotation(width, seed, dtype=torch.float32, device=device) for seed in rotation_seeds}  # (m, m)
    plan = [(number, index, proteins) for number, shard in enumerate(shards)
            for index, proteins in enumerate(protein_batches([len(sequence) for sequence in shard.sequences], token_budget))]
    offsets = [np.concatenate(([0], np.cumsum([len(sequence) for sequence in shard.sequences]))) for shard in shards]
    with EsmTaps(model, layers) as trained, EsmTaps(shuffled, layers) as control:
        for number, index, proteins in batches(plan) if batches is not None else plan:
            shard = shards[number]
            inputs = encode_sequences(tokenizer, [shard.sequences[protein] for protein in proteins], device)
            rows = np.concatenate([np.arange(offsets[number][protein], offsets[number][protein + 1]) for protein in proteins])  # (n,)
            states, control_states = trained.run(*inputs), control.run(*inputs)
            for layer in layers:
                arms = arm_values(*states[layer], rotations=rotations, sae=saes.get(layer))
                arms["shuffled_mlp_poles"] = poles(control_states[layer][1])  # (n, 2m)
                yield RetestBatch(number, index, shard.split, rows, layer, arms)


@dataclass(frozen=True)
class SearchedBases:
    """E13s at one block: the standardization E8's sparsest-rotation search ran in, and the rotations read in it by arm
    suffix (`start<seed>`, a search's Haar start; `searched<seed>`, the rotation it found)."""

    standardization: Standardization  # mean and scale, (f,) each
    rotations: Mapping[str, Tensor]  # suffix -> (f, f)


def standardized_poles(states: BlockStates, standardization: Standardization, rotation: Tensor | None = None) -> Tensor:
    """The units' sign poles in the search's standardized coordinates, rotated first when a rotation is given."""
    # rotation: (f, f)
    standard = standardization.apply(states.product())  # (n, f)
    return sign_poles(standard if rotation is None else rotate(standard, rotation))  # (n, 2f)


@torch.no_grad()
def esmc_arm_values(states: BlockStates, *, rotations: Mapping[int, Tensor], saes: Mapping[str, nn.Module],
                    control: BlockStates | None = None, sae_reads: Mapping[str, str] | None = None,
                    arms: Collection[str] | None = None, searched: SearchedBases | None = None) -> dict[str, Tensor]:
    """E13: one ESMC block's arms on a batch, each as the atlas reads its basis: the stream's dimensions, as InterPLM
    read neurons, and their poles; the units' gate-tagged poles (`mlp_native`) and their sign poles (`mlp_poles`, the
    reading the rotations and E1's ESM-2 arms share); the sign poles of each rotation of the units; each SAE's latents,
    from the stream or, where `sae_reads` names it, the feed-forward write; given the shuffled copy's states at the
    block, its units' sign poles (`shuffled_mlp_poles`); and, given E13s's bases, the units' sign poles in the search's
    standardized coordinates (`mlp_standard`) and after each of its rotations (`mlp_<suffix>`). `arms`, when given,
    keeps those arms beside every SAE's."""
    # states, control: a block's gate and up (n, f), stream and write (n, d); rotations: (f, f) each
    block = states.block
    readers: dict[str, Callable[[], Tensor]] = {
        "residual": lambda: states.stream.float(),  # (n, d)
        "residual_poles": functools.partial(basis_activations, Basis("residual", block), states),  # (n, 2d)
        "mlp_native": functools.partial(basis_activations, Basis("native", block), states),  # (n, 2f)
        "mlp_poles": functools.partial(basis_activations, Basis("sign", block), states),  # (n, 2f)
    }
    for seed, rotation in rotations.items():
        readers[f"mlp_rotation{seed}"] = functools.partial(basis_activations, Basis("rotation", block, seed=seed), states,
                                                           rotation=rotation)  # (n, 2f)
    if control is not None:
        readers["shuffled_mlp_poles"] = functools.partial(basis_activations, Basis("sign", block), control)  # (n, 2f)
    if searched is not None:
        readers["mlp_standard"] = functools.partial(standardized_poles, states, searched.standardization)  # (n, 2f)
        for suffix, rotation in searched.rotations.items():
            readers[f"mlp_{suffix}"] = functools.partial(standardized_poles, states, searched.standardization, rotation)
    if arms is not None and not set(arms) <= set(readers):
        raise ValueError(f"Arms {sorted(set(arms) - set(readers))} are not read at block {block}.")
    values = {arm: read() for arm, read in readers.items() if arms is None or arm in arms}
    for name, sae in saes.items():
        basis = Basis("sae", block, sae=name, reads=(sae_reads or {}).get(name, "stream"))
        values[f"sae:{name}"] = basis_activations(basis, states, sae=sae)  # (n, c)
    return values  # arm -> (n, F)


def _keeper(store: dict[int, BlockStates]) -> Callable[[BlockStates], None]:
    """A `BlockReader` consumer that copies each block's states, which the reader releases when the pass ends; the
    copies are made outside the pass's inference mode, so later reads of them are ordinary tensor operations."""
    def keep(states: BlockStates) -> None:
        with torch.inference_mode(False):
            stream = None if states.stream is None else states.stream.clone()
            write = None if states.write is None else states.write.clone()
            store[states.block] = BlockStates(states.block, states.gate.clone(), states.up.clone(), write=write, stream=stream)
    return keep


def stream_esmc_arms(
    model: nn.Module,
    shuffled: nn.Module | None,
    blocks: Sequence[int],
    shards: Sequence[RetestShard],
    saes: Mapping[int, Mapping[str, nn.Module]],
    *,
    device: str | torch.device,
    token_budget: int = TOKEN_BUDGET,
    rotation_seeds: Sequence[int] = ROTATION_SEEDS,
    sae_reads: Mapping[str, str] | None = None,
    arms: Collection[str] | None = None,
    searched: Mapping[int, SearchedBases] | None = None,
    batches: Callable[[Iterable[tuple[int, int, np.ndarray]]], Iterable[tuple[int, int, np.ndarray]]] | None = None,
) -> Iterator[RetestBatch]:
    """E13: every arm of every block of an ESMC model (`esmc_arm_values`, with its `sae_reads`, `arms` and each block's
    `searched` bases), block by block within each batch of the shards' proteins, each block's states captured in one
    pass; a shuffled copy, when given, adds its units' sign poles as `shuffled_mlp_poles`. A batch's `layer` is the
    0-based block. `batches` wraps the plan as in `stream_arms`."""
    # saes: block -> name -> SAE layer reading the block's output stream, or its feed-forward write by `sae_reads`;
    # batches: (p_b,) proteins in each plan entry
    units = transformer_units(model)
    rotations = {seed: haar_rotation(units, seed, dtype=torch.float32, device=device) for seed in rotation_seeds}  # (f, f)
    writes = [block for block, named in saes.items() if any((sae_reads or {}).get(name) == "write" for name in named)]
    plan = [(number, index, proteins) for number, shard in enumerate(shards)
            for index, proteins in enumerate(protein_batches([len(sequence) for sequence in shard.sequences], token_budget))]
    offsets = [np.concatenate(([0], np.cumsum([len(sequence) for sequence in shard.sequences]))) for shard in shards]
    trained_states: dict[int, BlockStates] = {}
    control_states: dict[int, BlockStates] = {}
    with ExitStack() as stack:
        trained = stack.enter_context(BlockReader(model, blocks, _keeper(trained_states), write_blocks=writes,
                                                  stream_blocks=blocks))
        control = None if shuffled is None else stack.enter_context(BlockReader(shuffled, blocks, _keeper(control_states)))
        for number, index, proteins in batches(plan) if batches is not None else plan:
            shard = shards[number]
            inputs = encode_batch(model.tokenizer, [shard.sequences[protein] for protein in proteins], torch.device(device))
            rows = np.concatenate([np.arange(offsets[number][protein], offsets[number][protein + 1]) for protein in proteins])  # (n,)
            trained_states.clear()
            trained.run(*inputs)
            if control is not None:
                control_states.clear()
                control.run(*inputs)
            for block in blocks:
                values = esmc_arm_values(trained_states[block], rotations=rotations, saes=saes.get(block, {}),
                                         control=control_states[block] if control is not None else None,
                                         sae_reads=sae_reads, arms=arms, searched=(searched or {}).get(block))
                yield RetestBatch(number, index, shard.split, rows, block, values)


def collect_ranges(batches: Iterable[RetestBatch]) -> FeatureRanges:
    """The first pass: every feature's minimum and maximum over every residue of both splits."""
    ranges = FeatureRanges()
    for batch in batches:
        for arm, values in batch.arms.items():  # (n, F)
            ranges.update((batch.layer, arm), values)
    return ranges


def collect_counts(batches: Iterable[RetestBatch], shards: Sequence[RetestShard], ranges: FeatureRanges,
                   concepts: ConceptSet, *, device: str | torch.device = "cpu") -> dict[tuple[int, str, str], ConceptCounts]:
    """The second pass: every arm's counts at every layer and split, with features scaled by the first pass's ranges."""
    counts: dict[tuple[int, str, str], ConceptCounts] = {}
    cached: tuple[tuple[int, int], BatchLabels] | None = None
    for batch in batches:
        if cached is None or cached[0] != (batch.shard, batch.batch):
            cached = ((batch.shard, batch.batch), batch_labels(shards[batch.shard].labels[batch.rows], concepts.domain_level, device))
        for arm, values in batch.arms.items():  # (n, F)
            key = (batch.layer, arm, batch.split)
            if key not in counts:
                counts[key] = ConceptCounts(len(concepts.names), values.shape[1], device=device)
            counts[key].update(ranges.scale((batch.layer, arm), values), cached[1])
    return counts


def retest_tables(counts: Mapping[tuple[int, str, str], ConceptCounts], concepts: ConceptSet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per layer and arm (and subset), InterPLM's report: features scored, features associated with a concept and
    their share, concepts identified, and the mean best test F1; beside it each concept's best test F1."""
    summary, per_concept = [], []
    readings = [(layer, arm, arm, None) for layer, arm, split in counts if split == SPLITS[0]]
    for layer, arm, _, _ in list(readings):
        for subset, (source, share) in SUBSETS.items():
            if arm == source:
                width = counts[layer, arm, SPLITS[0]].fired.shape[0]
                readings.append((layer, subset, arm, np.arange(int(width * share))))
    for layer, name, arm, features in readings:
        valid, test = (f1_scores(counts[layer, arm, split], concepts, split) for split in SPLITS)
        report = interplm_report(valid, test, concepts.reported, features=features)
        scored = np.isfinite(report.best_test_f1)  # (C,)
        summary.append({"layer": layer, "arm": name, "features": report.features,
                        "associated_features": report.associated_features,
                        "share": report.associated_features / report.features,
                        "identified_concepts": report.identified_concepts, "concepts_scored": int(scored.sum()),
                        "mean_best_test_f1": float(report.best_test_f1[scored].mean()) if scored.any() else float("nan")})
        per_concept.extend({"layer": layer, "arm": name, "concept": concepts.names[index], "best_feature": int(report.best_feature[index]),
                            "best_test_f1": float(report.best_test_f1[index])} for index in np.flatnonzero(concepts.reported))
    return pd.DataFrame(summary), pd.DataFrame(per_concept)
