# Scaling exposure readouts and network references

The exposure selection rule is applied independently at ESM-2 650M block 16; block 17 is a descriptive sensitivity analysis. Development selection and readout definitions are frozen before confirmation. The primary block-16 neuron is 4566. Continuous MAE and paired Brier use the original sequence references, fixed protein populations, group weighting and exposure labels.

Every untrained network selects its own development direction. There are 199 initialization references and nine orthogonal rotations. A descriptive inclusive upper-tail rank is (1 + number of references at least as high)/(K+1). The two 650M endpoints form their own multiplicity family. Block 16 improves the paired Brier endpoint but not continuous MAE, leaving the prespecified two-endpoint criterion unmet. Adjacent-layer results, untrained ranges, all rotations and the absolute losses are retained.

Haar matrices follow Gaussian QR with positive-diagonal sign correction and float64 projections. The initialization seed and exact model configuration are recorded in the scientific freeze metadata. Raw model activations remain float32; TensorFloat32 and dropout are disabled. CPU result verification requires no model weights.

The original source byte digests, file sizes, preserved freeze timestamps and the corresponding English protocol are listed in frozen_records.json. Date stamps identify freezes; they do not denote separate reproduction outputs.
