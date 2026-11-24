# Reproduction environments

## Numerical replay

Use CPython 3.11.15 and `cpu.txt` for the manuscript table and experimental calculations. `verify/run.py --experiments` writes its outputs to the chosen directory. It uses the stored first-party results and requires no model checkpoint or GPU.

## Figure rendering

The current Matplotlib figures were tested with CPython 3.11.17, the directly observed versions in `figure-modern.txt`, FreeType 2.14.3 and the Arial typeface. The original compact bar and comparison panels use CPython 3.7.0 and `figure-classic.txt`. Font availability and the raster backend can affect glyph metrics. The README and inventory in `figures` identify the rendering route for each figure. These drawing-package lists record the working versions; the fresh numerical replay uses a separate environment.

## Model inference

A local GPU with 32 GB of device memory was available for the molecular subset tests. The test device was an NVIDIA GeForce RTX 5090 with driver 610.88. The protein subset forward passes used CPU and the recorded Transformers/PyTorch configuration. A subset observation checks the specified input, layer, model and operation. Historical complete experiments varied in the number of encoders, molecules, protein groups, random networks, sparse dictionaries and output interventions.

The per-experiment documentation identifies which code produces end-to-end outputs and which operations have been replayed only on a bounded input. Deployment hardware, cache size and precision determine full-run time; historical budgets are given only where recorded. Collect the listed model releases separately from the study code. Large intermediate results are optional for readers who run the stored-result checks.

The molecular rewrite example ran one canonical molecule and five alternative representations in 14.6 seconds on the RTX 5090, including PyTorch import and model loading. Peak memory tracked by the PyTorch allocator was 22.9 MiB allocated and 36 MiB reserved. The driver and other processes consume additional memory. The seven separate protein CPU subset commands took 12.7 to 20.4 seconds each. These measurements apply to the supplied small examples.
