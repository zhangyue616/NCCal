# Calibrating sparse dictionaries

Sparse dictionaries are fitted on unlabelled training molecules, selected by unlabelled validation reconstruction, and scored on the same 173 concepts as native units. The primary recipe has twice the native width and TopK sparsity 16. Each of 199 initialized networks receives an independently fitted dictionary. Dictionary readability T1 is one-sided; the dictionary-versus-neuron advantage T2 is two-sided. Four primary tests across the two checkpoints form a Holm family. Other recipes are descriptive sensitivity analyses. Bootstrap intervals condition on the fitted dictionaries and omit retraining uncertainty.

Native and dictionary concept profiles, including unsuccessful recipes, inactive features, validation error and seeds, are preserved. This calibration uses the final-block pre-GELU site. It must not be conflated with the newly refitted block-two post-GELU dictionaries used for intervention in the basis comparison. Cost-stop recovery and deterministic zero-step checks are implementation deviations; they do not create a new scientific comparison.

