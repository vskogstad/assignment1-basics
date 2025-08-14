What I've changed from the base model.

Used step_law optimal learning rate/batch size, then grid search on lr around "optimal lr". https://arxiv.org/html/2503.04715v6
Scaled up model size to fit optimal batch-size:
Mixed precision using bfloat16.
Grouped query attention.
Initialization of projection and classification layers to zero (Like in NanoGPT)
Muon with AdamW optimizer. Using same learning rate for both optimizers, and scale Muon following: https://arxiv.org/abs/2502.16982
Layer norm scaling. https://arxiv.org/pdf/2502.05795

I still have poor MFU. My implementation of attention, softmax and cross-entropy is a lot slower than pytorch defaults. Could improve the loss a bit with hand-written kernels.