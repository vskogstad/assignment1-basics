Thanks for making this course openly available! 

Architecture:
Used mixed precision with bfloat16.
Initialization of projection and classification layers to zero (Like in NanoGPT).
Muon with AdamW optimizer. Using same learning rate for both optimizers, and scaling Muon LR following: https://arxiv.org/abs/2502.16982
Layer norm scaling. https://arxiv.org/pdf/2502.05795

Training:
Used step_law optimal learning rate/batch size as a starting point. https://arxiv.org/html/2503.04715v6
Scaled up model size to fit within memory while using optimal batch-size. Adjusted model dimensions to get as high MFU as possible. (Meaning prioritize wide matrix-operations over my slow attention implementation and using just 8 attention heads)
Gradually increased lr from "optimal lr" until it became unstable. 

I calculate validation-loss based on the entire validation set. To save flops, I just do it once at the end of the run. 

https://api.wandb.ai/links/skogstadv-hobbyist/4ux95ftu