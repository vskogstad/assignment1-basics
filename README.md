# CS336 Spring 2025 Assignment 1: Basics

My implementation of Assignment 1 in CS-336.  
Original tests uses resource module, which does not work on windows. Skipped the import and tests requiring it.   
I've made minor modifications to test files to force utf-8 format when opening files on windows.  

To do a training run using open web text, clone the repo and run main.py. This will download training and validation data from huggingface and execute a training run using the settings in config file "best.yaml".  

The model and training script is now a bit messy as I've tried to make it as efficient as possible without using external libraries or torch.nn.functional. Some older block modules used in ablations are broken as I've not kept them up to date with changes in function signatures. Likewise running older config files will probably not work as I've changed config.py when implementing new features.


## Leaderboard submission

[Final training run](https://api.wandb.ai/links/skogstadv-hobbyist/w94h1ysd)

Validation loss of 3.0305. [Leaderboard](https://github.com/stanford-cs336/assignment1-basics-leaderboard?tab=readme-ov-file#openwebtext-subsample-validation-loss-leaderboard)  

**Initial Architecture**

-Used mixed precision with bfloat16.  
-Initialization of projection and classification layers to zero (Like in NanoGPT-speedruns).  
-Muon with AdamW optimizer. Using same learning rate for both optimizers, and scaling Muon LR following: https://arxiv.org/abs/2502.16982  
-Layer norm scaling. https://arxiv.org/pdf/2502.05795  

Used step_law optimal learning rate/batch size as a starting point. https://arxiv.org/html/2503.04715v6  
Scaled up model size to fit within memory while using optimal batch-size. Adjusted model dimensions to get as high MFU as possible. (Meaning prioritize wide matrix-operations over my slow attention implementation and using just 8 attention heads). Ended up going with a lower MFU architecture in the end as it outperformed larger wider models.  
Incrementally increased lr from "optimal lr" until I felt I had spent too much on GPU rental. Final learning rate is 10x step_law optimal, so I am probably doing something wrong in my calculations, or our data sets are very different.  

**Improvements**

-Changed the data-loader from random, to randomized strided sampling without replacement. More unique training samples gave increased training loss(no repetitions) but brought the validation loss down to 3.126.  
-Implemented gated attention (Sort of like in qwen next, but I do full gates instead of per head and use SILU instead of sigmoid). Changed to just 6 attention heads, which is a bit worse but gives higher MFU. In sum this brought validation loss down to 3.1035. https://arxiv.org/pdf/2505.06708  
-QK-norm -> 3.094  
-Adjusting AdamW params from [0.90, 0.95] to [0.9, 0.999] gave a surprisingly large boost -> 3.0839  
-U-net architecture with learnable params. -> 3.0762  
-NorMuon optimizer: https://arxiv.org/abs/2510.05491 
-Mixing in extra embedding values in later value matrices like in NanoGPT speedrun. General idea: https://arxiv.org/pdf/2410.17897  
-Scaling down d_model from 1536 to 1024 (training for more steps). Still way above chinchilla optimal, but such a model would not get good MFU.
-Postponing full validation til after training for 90 mins (training for more steps). 

**Attempts**

-Lowering the LR for the LM-head layer as recommended in Dion. NanoGPT seems to be doing it the other way around? Gave very minimal improvements. https://arxiv.org/pdf/2504.05295  
-Document masking. Lower MFU and very slight performance decrease. I really expected this to work a lot better.  
-Sliding window attention. Hybrid with 3 layers sliding window 1 full layer is only slightly worse, but no speedup.  
-Grouped query attention. Worse, as expected.  
-Scaling output of each block like in Ernie. Worse. Think this might be interfering with my layernorm scaling.  
-Decreasing/increasing warmup steps or increasing learning rate after adding QK-norm gave no benefit.  
-QK-clip. I was not able to get this to work. In theory it should help a bit with the MFU compared to QK-norm.  
