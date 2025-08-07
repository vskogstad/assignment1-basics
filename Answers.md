**Understanding Unicode**

a) The Unicode character 0 is the terminate-string character. It is not a regular character.

b) Its repr is '\\x00' while the str is '\x00'.

c) For me the character does not display anything when used outside a print statement. In a print it shows as '\x00'. Online discussion, show that some terminals might skip the following output.


**Unicode encodings**

a) It takes up less space.

b) It works only for single-byte chars as the function are splitting per byte. It will fail for 'Å' for example.

c) 0xa5 followed by any other byte. It is not a valid start byte, and can only be a valid end-byte in two or more-byte characters.


**BPE training on tinystories**

a) It takes 160 seconds total on 4 processes using re.findall(), 230 secs with finditer(). Theoretically 64 GB of ram available, but just using 4 out of 8 cores. I think I might be IO-bound as I don't see improvement going up to 8.
 Almost all of the time post-tokenization is spent in the find_best_pair function iterating over a growing dictionary. After improving the find_best_pair algorithm, this time is reduced from 27.3 to 15.1 secs.


b) Pre-tokenization right now. With better parallelization, and more merges, I might also be limited by the find_best_pair algorithm.

         9800945 function calls (9800765 primitive calls) in 175.798 seconds

   Ordered by: internal time
   List reduced from 509 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       19  160.116    8.427  160.116    8.427 {method 'acquire' of '_thread.lock' objects}
     9742    9.489    0.001    9.634    0.001 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:62(find_best_pair)
     9742    3.805    0.000    5.393    0.001 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:134(update_dictionaries)
  1640484    0.387    0.000    0.387    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:195(<genexpr>)
        1    0.362    0.362    0.476    0.476 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:48(find_initial_merge_candidates)
  1582603    0.319    0.000    0.319    0.000 {method 'add' of 'set' objects}
  1147207    0.253    0.000    0.253    0.000 {method 'remove' of 'set' objects}
  1359551    0.238    0.000    0.238    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:196(<genexpr>)
  1661656    0.217    0.000    0.217    0.000 {method 'get' of 'dict' objects}
   287517    0.146    0.000    0.146    0.000 /home/vegard/snap/code/196/.local/share/uv/python/cpython-3.11.12-linux-x86_64-gnu/lib/python3.11/collections/__init__.py:728(__delitem__)


b' accomplishment'


**BPE on open webtext**

Heuristic sizing of good pairs with hyperparameter = 10, 8 processes and multithreaded encoding. 3.06 hours. Switched to doing encoding out of multithreaded part and saw speedups on other datasests afterwards. Could likely see further improvements here. 
      
         1741934743 function calls (1741934563 primitive calls) in 10914.034 seconds

   Ordered by: internal time
   List reduced from 510 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    31742 9628.374    0.303 9631.985    0.303 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:63(find_best_pair)
    31742  476.556    0.015  768.708    0.024 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:134(update_dictionaries)
       19  448.042   23.581  448.042   23.581 {method 'acquire' of '_thread.lock' objects}
304839051   91.334    0.000   91.334    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:197(<genexpr>)
288643714   66.422    0.000   66.422    0.000 {method 'add' of 'set' objects}
227115374   48.375    0.000   48.375    0.000 {method 'remove' of 'set' objects}
267020936   44.095    0.000   44.095    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:198(<genexpr>)
        1   35.459   35.459   52.107   52.107 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:49(find_initial_merge_candidates)
294497648   30.927    0.000   30.927    0.000 {method 'get' of 'dict' objects}
267066646   15.760    0.000   15.760    0.000 {method 'append' of 'list' objects}


b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'

b) The tokenizers reflect the information contained in the datasets. As owt is a more general dataset you have more specialized words from many fields. tinystories is geared only towards children stories.



**Tokenizer_experiments**

a) Compression ratio on TinyStories/OpenWebText is: 4.043/4.510. 

b) Compression on OpenWebText with Tinystories tokenizer reduces the compression from 4.043 to 3.375. The vocabulary of the tokenizer is not adapted to the source material.

c) Throughput in MB/s = 2.66. Estimated time spent = 825 * 1024 (MB) / 2.66  (MB/s) =  317594(s) or 88 hours ~= 4 days.

d) uint16 can store positive values up to 65535, which fits well with the vocabulary sizes we've been targetting. If we wanted to go up to say, 100 000 merges we would have to select a different data type.

**Resource accounting model**

a) Our model would have 1,557 billion parameters. Memory to load just the model is 5.80 GB

b)  '''
 # FLOPS (skipped rmsnorm)
    pos_flops = 1 * sequence_length * d_model
    ffn_flops = num_layers * (4 * sequence_length * d_ff * d_model)  # x @ W_up + W_down @ x
    mha_flops = num_layers * (
        (3 * 2 * sequence_length * d_model * d_model)  # Q K and V
        + (2 * sequence_length * d_model * sequence_length)  # Q @ K^T
        + (2 * sequence_length * sequence_length * d_model)  # attn @ V
        + (2 * sequence_length * d_model * d_model)  # out @ W_O
    )
    lmhead_flops = 2 * (sequence_length * vocab_size * d_model) # x @ W_lm_head

FFN:        39.8% (S) / 57.4% (XL)
Attention:  33.1% (S) / 37.9% (XL)
LM Head:    27.1% (S) /  4.7% (XL)
Positional: ~0% (negligible)

To do one forward pass requires 3.51 TFLOPs.

c) Linear layers make up the majority, for small models the lm_head becomes a huge fraction. As we scale the sequence length, mha will dominate.

d) 
FFN:        49.8% (M) / 54.4% (L)
Attention:  37.4% (M) / 38.1% (L)
LM Head:    12.7% (M) /  7.4% (L)
Positional: ~0% (negligible)
For small models the lm_head is a huge fraction. As we scale up the model it becomes less and less important. If we scale up the sequence length more than model dimensions, Attention will grow larger than ffn.

e) To do one forward pass on XL with extended context requires 133.42 TFLOPS. We see that part of attention is growing quadratically with sequence length.
FFN:        24.1% 
Attention:  73.9% 
LM Head:    2.0% 
Positional: ~0% (negligible)

**Tuning the learning rate**

Results after 10 iterations:
Loss with lr 1e1 = 3.07
Loss with lr 1e2 = 4.16e-23
Loss with lr 1e3 = 2.06e+19
A learning rate of 1e1 is already quite agressive and gives rapid convergence towards 0. If we increase lr by a factor of 10 to 1e2 loss will decrease faster and if we increase lr by a factor of 100 the loss diverges.

**Resource accounting AdamW**
a) memory = (num_activations + total_parameters + num_gradients + optimizer_states) * element_size 
Where:
    num_gradients = total_parameters
    optimizer_states = total_parameters * 2  # momentum + variance

    # Activations
    embedding_activations = sequence_length * d_model
    # ffn

    linear_activations = (d_model + d_ff) * sequence_length
    # mha

    KQV_activations = sequence_length * d_model * 3
    attention_matrix = sequence_length * sequence_length
    attention_output = sequence_length * d_model
    attention_activations = attention_output + attention_matrix + KQV_activations
    # total activations
    num_activations = (embedding_activations + num_layers * (linear_activations + attention_activations)) * batch_size

b) Memory in GB = 10,46 * batch_size + 23,2 

c)

d) This seems like a trick question, as you typically would not use the fp32 spec (19.5 TFLOPS) but TF32 spec (156 TFLOPS) to calculate the time needed (most operations are matmuls). Using 19.4 TFLOPS I get: 
Total flops A100, 400 steps, w. 1024 batch_size = 4309039353.69 TFLOPs
Total time = 5115.19 days. 
This shows the advantage of matmul optimizations done in the more recent gpus such as A100, as it decreases the required training time by 1/8. Still a long time, but this is 0.4T tokens. It is not that far of from SOTA training runs of 15T tokens like Kimi K2. Even with a smaller model this takes a lot of time using no mixed training and a single GPU.

**Learning rate**
I linearily increased the learning rate from 0.0001 to 0.01. It breaks down at iteration 400 with learning rate 0.0045. This is a likely upper bound for a full training run. I then did 3 training runs with 1000 steps of cosine scheduler using max_learning rate of [0.003, 0.004, 0.005] and min_learning rate 10% of max. It failed for all three.
Found a bug in my softmax function that made training unstable. I then reran experiments, increasing the LR linearly over 1000 steps to 0.01. I did not have any divergence as LR increased during this run, and ended up with a final loss of 1.88, which is surprisingly good. I got worse results when running for the same amount of steps and trying to use 0.01 as the peak learning rate with 50 or 200 steps of warmup. The model does not converge to similar losses as my linear "LR search". It seems like a long slow warmup is beneficial for the model.

I then tried to follow https://arxiv.org/html/2503.04715v1. They defined their step law as LR_optim = 1.79 * N**(-0.713) * D **(0.307). They reccomend really large batch sizes (and they train for a lot of tokens to find optimal point in the loss landscape). With my limited amount of tokens, I don't think we can go all the way up to the perfect batch size as the number of steps will be to small. 
As the LR is not really connected to the batch size, I just use it directly giving 0.005 as max LR. min LR of 0.00001. Doing this for a full training loss yields a loss of 1.33.

After this I tried running the Muon optimizer with 0 tuning of LR, yielding worse performance and a final loss of 1.7. Looking at https://arxiv.org/abs/2502.16982 I used their modified optimization formula for converting tuned adam learning rates. I found a bug in my implementation which made the previous result irrellevant. Running the model using Muon yielded a final loss of 1.35 with half the training data of the adamw-run (No actual tuning for muon).


**Batch-size experiment**
Reccomended batch sizes in literature are high, both for system reasons and for extracting generalized learning across the batch. Given our very limited amount of tokens, where our model is nowhere near overtrained, a theoretically optimal batch size, will give very few steps (< 26 steps for our case). Taking more steps in roughly the right direction is going to give a lower final loss than fewer steps in exactly the right direction. (Although the later would probably yield a better model with continued training).

I started tuning with as high a batch size as I could fit (512) and decreased to 256, 128, 64 and 1. It reached the lowest loss for for 128 though the MFU was better with high


**Generated text**

The output is now mostly a sequence of words with correct spacing. Usage of punctation is there, but not always used correctly.




The model is now keeping track of the same person staying in the story across sentences. Strength of the model is the most important parameter, but temperature and top_p when sampling also affects the results.


**Layer norm ablation**

Crashes almost directly after warmup. Loss goes down then explodes throughout the warmup phase. Get NaNs in the end. As expected. 


**Post norm vs pre-norm**

Trains ok with post-norm but gets slightly worse, loss and MFU.

**NoPE vs RoPE**

Much like post norm, the NoPE-version trains fine, but has slightly worse loss than RoPE.


**SILU vs SwiGLU**

Really poor final loss using SiLU. Interestingly, using SiLU instead of SwiGLU gives a lot better MFU. The loss is so much worse that I think it should be optimizeable, have not seen this kind of discrepancy in papers. If so, it could be a good option for time constrained challenges like the final leaderboard. (Alternatively I could try to improve the speed of the SwiGLU function instead)

**Experiment on OWT**



This model has a larger vocabulary(less examples per token) and much more complex training data, requiring significantly more time and greater context length to learn from. As the model gets more noisy data it will take longer for it to learn the basics. 

**Final optimizations**

Shared embedding matrix in/out
Muon optimizer