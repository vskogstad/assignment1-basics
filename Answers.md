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
      
         1735332825 function calls (1735332645 primitive calls) in 11018.474 seconds

   Ordered by: internal time
   List reduced from 509 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    31742 9421.821    0.297 9425.142    0.297 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:62(find_best_pair)
       19  811.573   42.714  811.573   42.714 {method 'acquire' of '_thread.lock' objects}
    31742  447.535    0.014  724.225    0.023 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:134(update_dictionaries)
304839051   92.894    0.000   92.894    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:195(<genexpr>)
288643714   63.660    0.000   63.660    0.000 {method 'add' of 'set' objects}
227115374   45.823    0.000   45.823    0.000 {method 'remove' of 'set' objects}
267020936   36.078    0.000   36.078    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:196(<genexpr>)
        1   32.440   32.440   48.709   48.709 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:48(find_initial_merge_candidates)
294497648   28.426    0.000   28.426    0.000 {method 'get' of 'dict' objects}
267066646   14.648    0.000   14.648    0.000 {method 'append' of 'list' objects}


b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'

b) The tokenizers reflect the information contained in the datasets. As owt is a more general dataset you have more specialized words from many fields. tinystories is geared only towards children stories.



**Tokenizer_experiments**
a) Compression ratio on TinyStories/OpenWebText is: / . 

b) Compression on OpenWebText with Tinystories tokenizer reduces the compression to . The vocabulary is not adapted to the source material.

c) Throughput in bytes/s = . Estimated time spent = 825 * 1024 * 1024 * 1024 (bytes) / Throughput  (bytes/s) =  (s) or roughly 

d) uint16 can store positive values up to 65535, which fits well with the vocabulary sizes we've been targetting. If we wanted to go up to say, 100 000 merges we would have to select a different data type.

**Resource accounting model**



**Tuning the learning rate**
Results after 10 iterations:
Loss with lr 1e1 = 3.07
Loss with lr 1e2 = 4.16e-23
Loss with lr 1e3 = 2.06e+19
A learning rate of 1e1 is already quite agressive and gives rapid convergence towards 0. If we increase lr by a factor of 10 to 1e2 loss will decrease faster and if we increase lr by a factor of 100 the loss diverges.

**Resource accounting AdamW**



**Final optimizations**
Shared embedding matrix in/out
Muon optimizer