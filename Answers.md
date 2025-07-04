**Understanding Unicode**
a) The Unicode character 0 is the terminate-string character. It is not a regular character.

b) Its repr is '\\x00' while the str is '\x00'.

c) For me the character does not display anything when used outside a print statement. In a print it shows as '\x00'. Online discussion, show that some terminals might skip the following output.


**Unicode encodings**
a) It takes up less space.

b) It works only for single-byte chars as the function are splitting per byte. It will fail for 'Å' for example.

c) 0xa5 followed by any other byte. It is not a valid start byte, and can only be a valid end-byte in two or more-byte characters.


**BPE training on tinystories**
a) It takes 160 seconds on 4 processes using re.findall(), 230 secs with finditer(). Theoretically 64 GB of ram available, but just using 4 out of 8 cores. I think I might be IO-bound as I don't see improvement going up to 8.
 Almost all of the time post-tokenization is spent in the find_best_pair function iterating over a growing dictionary. ' accomplishment' is the longest word in the tokenizer.

         9446312 function calls (9446230 primitive calls) in 160.372 seconds

   Ordered by: internal time
   List reduced from 359 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       19  127.502    6.711  127.502    6.711 {method 'acquire' of '_thread.lock' objects}
     9742   27.296    0.003   27.314    0.003 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:52(find_best_pair)
     9742    3.368    0.000    4.944    0.001 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:73(update_dictionaries)
  1640484    0.367    0.000    0.367    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:127(<genexpr>)
        1    0.342    0.342    0.462    0.462 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:39(find_initial_merge_candidates)
  1451534    0.305    0.000    0.305    0.000 {method 'add' of 'set' objects}
  1073742    0.246    0.000    0.246    0.000 {method 'remove' of 'set' objects}
  1708452    0.225    0.000    0.225    0.000 {method 'get' of 'dict' objects}
  1359551    0.210    0.000    0.210    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:128(<genexpr>)
   287517    0.136    0.000    0.136    0.000 /home/vegard/snap/code/196/.local/share/uv/python/cpython-3.11.12-linux-x86_64-gnu/lib/python3.11/collections/__init__.py:728(__delitem__)


