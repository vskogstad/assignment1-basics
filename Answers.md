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

         9642509 function calls (9642329 primitive calls) in 211.969 seconds

   Ordered by: internal time
   List reduced from 508 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       19  191.267   10.067  191.267   10.067 {method 'acquire' of '_thread.lock' objects}
     9742   15.086    0.002   15.230    0.002 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:62(find_best_pair)
     9742    3.385    0.000    4.858    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:134(update_dictionaries)
  1640484    0.351    0.000    0.351    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:195(<genexpr>)
        1    0.333    0.333    0.449    0.449 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:48(find_initial_merge_candidates)
  1578863    0.314    0.000    0.314    0.000 {method 'add' of 'set' objects}
  1143475    0.248    0.000    0.248    0.000 {method 'remove' of 'set' objects}
  1359551    0.211    0.000    0.211    0.000 /home/vegard/projects/stanford/assignment1-basics/cs336_basics/train_bpe.py:196(<genexpr>)
  1564246    0.192    0.000    0.192    0.000 {method 'get' of 'dict' objects}
   287517    0.134    0.000    0.134    0.000 /home/vegard/snap/code/196/.local/share/uv/python/cpython-3.11.12-linux-x86_64-gnu/lib/python3.11/collections/__init__.py:728(__delitem__)


