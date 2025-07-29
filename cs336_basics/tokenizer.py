import json
import numpy as np
import pickle
import time
from collections.abc import Iterable, Iterator
from collections import deque
from itertools import chain

import regex as re

# TODO: Pass final test(memory)
# TODO: Improve efficiency of tokenization (Lots of potential here)


class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Not optimized at all. Trying to get it to work first.
        """
        self.vocab = vocab
        self.merges = merges
        #print(f" This is before adding anything {vocab[50256]=}")
        # find special tokens in vocab as well as additional
        self.special_tokens = [tok for id, tok in vocab.items() if str(tok).startswith("b'<|")]
        #print("already there:", self.special_tokens)
        new_special_tokens = []
        if special_tokens:
            new_special_tokens = [tok.encode("utf-8") for tok in special_tokens if tok.encode("utf-8") not in self.special_tokens]
            #print(f"{new_special_tokens=}",)
        self.special_tokens += new_special_tokens 
        self.vocab.update({id:vocab for id, vocab in enumerate(new_special_tokens, start = len(self.vocab))})
        self.vocab_to_int = {value:key for key, value in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Only works with .pkl files for now. Serialization of bytes with json is not totally straight forward, might need to import base64.
        """
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        
    def encode(self, text: str) -> list[int]:
        
        escaped = [re.escape(token.decode()) for token in self.special_tokens]
        # regex finishes at first match. if we sort by length desc, we will get substrings of longer strings at the end. 
        # matching <|endoftext|><|endoftext|> before <|endoftext|>
        escaped.sort(key=len, reverse=True)
        SPECIAL = r"|".join(escaped)
        #text = text.encode("utf-8")
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        segments = []
        try:
            specials = [self.vocab_to_int[spec.encode()] for spec in re.findall(SPECIAL, text)]
            splitted = re.split(SPECIAL, text)
        except KeyError:
            specials = []
            splitted = text
            #import sys;sys.exit()
        indeces = []
        # UGLY! 
        for i, chunk in enumerate(splitted):
                segments = []
                for segment in PAT.findall(chunk): 
                    segments.append(segment.encode())
                
                for segment in segments:
                    # Convert to token IDs
                    segment = [self.vocab_to_int[bytes([byte_val])] for byte_val in segment]
                    
                    # OPTIMIZED MERGE PROCESS
                    while len(segment) >= 2:
                        # Find all pairs in current segment and their positions
                        pairs_in_segment = {}
                        for idx in range(len(segment) - 1):
                            pair = (segment[idx], segment[idx + 1])
                            if pair not in pairs_in_segment:
                                pairs_in_segment[pair] = []
                            pairs_in_segment[pair].append(idx)
                        
                        # Find the highest priority merge that exists in this segment
                        best_merge_idx = None
                        best_pair = None
                        
                        for merge_idx, merge_bytes in enumerate(self.merges):
                            a, b = merge_bytes
                            pair_ids = (self.vocab_to_int[a], self.vocab_to_int[b])
                            if pair_ids in pairs_in_segment:
                                best_merge_idx = merge_idx
                                best_pair = pair_ids
                                break  # Found highest priority merge
                        
                        if best_pair is None:
                            break  # No more merges possible
                        
                        # Apply the best merge
                        new_token = self.merges[best_merge_idx][0] + self.merges[best_merge_idx][1]
                        new_token_id = self.vocab_to_int[new_token]
                        
                        # Merge all instances of this pair (from right to left to avoid index shifting)
                        positions = pairs_in_segment[best_pair]
                        for pos in reversed(positions):
                            if pos < len(segment) - 1 and segment[pos] == best_pair[0] and segment[pos + 1] == best_pair[1]:
                                segment[pos:pos + 2] = [new_token_id]
                    
                    indeces.append(segment)
                
                if specials and i < len(splitted) - 1:
                    indeces.append([specials[i]])
            
        return list(chain.from_iterable(indeces))
    
    def encode2(self, text: str) -> list[int]:
        
        escaped = [re.escape(token.decode()) for token in self.special_tokens]
        # regex finishes at first match. if we sort by length desc, we will get substrings of longer strings at the end. 
        # matching <|endoftext|><|endoftext|> before <|endoftext|>
        escaped.sort(key=len, reverse=True)
        SPECIAL = r"|".join(escaped)
        #text = text.encode("utf-8")
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        segments = []
        try:
            specials = [self.vocab_to_int[spec.encode()] for spec in re.findall(SPECIAL, text)]
            splitted = re.split(SPECIAL, text)
        except KeyError:
            specials = []
            splitted = text
            #import sys;sys.exit()
        indeces = []
        # UGLY! 
        for i, chunk in enumerate(splitted):
            segments = []
            for segment in PAT.findall(chunk): 
                segments.append(segment.encode())
            for segment in segments:
                # Need this line to work with GPT2-vocab. Can't assume sorted in ascii order. 
                segment = [self.vocab_to_int[bytes([byte_val])] for byte_val in segment]
                for best_pair in self.merges:
                    if len(segment) < 2: # No more meges possible
                        break
                    a, b = best_pair
                    new_token = a + b

                    new_token_id = self.vocab_to_int[new_token]
                    best_pair = self.vocab_to_int[a], self.vocab_to_int[b]
                    skip = False
                    word_tokenization = []

                    for c1, c2 in zip(segment, segment[1:]):

                        if skip:
                            skip = False
                            continue

                        if best_pair == (c1, c2):
                            skip = True
                            #print(f"merging {c1} and {c2} into {new_token_id}")
                            word_tokenization.append(new_token_id)

                        else:
                            word_tokenization.append(c1)
                    else:
                        if not skip:
                            word_tokenization.append(c2)
                    segment = word_tokenization
                indeces.append(segment)
            if specials and i < len(splitted) - 1:
                indeces.append([specials[i]])
        
        return list(chain.from_iterable(indeces))
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        We are receiving a file object "iterable". We read a chunk from the file and tokenize it. 
        The EncodingIterator will yield the resulting tokens one by one.
        """
        class EncodingIterator:
            def __init__(self, tokenizer, iterable):
                self.tokens = deque()
                self.iterable = iterable
                self.tokenizer = tokenizer
                self.chunk_size = 1024*1024
                self.buffer = ""

            def __iter__(self):
                return self
            
            def __next__(self):
                if not self.tokens:
                    chunk = self.buffer + self.iterable.read(self.chunk_size)
                    # find newline character to split on.
                    split_point = chunk.rfind("\n")  # searching from right to left in the chunk
                    if split_point == -1:
                        print("Did not find a proper split point, splitting at chunk end")
                        split_point = len(chunk)
                    self.buffer = chunk[split_point:]
                    chunk = chunk[:split_point]

                    if not chunk:
                        if self.buffer:
                            chunk = self.buffer 
                            self.buffer = ""
                        else:
                            raise StopIteration
                    self.tokens = deque(self.tokenizer.encode(chunk)) # Reverse even faster?

                return self.tokens.popleft()
        return EncodingIterator(self, iterable)



    def decode(self, ids: list[int]) -> str:
        
        #decoded = sum([self.vocab[id] for id in ids])
        byte_string = b""
        for id in ids:
            byte_string += self.vocab[id]
        return str(byte_string, "utf-8", errors="replace")
    

    @staticmethod
    def throughput(filename: str, tokenizer) -> None:
        with open(filename, encoding="utf-8") as f:
            text = f.read()
            num_bytes = len(bytes(text, encoding="utf-8"))
            f.seek(0)
            tokenizer.encode("warmup")
            t0 = time.time()
            indices = []
            for _id in tokenizer.encode_iterable(f):
                indices.append(_id)
                #indices = tokenizer.encode_iterable(f)
            t1 = time.time()
            throughput = num_bytes / (t1 - t0) 

        compression_ratio = Tokenizer.compression_ratio(text, indices)
        print(f"For the {filename} dataset we get {compression_ratio=}")
        print(f" and {throughput=} bytes/s, {throughput/1024**2:.2f} MB/s")

        
        return throughput

    @staticmethod
    def compression_ratio(string: str, indices: list[int]) -> float:
        bytes_string = len(bytes(string, encoding="utf-8"))
        bytes_indices = len(indices)
        compression_ratio = bytes_string / bytes_indices
        return compression_ratio


def file_to_numpy(output: str, text_file: str, tokenizer):
    """Tokenizes a text file into a one-dimensional numpy vector for training"""
    ids = []
    with open(file=text_file, encoding='utf-8') as f:
        for _id in tokenizer.encode_iterable(f):
            ids.append(_id)
    np.save(output, arr=ids) #, mmap_mode="w")


if __name__ == "__main__":

    special_tokens = ["<|imstart|>","<|endoftext|>"]
    with open("cs336_basics/vocab-tiny.pkl", "rb") as vocab_file:
        vocab = pickle.load(vocab_file)
    with open("cs336_basics/merges-tiny.pkl", "rb") as merges_file:
        merges = pickle.load(merges_file)
    # with open("/home/vegard/projects/stanford/assignment1-basics/tests/fixtures/gpt2_vocab.json", "rb") as vocab_file:
    #     vocab = json.load(vocab_file)
    # with open("/home/vegard/projects/stanford/assignment1-basics/tests/fixtures/train-bpe-reference-merges.txt", "rb") as merges_file:
    #     merges = merges_file.read
    #with open("cs336_basics/tokenizer_tinystories.json","r") as f:
    #        vocab, merges = json.load(fp=f ) #json.dump(model, f, default=repr, indent=4)
    tokenizer = Tokenizer.from_files(vocab_filepath="cs336_basics/vocab-tiny.pkl", merges_filepath="cs336_basics/merges-tiny.pkl", special_tokens=special_tokens)
    print(type(tokenizer.vocab), type(tokenizer.merges))
    enc = tokenizer.encode("Lets test how lucky we can get")
    print(enc)
    dec = tokenizer.decode(enc)
    print(dec)
    
    ## 
    file_to_numpy(output=r"cs336_basics/test_array2", text_file=r"data/TinyStories-valid.txt", tokenizer=tokenizer)
    tokenizer.throughput(filename="data/TinyStories-mini.txt", tokenizer=tokenizer) 
    import sys; sys.exit()

    # Test throughput
    import cProfile
    import pstats
    with cProfile.Profile() as profile:
        tokenizer.throughput(filename="data/sample_owt.txt", tokenizer=tokenizer)
        tokenizer.throughput(filename="data/sample_tiny.txt", tokenizer=tokenizer) 
        
        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)
    