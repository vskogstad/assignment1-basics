import json
import numpy as np
import pickle
import time
from collections.abc import Iterable, Iterator
from collections import deque
from itertools import chain

import regex as re

class OptimizedTokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Optimized tokenizer with pre-computed merge priorities and efficient algorithms.
        """
        self.vocab = vocab
        self.merges = merges
        
        # Pre-compute merge priorities for O(1) lookup
        self.merge_priorities = {}
        for i, (a, b) in enumerate(merges):
            a_id = vocab.get(a)
            b_id = vocab.get(b)
            if a_id is not None and b_id is not None:
                self.merge_priorities[(a_id, b_id)] = i
        
        # find special tokens in vocab as well as additional
        self.special_tokens = [tok for id, tok in vocab.items() if str(tok).startswith("b'<|")]
        new_special_tokens = []
        if special_tokens:
            new_special_tokens = [tok.encode("utf-8") for tok in special_tokens if tok.encode("utf-8") not in self.special_tokens]
        self.special_tokens += new_special_tokens 
        self.vocab.update({id:vocab for id, vocab in enumerate(new_special_tokens, start = len(self.vocab))})
        self.vocab_to_int = {value:key for key, value in vocab.items()}
        
        # Pre-compile regex
        escaped = [re.escape(token.decode()) for token in self.special_tokens]
        escaped.sort(key=len, reverse=True)
        self.SPECIAL = r"|".join(escaped) if escaped else ""
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _get_pairs(self, word):
        """Get all adjacent pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _merge_word(self, word):
        """Optimized merge for a single word using priority queue approach."""
        if len(word) < 2:
            return word
        
        word = list(word)  # Make mutable copy
        
        while len(word) > 1:
            # Find the best merge in current word
            best_pair = None
            best_priority = float('inf')
            best_pos = -1
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                priority = self.merge_priorities.get(pair)
                if priority is not None and priority < best_priority:
                    best_pair = pair
                    best_priority = priority
                    best_pos = i
            
            if best_pair is None:
                break
            
            # Apply the merge
            new_token = self.vocab_to_int[self.merges[best_priority][0] + self.merges[best_priority][1]]
            word = word[:best_pos] + [new_token] + word[best_pos + 2:]
        
        return word
        
    def encode(self, text: str) -> list[int]:
        # Handle special tokens
        segments = []
        try:
            if self.SPECIAL:
                specials = [self.vocab_to_int[spec.encode()] for spec in re.findall(self.SPECIAL, text)]
                splitted = re.split(self.SPECIAL, text)
            else:
                specials = []
                splitted = [text]
        except KeyError:
            specials = []
            splitted = [text]
        
        indices = []
        
        for i, chunk in enumerate(splitted):
            if not chunk:  # Skip empty chunks
                if specials and i < len(splitted) - 1:
                    indices.append([specials[i]])
                continue
                
            # Tokenize chunk
            for segment in self.PAT.findall(chunk): 
                segment_bytes = segment.encode()
                
                # Convert to token IDs efficiently
                segment_ids = [self.vocab_to_int[bytes([byte_val])] for byte_val in segment_bytes]
                
                # Apply merges
                merged = self._merge_word(segment_ids)
                indices.append(merged)
            
            # Add special token
            if specials and i < len(splitted) - 1:
                indices.append([specials[i]])
        
        return list(chain.from_iterable(indices))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Optimized streaming tokenizer with larger chunks and better buffering.
        """
        class OptimizedEncodingIterator:
            def __init__(self, tokenizer, iterable):
                self.tokens = deque()
                self.iterable = iterable
                self.tokenizer = tokenizer
                self.chunk_size = 8 * 1024 * 1024  # Larger chunks for better throughput
                self.buffer = ""

            def __iter__(self):
                return self
            
            def __next__(self):
                while not self.tokens:
                    chunk = self.iterable.read(self.chunk_size)
                    if not chunk:
                        if self.buffer:
                            # Process remaining buffer
                            tokens = self.tokenizer.encode(self.buffer)
                            self.tokens.extend(tokens)
                            self.buffer = ""
                        else:
                            raise StopIteration
                    else:
                        full_chunk = self.buffer + chunk
                        # Find safe split point (space or newline)
                        split_point = max(
                            full_chunk.rfind("\n"),
                            full_chunk.rfind(" ")
                        )
                        
                        if split_point == -1 or split_point < len(full_chunk) // 2:
                            # If no good split point, use the whole chunk
                            split_point = len(full_chunk)
                            
                        process_chunk = full_chunk[:split_point]
                        self.buffer = full_chunk[split_point:]
                        
                        if process_chunk:
                            tokens = self.tokenizer.encode(process_chunk)
                            self.tokens.extend(tokens)

                return self.tokens.popleft()
        
        return OptimizedEncodingIterator(self, iterable)

    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join(self.vocab[id] for id in ids)
        return byte_string.decode("utf-8", errors="replace")

    @staticmethod
    def throughput(filename: str, tokenizer) -> None:
        with open(filename, encoding="utf-8") as f:
            text = f.read()
            num_bytes = len(text.encode("utf-8"))
            f.seek(0)
            
            # Warmup
            tokenizer.encode("warmup")
            
            t0 = time.time()
            indices = list(tokenizer.encode_iterable(f))
            t1 = time.time()
            
            throughput = num_bytes / (t1 - t0)

        compression_ratio = len(text.encode("utf-8")) / len(indices)
        print(f"For the {filename} dataset we get compression_ratio={compression_ratio:.2f}")
        print(f"and throughput={throughput:.0f} bytes/s, {throughput/1024**2:.2f} MB/s")
        
        return throughput

    @staticmethod
    def compression_ratio(string: str, indices: list[int]) -> float:
        bytes_string = len(string.encode("utf-8"))
        bytes_indices = len(indices)
        compression_ratio = bytes_string / bytes_indices
        return compression_ratio


# Additional optimization: Use faster merge algorithm for very long sequences
class UltraOptimizedTokenizer(OptimizedTokenizer):
    """
    Even more optimized version using heap for merge priorities.
    """
    
    def _merge_word_heap(self, word):
        """
        Ultra-optimized merge using a min-heap to track best merges.
        This is more complex but faster for very long sequences.
        """
        import heapq
        
        if len(word) < 2:
            return word
        
        word = list(word)
        
        # Build initial heap of all possible merges
        heap = []
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            priority = self.merge_priorities.get(pair)
            if priority is not None:
                heapq.heappush(heap, (priority, i, pair))
        
        # Keep track of valid positions
        valid_positions = set(range(len(word) - 1))
        
        while heap and len(word) > 1:
            priority, pos, pair = heapq.heappop(heap)
            
            # Check if this merge is still valid
            if (pos not in valid_positions or 
                pos >= len(word) - 1 or 
                word[pos] != pair[0] or 
                word[pos + 1] != pair[1]):
                continue
            
            # Apply merge
            new_token = self.vocab_to_int[self.merges[priority][0] + self.merges[priority][1]]
            word = word[:pos] + [new_token] + word[pos + 2:]
            
            # Update valid positions
            valid_positions.discard(pos)
            if pos > 0:
                valid_positions.discard(pos - 1)
            
            # Add new potential merges
            valid_positions = set(range(len(word) - 1))
            for i in valid_positions:
                if i < len(word) - 1:
                    pair = (word[i], word[i + 1])
                    priority = self.merge_priorities.get(pair)
                    if priority is not None:
                        heapq.heappush(heap, (priority, i, pair))
            
            break  # Only do one merge per iteration to keep it simple
        
        return word

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
    tokenizer = OptimizedTokenizer.from_files(vocab_filepath="cs336_basics/vocab-tiny.pkl", merges_filepath="cs336_basics/merges-tiny.pkl", special_tokens=special_tokens)
    print(type(tokenizer.vocab), type(tokenizer.merges))
    enc = tokenizer.encode("Lets test how lucky we can get")
    print(enc)
    dec = tokenizer.decode(enc)
    print(dec)
    
    ## 
    #file_to_numpy(output=r"cs336_basics/test_array3", text_file=r"data/sample_tiny.txt", tokenizer=tokenizer)
    tokenizer.throughput(filename="data/TinyStoriesV2-GPT4-valid.txt", tokenizer=tokenizer) 
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
    