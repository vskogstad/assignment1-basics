import json
import pickle
from collections.abc import Iterable, Iterator
from itertools import chain
import regex as re


class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
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
        with open("cs336_basics/vocab-tiny.pkl", "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open("cs336_basics/merges-tiny.pkl", "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        

    def encode(self, text: str) -> list[int]:

        escaped = [re.escape(token.decode()) for token in self.special_tokens]
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
        for i, chunk in enumerate(splitted):
            segments = []
            for segment in PAT.findall(chunk): 
                segments.append(segment.encode())
            for segment in segments:
                # Need this line to work with GPT2-vocab. Can't assume Had to get LLM help for this one.

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
        return NotImplementedError

    def decode(self, ids: list[int]) -> str:
        
        #decoded = sum([self.vocab[id] for id in ids])
        byte_string = b""
        for id in ids:
            byte_string += self.vocab[id]
        return str(byte_string, "utf-8", errors="replace")
    

    @staticmethod
    def compression_ratio(string: str, indices: list[int]) -> float:
        bytes_string = len(bytes(string, encoding="utf-8"))
        bytes_indices = len(indices)
        compression_ratio = bytes_string / bytes_indices
        return compression_ratio

    

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
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    print(type(tokenizer.vocab), type(tokenizer.merges))
    enc = tokenizer.encode("Lets test how lucky we can get")
    print(enc)
    dec = tokenizer.decode(enc)
    print(dec)