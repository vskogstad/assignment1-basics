import json
import pickle
from collections.abc import Iterable, Iterator
from itertools import chain, accumulate
import regex as re


class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        
        # find special tokens in vocab as well as additional
        self.special_tokens = [tok for id, tok in vocab.items() if str(tok).startswith("<|")]
        new_special_tokens = []
        if special_tokens:
            new_special_tokens = [tok for tok in special_tokens if tok not in self.special_tokens]
            print(new_special_tokens)
        self.special_tokens += new_special_tokens 
        self.vocab.update({id:vocab for id, vocab in enumerate(new_special_tokens, start= len(self.vocab))})
        self.vocab_to_int = {value:key for key, value in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = json.load(vocab_filepath)
        merges = json.load(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:

        escaped = [re.escape(token) for token in self.special_tokens]
        SPECIAL = r"|".join(escaped)
        #text = text.encode("utf-8")
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        segments = []
        for segment in PAT.findall(text): 
            segments.append(segment.encode())
        # merge

        indeces = []
        for segment in segments:
            for best_pair in self.merges:
                if len(segment) <= 1: # No more meges possible
                    break
                a, b = best_pair
                new_token = a + b
                #print(self.vocab_to_int["b' t'"])
                new_token_id = self.vocab_to_int[new_token]
                best_pair = self.vocab_to_int[a], self.vocab_to_int[b]
                skip = False
                word_tokenization = []
                #print(best_pair)
                for c1, c2 in zip(segment, segment[1:]):
                    #print(type(c1), type(c2), type(best_pair[0]), type(b))
                    #import sys; sys.exit()
                    if skip:
                        skip = False
                        continue

                    if best_pair == (c1, c2):
                        skip = True
                        word_tokenization.append(new_token_id)

                    else:
                        word_tokenization.append(c1)

                else:
                    if not skip:
                        word_tokenization.append(c2)
                segment = word_tokenization
            indeces.append(segment)

        return list(chain.from_iterable(indeces))
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return NotImplementedError

    def decode(self, ids: list[int]) -> str:
        #decoded = sum([self.vocab[id] for id in ids])
        byte_string = b""
        for id in ids:
            byte_string += self.vocab[id]
        return str(byte_string, "utf-8", errors="replace")
    
    # @staticmethod
    # def pretokenize_string(text: str, special_tokens: list[str]) -> dict[str, int]:
    #     # Preprocessing pattern for special tokens 
    #     escaped = [re.escape(token) for token in special_tokens]
    #     SPECIAL = r"|".join(escaped)
    #     # general word-like matching
    #     PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    #     counts = set()
    #     #text_chunk = text.decode("utf-8", errors="ignore")
    #     split_special = re.split(SPECIAL, text)
    #     for document in split_special:
    #         for word in PAT.findall(document): # Way faster than re.finditer(). Should not be problematic.
    #             counts.add(word.encode())

    #     return counts   
    @staticmethod
    def compression_ratio(string: str, indices: list[int]) -> float:
        bytes_string = len(bytes(string, encoding="utf-8"))
        bytes_indices = len(indices)
        compression_ratio = bytes_string / bytes_indices
        return compression_ratio

    

if __name__ == "__main__":
    vocab = {
        "98": "b'b'",
        "99": "b'c'",
        "100": "b'd'",
        "101": "b'e'",
        "102": "b'f'",
        "103": "b'g'",
        "104": "b'h'",
        "105": "b'i'",
        "106": "b'j'",
        "107": "b'k'",
        "108": "b'l'",
        "109": "b'm'",
        }
    special_tokens = ["<|imstart|>"]
    merges = [
        [
            "b' '",
            "b't'"
        ],
        [
            "b'h'",
            "b'e'"
        ],
        [
            "b' '",
            "b'a'"
        ],
        [
            "b' '",
            "b's'"
        ],
        [
            "b' '",
            "b'w'"
        ],
        ]
    with open("cs336_basics/vocab-tiny.pkl", "rb") as vocab_file:
        vocab = pickle.load(vocab_file)
    with open("cs336_basics/merges-tiny.pkl", "rb") as merges_file:
        merges = pickle.load(merges_file)

    #with open("cs336_basics/tokenizer_tinystories.json","r") as f:
    #        vocab, merges = json.load(fp=f ) #json.dump(model, f, default=repr, indent=4)
    a = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    enc = a.encode("Hello, how are you?")
    print(enc)
    dec = a.decode(enc)
    print(dec)