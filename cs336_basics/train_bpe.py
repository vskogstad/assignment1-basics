import cProfile
import pstats
from typing import BinaryIO

from cs336_basics.pretokenization import (find_chunk_boundaries,
                                          pretokenize_file)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges = vocab_size - 256 - len(special_tokens)
    vocab = {i:bytes([i]) for i in range(256)}
    vocab.update({256+i:token.encode("utf-8") for i, token in enumerate(special_tokens)})

    counts = pretokenize_file(filepath=input_path, num_processes=num_processes, special_tokens=special_tokens)
    candidates = find_merge_candidates(counts, vocab)

    return merge_pairs(candidates, num_merges=num_merges, counts=counts, vocab=vocab)

def find_merge_candidates(counts: dict, vocab: dict) -> dict[int, tuple]:
    """look through all dict entries, find pairs and add to new dict."""
    merge_candidates = {}
    for k, v in counts.items():
        for c1, c2 in zip(k, k[1:]):
            pair = (c1, c2) 
            merge_candidates[pair] = merge_candidates.get(pair, 0) + v

    # sort by number of occurences first, then "largest" characters in lexicographical order using vocab
    # Not happy about using the vocab dict as a sorting key. Perhaps an indication that merge candidates "should" not be stored as ints in the first place?
    merge_candidates = sorted(merge_candidates.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])), reverse=True)

    return merge_candidates

def merge_pairs(candidates, num_merges, counts, vocab):
    """merge num_merges most common pairs"""
    merges = []
    token_id = len(vocab)
    for _ in range(num_merges):
        best_pair = candidates[0][0] 

        token_a, token_b = best_pair
        bytes_a, bytes_b = vocab[token_a], vocab[token_b]

        #print(new_merge)
        vocab[token_id] = bytes_a + bytes_b  # need to return a byte mapping here
        #print("u")
        merges.append((bytes_a, bytes_b))
        counts = update_counts(counts, best_pair, token_id=token_id)
        #print(counts)
        candidates = find_merge_candidates(counts, vocab) # first thing to improve. need to combine this and previous step.
        #print(new_merge)
        token_id += 1

    return vocab, merges

def update_counts(counts, new_merge, token_id):
    #new_token = "".join(new_merge)
    new_counts = {}
    for k, v in counts.items():
        
        new_k = []
        skip = False
        update = False
        for c1, c2 in zip(k, k[1:]):
            # print(new_merge, (c1, c2))
            # need to merge in new token without screwing with existing...
            if skip:
                skip = False
                continue
            if new_merge == (c1, c2):
                #print(f"{c1, c2} should be merged")
                new_k.append(token_id)
                skip = True
                update = True
            else:
                new_k.append(c1)
                
        else:
            if update and not skip:
                new_k.append(c2)
        if update:
            #print(f"going from {k} to {tuple(new_k)}")
            k = tuple(new_k)
            
        new_counts[k] = new_counts.get(k, 0) + v

    return new_counts





if __name__ == "__main__":
    
    with cProfile.Profile() as profile:
        vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt", vocab_size=270, special_tokens=["<|endoftext|>","<|imstart|>"], num_processes=4)#TinyStoriesV2-GPT4-valid.txt", vocab_size=270, special_tokens=[])

        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)


    print(f"{len(vocab)=}, {vocab.items()=}, {merges[0]}")