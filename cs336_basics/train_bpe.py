import cProfile
import pstats
from typing import BinaryIO

from cs336_basics.pretokenization import (find_chunk_boundaries,
                                          pretokenize_file)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges = vocab_size - 255
    counts = pretokenize_file(filepath=input_path, num_processes=4, special_tokens=special_tokens)
    candidates = find_merge_candidates(counts)

    return merge_pairs(candidates, num_merges=num_merges, counts=counts)

def find_merge_candidates(counts):
    # look through all dict antries, find pairs and add to new dict.
    merge_candidates = {}
    for k, v in counts.items():
        #k = k.encode("utf-8")
        for c1, c2 in zip(k, k[1:]):
            byte_pair = (c1, c2) #(bytes([c1]), bytes([c2]))
            merge_candidates[byte_pair] = merge_candidates.get(byte_pair, 0) + v

    # sort by number of occurences first, then "largest" characters in pair
    merge_candidates = sorted(merge_candidates.items(), key=lambda x: (x[1], x[0]), reverse=True)
    #print(merge_candidates)
    return merge_candidates

def merge_pairs(candidates, num_merges, counts):
    # merge num_merges most common pairs
    merges = []
    vocab = {i:chr(i).encode() for i in range(256)}
    for i in range(num_merges):
        new_merge = candidates[0][0] #bytes(candidates[0][0]) #tuple(bytes([i]) for i in candidates[0][0])
        #print(f"new merge {new_merge}")
        token_id = 255+i
        #print(new_merge)
        vocab[token_id] = new_merge  # need to return a byte mapping here
        #print("u")
        merges.append(new_merge)
        counts = update_counts(counts, new_merge, token_id=token_id)
        #print(counts)
        candidates = find_merge_candidates(counts) # first thing to improve. need to combine this and previous step.
        #print(new_merge)

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
        vocab, merges = train_bpe(input_path="data/minimal.txt", vocab_size=270, special_tokens=["<|endoftext|>","<|imstart|>"])#TinyStoriesV2-GPT4-valid.txt", vocab_size=270, special_tokens=[])

        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)


    print(f"{len(vocab)=}, {vocab.items()=}, {merges[0]}")