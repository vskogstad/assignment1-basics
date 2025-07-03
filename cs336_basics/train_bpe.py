import cProfile
import pstats
from typing import BinaryIO
from collections import Counter

from cs336_basics.pretokenization import (find_chunk_boundaries,
                                          pretokenize_file)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges = vocab_size - 256 - len(special_tokens)
    vocab = {i:bytes([i]) for i in range(256)}
    vocab.update({256+i:token.encode("utf-8") for i, token in enumerate(special_tokens)})

    counts = pretokenize_file(filepath=input_path, num_processes=num_processes, special_tokens=special_tokens)
    candidates, used_words = find_merge_candidates(counts, vocab)

    return merge_pairs(candidates, used_words, num_merges=num_merges, counts=counts, vocab=vocab)

def find_merge_candidates(counts: dict, vocab: dict) -> dict[int, tuple]:
    """look through all dict entries, find pairs and add to new dict."""
    merge_candidates = Counter()
    used_words = {}
    for k, v in counts.items():
        for c1, c2 in zip(k, k[1:]):
            pair = (c1, c2) 
            merge_candidates[pair] += v
            #merge_candidates[pair] = merge_candidates.get(pair, 0) + v
            used_words[pair] = used_words.get(pair, []) 
            used_words[pair].append(k)
    # sort by number of occurences first, then "largest" characters in lexicographical order using vocab
    # Not happy about using the vocab dict as a sorting key. Perhaps an indication that merge candidates "should" not be stored as ints in the first place?
    merge_candidates = sorted(merge_candidates.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])), reverse=True)

    return merge_candidates, used_words

def merge_pairs(candidates, used_words, num_merges, counts, vocab):
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
        counts = update_counts(counts=counts,used_words=used_words, best_pair=best_pair, token_id=token_id)
        #print(counts)
        candidates, used_words = find_merge_candidates(counts, vocab) # first thing to improve. need to combine this and previous step.
        #print(new_merge)
        token_id += 1

    return vocab, merges

def update_counts(counts, used_words, best_pair, token_id):
    # Iterate over used words in last pair. 
    # Count pairs using old and new encoding and update counts dictionary with the difference
    new_counts = Counter()

    for word in used_words:
        k, v = word, counts[word]
        
        new_k = []
        skip = False
        last_pair = None
        for c1, c2 in zip(k, k[1:]):
            # print(new_merge, (c1, c2))
            # need to merge in new token without screwing with existing...
            if skip:
                skip = False
                # for pair bordering on right side
                c1_old, c2_old = last_pair
                new_counts[(c1_old, c2_old)] -= v
                new_counts[(token_id, c2)] += v
                last_pair = (token_id, c2)
                continue
            if best_pair == (c1, c2):
                new_k.append(token_id)
                skip = True
                # update counts dict
                if last_pair: # for pair bordering on left side
                    c1_old, c2_old = last_pair
                    new_counts[(c1_old, c2_old)] -= v
                    new_counts[(c1_old, token_id)] += v
                # remove now merged pair
                new_counts[(c1, c2)] -= v
                # Temporary! For speed, we don't bother updating counts of the original pair back to 0. Instead we delete it directly

                # update = True
            else:
                new_k.append(c1)

            last_pair = c1, c2

        else:
            if not skip:
                new_k.append(c2)
        # if update:
            #print(f"going from {k} to {tuple(new_k)}")
        k = tuple(new_k)
            
        new_counts[k] = new_counts.get(k, 0) + v


    print(new_counts)
    # Delete best_pair directly to save excess updates.
    counts.update(new_counts)
    assert counts[best_pair] == 0



    return counts





if __name__ == "__main__":
    
    with cProfile.Profile() as profile:
        vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt", vocab_size=270, special_tokens=["<|endoftext|>","<|imstart|>"], num_processes=4)#TinyStoriesV2-GPT4-valid.txt", vocab_size=270, special_tokens=[])

        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)


    print(f"{len(vocab)=}, {vocab.items()=}, {merges[0]}")