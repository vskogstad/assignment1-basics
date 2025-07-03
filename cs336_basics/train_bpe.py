import cProfile
import pstats
from collections import Counter
from typing import BinaryIO

from cs336_basics.pretokenization import (find_chunk_boundaries,
                                          pretokenize_file)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Trains a BPE-tokenizer of a given vocab size on the data located at input path"""

    num_merges = vocab_size - 256 - len(special_tokens)
    vocab = {i:bytes([i]) for i in range(256)}
    vocab.update({256+i:token.encode("utf-8") for i, token in enumerate(special_tokens)})

    counts = pretokenize_file(filepath=input_path, num_processes=num_processes, special_tokens=special_tokens)
    candidates, used_words = find_merge_candidates(counts, vocab)
    return merge_pairs(candidates, used_words, num_merges=num_merges, counts=counts, vocab=vocab)

def find_merge_candidates(counts: Counter, vocab: dict) -> tuple[dict[int, tuple], dict[set]]:
    """look through all dict entries, find pairs and add to new dict."""
    merge_candidates = Counter() # Counts occurences of every pair of tokens
    used_words = {} # Keys: pair or tokens | Value: set of all words containg this pair
    for word_tuple, num_occurences in counts.items():
        for c1, c2 in zip(word_tuple, word_tuple[1:]):
            pair = (c1, c2) 
            merge_candidates[pair] += num_occurences
            #merge_candidates[pair] = merge_candidates.get(pair, 0) + v
            used_words[pair] = used_words.get(pair, set()) 
            used_words[pair].add(word_tuple)
    
    return merge_candidates, used_words

def merge_pairs(candidates, used_words, num_merges, counts, vocab):
    """merge num_merges most common pairs"""
    merges = []
    token_id = len(vocab)
    for _ in range(num_merges):
        # sort by number of occurences first, then "largest" characters in lexicographical order using vocab
        # Not happy about using the vocab dict as a sorting key. Perhaps an indication that merge candidates "should" not be stored as ints in the first place?
        best_pair = sorted(candidates.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])), reverse=True)[0][0] 
        
        token_a, token_b = best_pair
        bytes_a, bytes_b = vocab[token_a], vocab[token_b]

        #print(new_merge)
        vocab[token_id] = bytes_a + bytes_b  # need to return a byte mapping here
        #print("u")
        merges.append((bytes_a, bytes_b))
        counts, candidates, used_words = update_candidates(counts=counts,candidates=candidates, used_words=used_words, best_pair=best_pair, token_id=token_id)
        #print(counts)
        #candidates, used_words = find_merge_candidates(counts, vocab) # first thing to improve. need to combine this and previous step.
        #print(new_merge)
        token_id += 1
        #print(merges)
    return vocab, merges

def update_candidates(counts, candidates, used_words, best_pair, token_id):
    # Iterate over used words in the best pair. 
    # Count pairs using old and new encoding and update candidate and counts dictionary with the difference
    new_candidates = Counter()
    words_to_merge = used_words[best_pair]
    
    for word in words_to_merge:
        
        word_tuple, num_occurences = word, counts[word]
        
        word_tokenization = []
        skip = False
        last_pair = None
        # iterates over e
        for c1, c2 in zip(word_tuple, word_tuple[1:]):
            # print(new_merge, (c1, c2))
            # need to merge in new token without screwing with existing...
            if skip:
                skip = False
                # for new pair on right side of merge
                new_candidates[(c1, c2)] -= num_occurences
                new_candidates[(token_id, c2)] += num_occurences
                last_pair = (token_id, c2)
                continue

            if best_pair == (c1, c2):
                word_tokenization.append(token_id)
                skip = True
                if last_pair: # for new pair on left side of merge
                    c1_old, c2_old = last_pair
                    new_candidates[(c1_old, c2_old)] -= num_occurences
                    new_candidates[(c1_old, token_id)] += num_occurences

                # remove now merged pair
                new_candidates[(c1, c2)] -= num_occurences
                # Temporary! For speed, we don't bother updating counts of the original pair back to 0. Instead we delete it directly

            else:
                word_tokenization.append(c1)

            last_pair = c1, c2

        else:
            if not skip:
                word_tokenization.append(c2)

        # update the counts dictionary
        del counts[word_tuple]
        #print(f"deleted counts[{word_tuple=}]")
        new_k = tuple(word_tokenization)
        counts[new_k] += num_occurences

        # update used words, should ideally be done during first loop:
        old_pairs = set((a, b) for a, b in zip(word_tuple, word_tuple[1:]))
        new_pairs = set((a, b) for a, b in zip(new_k, new_k[1:]))

        #old pairs not in new are removed
        #print(f"{k=}, {v=}  |  {old_pairs=}, {new_pairs=}, {old_pairs.difference(new_pairs)=}, {new_pairs.difference(old_pairs)=}")
        #import sys; sys.exit()
        
        for tup in old_pairs:
            if tup != best_pair:
                used_words[tup].remove(word_tuple) 
        for ntup in new_pairs:
            used_words[ntup] = used_words.get(ntup, set())
            used_words[ntup].add(new_k) 
        

    #print(new_candidates)
    # Delete best_pair directly to save excess updates.
    candidates.update(new_candidates)
    #print(candidates[best_pair])
    assert candidates[best_pair] == 0
    #import sys; sys.exit()
    del candidates[best_pair]
    del used_words[best_pair]


    return counts, candidates, used_words





if __name__ == "__main__":
    
    with cProfile.Profile() as profile:
        vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-valid.txt", vocab_size=270, special_tokens=["<|endoftext|>","<|imstart|>"], num_processes=4)#TinyStoriesV2-GPT4-valid.txt", vocab_size=270, special_tokens=[])

        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)


    print(f"{len(vocab)=}, {vocab.items()=}, {merges[0]}")