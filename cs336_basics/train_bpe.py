import cProfile
import pstats
from collections import Counter
from typing import BinaryIO

from cs336_basics.pretokenization import find_chunk_boundaries, pretokenize_file

# Latest: Instead of iterating over the entire candidates dict each time. Keep a best_pairs dictionary with a list of n best pairs. 
#       When adding new tokens to the vocab, iterate over new tokens and at them to the best_pairs list if they are above the minimum threshold in the list. 
#       Once the value of the highest pair in best_pairs is below the threshold, we can no longer guarantee that it contains the best merging candidate, we then do a full iteration and create a new best_pair list.


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Trains a BPE-tokenizer of a given vocab size on the data located at input path"""

    num_merges = vocab_size - 256 - len(special_tokens)
    vocab = {i:bytes([i]) for i in range(256)}
    vocab.update({256+i:token.encode("utf-8") for i, token in enumerate(special_tokens)})

    counts = pretokenize_file(filepath=input_path, num_processes=num_processes, special_tokens=special_tokens)
    candidates, used_words = find_initial_merge_candidates(counts, vocab)
    
    # Start merging tokens and updating the initial dictionaries incrementally.
    merges = []
    token_id = len(vocab)
    best_pairs = {"max_pairs_sorted":[], 
                "good_pairs":set(), 
                "resorting_value":0,
                "resort_needed":True
                } 
    
    for _ in range(num_merges):
        

        candidates, best_pairs = find_best_pair(candidates, vocab, best_pairs) # sorted(candidates.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])), reverse=True)[0][0] 
        
        token_a, token_b = best_pairs["max_pairs_sorted"][-1] # sorted in reverse order for efficient poping.
        bytes_a, bytes_b = vocab[token_a], vocab[token_b]

        vocab[token_id] = bytes_a + bytes_b  # need to return a byte mapping
        merges.append((bytes_a, bytes_b))

        counts, candidates, used_words, best_pairs = update_dictionaries(counts=counts, candidates=candidates, used_words=used_words, bps=best_pairs, token_id=token_id)
        token_id += 1

    return vocab, merges

def find_initial_merge_candidates(counts: Counter, vocab: dict) -> tuple[dict[int, tuple], dict[set]]:
    """look through all dict entries, find pairs and add to new dict."""
    merge_candidates = Counter() # Counts occurences of every pair of tokens
    used_words = {} # Keys: pair or tokens | Value: set of all words containg this pair
    for word_tuple, num_occurences in counts.items():
        for c1, c2 in zip(word_tuple, word_tuple[1:]):
            pair = (c1, c2) 
            merge_candidates[pair] += num_occurences
            used_words[pair] = used_words.get(pair, set()) 
            used_words[pair].add(word_tuple)
    
    return merge_candidates, used_words


def find_best_pair(candidates: Counter[tuple[int,int]: int], vocab, bps: dict[list[tuple[int, int]], list[tuple[int, int]], int, int]):
    """Takes unordered dicts of candidates and finds the best pair in linear time"""
    
    if not bps["resort_needed"]: # and bps["max_pairs_sorted"]:
        return candidates, bps
    
    # if the list is empty or max_value in list is lower than limit we sort candidates dict.
    pairs_under = set()
    if bps["good_pairs"]:
        sort_needed = True
        for pair in bps["good_pairs"]:
            if candidates[pair] >= bps["resorting_limit"]:
                sort_needed = False
                break
            else:
                pairs_under.add(pair)
                
    [bps["good_pairs"].remove(pair) for pair in pairs_under]
    
    if not bps["good_pairs"] or sort_needed:
        print("Finding new good pairs (sorting)")
        # sort by number of occurences first, then "largest" characters in lexicographical order using vocab
        sorted_list = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
        # print(sorted_list[:15])
        candidates = Counter({pair:candidates[pair] for pair, _ in sorted_list}) # Dubious. Sorting for marginally faster sorting next time
        # print(candidates)
        # iterate from starting 
        # resorting_limit, list of n best pairs


        
        cap = 10 # this is a hyper-parameter. Too small is bad. For 10k merges 16s with 4, 32 s with 2.
        start = len(sorted_list) // cap
        # Iterate from "start" point until value decreases. All of these will be included in best_pairs
        bps["resorting_limit"] = sorted_list[start][1]
        pos = start
        
        while pos < len(sorted_list):
            if sorted_list[pos][1] < bps["resorting_limit"]:
                break
            pos += 1
        bps["good_pairs"] = set(pair for pair, _ in sorted_list[:pos])


    # We now go through the good_pairs_list lineary and find the best pair
    max_num = 0
    max_pair = []

    # sort by number of occurences first, 
    for pair in bps["good_pairs"]:
        num = candidates[pair]
        if num < max_num:
            continue
        if num == max_num:
            max_pair.append(pair)
        else:
            max_num = num
            max_pair = [pair]
    
    bps["max_pairs_sorted"] = max_pair


    # If tie, sort by "largest" characters in lexicographical order using vocab
    if len(max_pair) > 1:
        bps["max_pairs_sorted"] = sorted(max_pair, key=lambda x: (vocab[x[0]], vocab[x[1]]), reverse=False)
    
    
    return candidates, bps

    

def update_dictionaries(counts: Counter[int], candidates: Counter[int], used_words: dict[list], bps: dict, token_id: int):
    """Iterate over used words in the best pair. 
    For the best pair, go through all of the words in used_words, update to the new merged token and update:
        -The counts dictionary with new keys (the tokens) has updated after merging.
        -The candidates dictionary with new counts for all neigbouring pairs in the used words.
        -The used_words dictionary and remove/add links to words that have lost and gained pairs after the merge(tup and ntup).
    """
    """best_pairs = {"max_pairs_sorted":[], 
                      "best_pairs":[], 
                      "resorting_value":0,
                      "resort_needed":True
                      } """
    new_candidates = Counter()
    best_pair = bps["max_pairs_sorted"].pop(-1) # popping the best pair
    bps["good_pairs"].remove(best_pair)

    for word_tuple in used_words[best_pair]:
        
        num_occurences = counts[word_tuple]
        word_tokenization = []
        skip = False
        last_pair = None
        # need to merge in new token without screwing with existing...
        for c1, c2 in zip(word_tuple, word_tuple[1:]):
           
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

        # old pairs are removed       
        for tup in old_pairs:
            if tup != best_pair: # cant remove as we are iterating over used_words[best_pair]
                used_words[tup].remove(word_tuple) 
        # new pairs are added
        for ntup in new_pairs:
            used_words[ntup] = used_words.get(ntup, set())
            used_words[ntup].add(new_k) 

    # Checking if we can skip resorting our pairs. The base idea here is:
    # - we are minting a new token, so we will only possibly decrease values of our existing max_pairs_sorted. 
    # - After merging we create the new pairs shown in new_candidates, those that are above the minimum value gets added to good pairs
    # - The merging can create have a new pair which have n_best_pair new tokens, this will force a recalculation of best_pairs_sorted.

    bps["resort_needed"] = True # default we will have to resort.
    # print(bps["max_pairs_sorted"])
    # Loop over max_pairs and check if they exist in new_candidates. If so they are no longer a max_pair, and we remove them.
    bps["max_pairs_sorted"] = [pair for pair in bps["max_pairs_sorted"] if pair not in new_candidates.keys()] 
    # print("after:", bps["max_pairs_sorted"])
    if bps["max_pairs_sorted"]: # Need to have items left if we should skip resorting
        bps["resort_needed"] = False
    
    for pair, num_occurences in new_candidates.items():
        candidates[pair] += num_occurences
        if num_occurences >= bps["resorting_value"]:
            bps["good_pairs"].add(pair)
            if num_occurences == candidates[best_pair]: # We have a new best_pair candidate
                bps["resort_needed"] = True
    # candidates.update(new_candidates)
    assert candidates[best_pair] == 0 #Temp check. Delete best_pair directly to save excess updates.
    #import sys; sys.exit()
    del candidates[best_pair]
    del used_words[best_pair]
    

    return counts, candidates, used_words, bps





if __name__ == "__main__":
    
    with cProfile.Profile() as profile:
        vocab, merges = train_bpe(input_path="data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=["<|endoftext|>","<|imstart|>"], num_processes=8)#TinyStoriesV2-GPT4-valid.txt", vocab_size=270, special_tokens=[])

        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)
        
        longest_token = max(vocab.values(), key=lambda x: len(x.__repr__()))
        print(longest_token)
        # save data
        import sys; sys.exit()
        import json
        with open("cs336_basics/tokenizer_owt.json","w") as f:
            json.dump((vocab, merges), f, default=repr, indent=4)




