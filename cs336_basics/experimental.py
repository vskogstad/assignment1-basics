import cProfile
import pstats
import regex as re


def pre_tokenize(file):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(file) as f:
        text = f.read()
    split_text = re.findall(PAT, text) #"some text that i'll pre-tokenize")
    counts = dict()
    for word in split_text:
        counts[tuple(word)] = counts.get(tuple(word), 0) + 1
    print(counts)
    return counts

def find_merge_candidates(counts):
    # look through all dict antries, find pairs and add to new dict.
    merge_candidates = {}
    for k, v in counts.items():
        for c1, c2 in zip(k, k[1:]):
            merge_candidates[(c1, c2)] = merge_candidates.get((c1, c2), 0) + v

    # sort by number of occurences first, then "largest" characters in pair
    merge_candidates = sorted(merge_candidates.items(), key=lambda x: (x[1], x[0]), reverse=True)
    print(merge_candidates)
    return merge_candidates

def merge_pairs(candidates, num_merges, counts):
    # merge num_merges most common pairs
    merges = []
    for _ in range(num_merges):
        new_merge = candidates[0][0]
        merges.append(new_merge)
        update_counts(counts, new_merge)
        candidates = find_merge_candidates(candidates) # first thing to improve. need to combine this and previous step.
        print(new_merge)

    return merges

def update_counts(counts, new_merge):
    for k, v in counts.items():
        for c1, c2 in zip(k, k[1:]):
            # need to merge in new token without screwing with existing...
            if new_merge == (c1, c2):

                
    return candidates

if __name__ == '__main__':
    with cProfile.Profile() as profile:
        num_merges = 1

        counts = pre_tokenize("data/minimal.txt")
        candidates = find_merge_candidates(counts)
        merges = merge_pairs(candidates, num_merges=num_merges, counts=counts)

        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)


