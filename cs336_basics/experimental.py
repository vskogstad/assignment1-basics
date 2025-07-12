import cProfile
import pstats

import regex as re  # supports negative look-ahead


def pre_tokenize(file_path):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(file_path) as f:
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
        counts = update_counts(counts, new_merge)
        candidates = find_merge_candidates(counts) # first thing to improve. need to combine this and previous step.
        print(new_merge)

    return merges

def update_counts(counts, new_merge):
    new_token = "".join(new_merge)
    new_counts = {}
    for k, v in counts.items():
        
        new_k = []
        skip = False
        update = False
        for c1, c2 in zip(k, k[1:]):
            # need to merge in new token without screwing with existing...
            if skip:
                skip = False
                continue
            if new_merge == (c1, c2):
                print(f"{c1, c2} should be merged")
                new_k.append(new_token)
                skip = True
                update = True
            else:
                new_k.append(c1)
                
        else:
            if update and not skip:
                new_k.append(c2)
        if update:
            print(f"going from {k} to {tuple(new_k)}")
            k = tuple(new_k)
        new_counts[k] = new_counts.get(k, 0) + v

    return new_counts

if __name__ == '__main__':
    # Test if it's I/O bound
    import time

    # Time just reading the file
    start = time.time()
    with open("data/TinyStoriesV2-GPT4-train.txt", 'rb') as f:
        data = f.read()
    read_time = time.time() - start
    print(f"Pure read time: {read_time:.2f}s")
    print(f"Read speed: {len(data) / read_time / 1024 / 1024:.2f} MB/s")
    import sys; sys.exit()

    with cProfile.Profile() as profile:
        num_merges = 15
        file = "data/minimal.txt"
        counts = pre_tokenize(file_path=file)
        candidates = find_merge_candidates(file)
        merges = merge_pairs(candidates, num_merges=num_merges, counts=counts)
        print(merges)
        result = pstats.Stats(profile)
        result.sort_stats(pstats.SortKey.TIME)
        result.print_stats(10)



