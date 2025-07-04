import os
from collections import Counter
from multiprocessing import Lock, Pool
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    
    chunk_bu_boundaries = chunk_boundaries[:]
    backup_split_token = ".\n".encode("utf-8")


    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        no_backup = True
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            # Find a backup breakpoint
            if no_backup:
                found_bu = mini_chunk.find(backup_split_token)
                if found_bu != -1:
                    chunk_bu_boundaries[bi] = initial_position + found_bu
                    no_backup = False

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    print(f"{sorted(set(chunk_boundaries))=}, {sorted(set(chunk_bu_boundaries))}")
    return sorted(set(chunk_boundaries))

def split_chunk(filepath, SPECIAL, start, end):
    print(f"This is process {os.getpid()}")
    counts = Counter()
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #lock = Lock()
        split_special = re.split(SPECIAL, chunk)
        for document in split_special:
            for word in PAT.findall(document): # Could potential fail if large and no <|endoftex|>. Use re.finditer() if problematic.
                counts[tuple(word.encode())] += 1
                
    return counts

def pretokenize_file(filepath: str, num_processes: int, special_tokens: list[str]) -> dict[str, int]:
    # Preprocessing special tokens 
    escaped = [re.escape(token) for token in special_tokens]
    SPECIAL = r"|".join(escaped)
    
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        

    # Multiprocessing
    p = Pool(num_processes)
    args = [(filepath, SPECIAL, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    collected = p.starmap(split_chunk, args)
    
    # Hopefully temporary code to merge dictionaries from each process
    counts = collected[0]
    for d in collected[1:]:
        for k, v in d.items():
            counts[k] = counts.get(k, 0) + v
    return counts