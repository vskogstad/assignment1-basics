import os
import mmap
from collections import Counter
from multiprocessing import Pool
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


    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):

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

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def split_chunk(filepath: str, SPECIAL, PAT, start: int, end: int) -> Counter[tuple[int]: int]: 
    """Opens filepath and works on a chunk of the file specified by the start" and "end" params.
    The SPECIAL-pattern is for finding and splitting on special tokens.
    Maybe working on a shared dictionary would be better to avoid merging at the end? Locking might slow down more than you gain however.
    """
    # print(f"This is process {os.getpid()}")
    counts = Counter()

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        split_special = re.split(SPECIAL, chunk)
        for document in split_special:
            for word in PAT.finditer(document): # Could potential fail if large and no <|endoftex|>. Use re.finditer() if problematic.
                counts[tuple(word.group().encode())] += 1
                
    return counts

def pretokenize_file(filepath: str, num_processes: int, special_tokens: list[str]) -> dict[str, int]:
    # Preprocessing pattern for special tokens 
    escaped = [re.escape(token) for token in special_tokens]
    SPECIAL = r"|".join(escaped)
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        

    # Multiprocessing

    with Pool(num_processes) as p:
        args = [(filepath, SPECIAL, PAT, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
        collected = p.starmap(split_chunk, args)
    
    # Merge dictionaries from each process
    counts = collected[0]
    for d in collected[1:]:
        counts.update(d)
    return counts