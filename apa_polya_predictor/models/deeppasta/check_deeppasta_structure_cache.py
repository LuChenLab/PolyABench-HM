#!/usr/bin/env python3
import argparse
import json
import numpy as np
from collections import Counter


def read_fasta_headers(fasta):
    headers = []
    with open(fasta) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                headers.append(line[1:])
    return headers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--seq_cache", required=True)
    parser.add_argument("--struct_cache", required=True)
    parser.add_argument("--seq_len", type=int, default=240)
    parser.add_argument("--n_structures", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=200000)
    args = parser.parse_args()

    ss_path = args.struct_cache + ".ss.uint8.npy"
    ss_headers_path = args.struct_cache + ".ss.headers.npy"
    ss_meta_path = args.struct_cache + ".ss.meta.json"

    seq_x_path = args.seq_cache + ".x.uint8.npy"
    seq_y_path = args.seq_cache + ".y.npy"
    seq_headers_path = args.seq_cache + ".headers.npy"

    print("[1] loading structure cache")
    ss = np.load(ss_path, mmap_mode="r")
    ss_headers = np.load(ss_headers_path, allow_pickle=True)
    seq_x = np.load(seq_x_path, mmap_mode="r")
    seq_y = np.load(seq_y_path, mmap_mode="r")
    seq_headers = np.load(seq_headers_path, allow_pickle=True)

    print("ss shape:", ss.shape)
    print("ss dtype:", ss.dtype)
    print("seq_x shape:", seq_x.shape)
    print("seq_y shape:", seq_y.shape)
    print("ss_headers:", ss_headers.shape)
    print("seq_headers:", seq_headers.shape)

    assert ss.ndim == 3, "ss should be [N, 3, L]"
    assert ss.shape[1] == args.n_structures, f"ss second dim should be {args.n_structures}"
    assert ss.shape[2] == args.seq_len, f"ss length should be {args.seq_len}"
    assert ss.dtype == np.uint8, "ss dtype should be uint8"

    assert seq_x.shape[0] == ss.shape[0], "sequence cache N != structure cache N"
    assert seq_y.shape[0] == ss.shape[0], "label N != structure cache N"
    assert len(seq_headers) == ss.shape[0], "seq headers N != structure cache N"
    assert len(ss_headers) == ss.shape[0], "ss headers N != structure cache N"

    print("[2] checking value range and class counts")
    total_counts = Counter()
    global_min = 99
    global_max = -1

    for start in range(0, ss.shape[0], args.chunk_size):
        end = min(start + args.chunk_size, ss.shape[0])
        block = ss[start:end]

        bmin = int(block.min())
        bmax = int(block.max())
        global_min = min(global_min, bmin)
        global_max = max(global_max, bmax)

        vals, counts = np.unique(block, return_counts=True)
        for v, c in zip(vals, counts):
            total_counts[int(v)] += int(c)

        print(f"  checked {end}/{ss.shape[0]}")

    print("global min/max:", global_min, global_max)
    print("class counts:", dict(sorted(total_counts.items())))

    assert global_min >= 0, "ss has value < 0"
    assert global_max <= 6, "ss has value > 6"

    print("[3] checking FASTA order")
    fasta_headers = read_fasta_headers(args.fasta)

    assert len(fasta_headers) == ss.shape[0], "FASTA N != structure cache N"

    same_fasta_order = list(ss_headers) == fasta_headers
    print("same order with FASTA:", same_fasta_order)
    assert same_fasta_order, "structure headers are not in original FASTA order"

    print("[4] checking sequence cache header order")
    same_seq_order = list(ss_headers) == list(seq_headers)
    print("same order with sequence cache:", same_seq_order)
    assert same_seq_order, "structure headers are not aligned with sequence cache headers"

    print("[5] meta")
    with open(ss_meta_path) as f:
        meta = json.load(f)
    print(json.dumps(meta, indent=2)[:2000])

    print("\nPASS: DeepPASTA structure cache is usable.")


if __name__ == "__main__":
    main()
