#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

import numpy as np


BASES = "ACGT"
UNK_TOKEN = "<UNK>"


def iter_fasta(path, max_records=None):
    header = None
    chunks = []
    n = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                    n += 1
                    if max_records is not None and n >= max_records:
                        return

                header = line[1:]
                chunks = []
            else:
                chunks.append(line)

        if header is not None:
            yield header, "".join(chunks)


def seq_to_kmer_tokens(seq, k=5):
    seq = seq.upper().replace("U", "T")
    tokens = []

    for i in range(0, len(seq) - k + 1):
        kmer = seq[i:i + k]

        if set(kmer).issubset(set(BASES)):
            tokens.append(kmer)
        else:
            tokens.append(UNK_TOKEN)

    return tokens


class FastaKmerCorpus:
    def __init__(self, fasta, k=5, seq_len=240, max_records=None):
        self.fasta = fasta
        self.k = int(k)
        self.seq_len = int(seq_len)
        self.max_records = max_records

    def __iter__(self):
        for header, seq in iter_fasta(self.fasta, max_records=self.max_records):
            seq = seq.upper()

            if len(seq) != self.seq_len:
                raise ValueError(f"{header}: length={len(seq)}, expected={self.seq_len}")

            tokens = seq_to_kmer_tokens(seq, k=self.k)

            if len(tokens) > 0:
                yield tokens


def kmer_to_index(kmer):
    idx = 0
    for ch in kmer:
        idx *= 4
        idx += BASES.index(ch)
    return idx


def index_to_kmer(idx, k=5):
    chars = []
    x = int(idx)

    for _ in range(k):
        chars.append(BASES[x % 4])
        x //= 4

    return "".join(reversed(chars))


def count_fasta_records(fasta, max_records=None):
    n = 0
    for _header, _seq in iter_fasta(fasta, max_records=max_records):
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain PASNet 5-mer Word2Vec embedding from train FASTA."
    )

    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out_prefix", required=True)

    parser.add_argument("--kmer", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=240)
    parser.add_argument("--embed_dim", type=int, default=90)

    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--batch_words", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--max_records", type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise ImportError(
            "gensim is not installed. Install it in your biobench env, for example: "
            "pip install gensim"
        ) from e

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    corpus = FastaKmerCorpus(
        fasta=args.fasta,
        k=args.kmer,
        seq_len=args.seq_len,
        max_records=args.max_records,
    )

    logging.info("Counting FASTA records...")
    n_records = count_fasta_records(args.fasta, max_records=args.max_records)
    logging.info(f"FASTA records: {n_records}")

    logging.info("Building Word2Vec model...")
    model = Word2Vec(
        vector_size=args.embed_dim,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=1,
        negative=args.negative,
        sample=args.sample,
        seed=args.seed,
        batch_words=args.batch_words,
    )

    logging.info("Building vocabulary...")
    model.build_vocab(corpus)
    logging.info(f"Word2Vec corpus_count: {model.corpus_count}")
    logging.info(f"Word2Vec vocab size: {len(model.wv)}")

    logging.info("Training Word2Vec...")
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=args.epochs,
    )

    vocab_size = 4 ** args.kmer + 1
    unk_idx = 4 ** args.kmer

    rng = np.random.default_rng(args.seed)
    embedding = rng.normal(
        loc=0.0,
        scale=0.02,
        size=(vocab_size, args.embed_dim),
    ).astype(np.float32)

    found = 0
    missing = 0

    for idx in range(4 ** args.kmer):
        kmer = index_to_kmer(idx, k=args.kmer)

        if kmer in model.wv:
            embedding[idx] = model.wv[kmer].astype(np.float32)
            found += 1
        else:
            missing += 1

    if UNK_TOKEN in model.wv:
        embedding[unk_idx] = model.wv[UNK_TOKEN].astype(np.float32)
        unk_strategy = "word2vec_UNK"
    else:
        embedding[unk_idx] = embedding[:unk_idx].mean(axis=0)
        unk_strategy = "mean_of_ACGT_kmers"

    npy_path = str(out_prefix) + f".kmer{args.kmer}.dim{args.embed_dim}.embedding.npy"
    txt_path = str(out_prefix) + f".kmer{args.kmer}.dim{args.embed_dim}.word2vec.txt"
    meta_path = str(out_prefix) + f".kmer{args.kmer}.dim{args.embed_dim}.meta.json"

    np.save(npy_path, embedding)
    model.wv.save_word2vec_format(txt_path)

    meta = {
        "source_fasta": args.fasta,
        "num_records": int(n_records),
        "kmer": int(args.kmer),
        "seq_len": int(args.seq_len),
        "kmer_len": int(args.seq_len - args.kmer + 1),
        "embed_dim": int(args.embed_dim),
        "vocab_size_for_pasnet": int(vocab_size),
        "unk_idx": int(unk_idx),
        "unk_token": UNK_TOKEN,
        "unk_strategy": unk_strategy,
        "word2vec_vocab_size": int(len(model.wv)),
        "found_acgt_kmers": int(found),
        "missing_acgt_kmers": int(missing),
        "word2vec_params": {
            "sg": 1,
            "method": "skip-gram",
            "window": int(args.window),
            "epochs": int(args.epochs),
            "workers": int(args.workers),
            "negative": int(args.negative),
            "sample": float(args.sample),
            "min_count": int(args.min_count),
            "batch_words": int(args.batch_words),
            "seed": int(args.seed),
        },
        "output_embedding_npy": npy_path,
        "output_word2vec_txt": txt_path,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("PASNet Word2Vec embedding written:")
    print(f"  embedding_npy: {npy_path}")
    print(f"  word2vec_txt: {txt_path}")
    print(f"  meta: {meta_path}")
    print(f"  embedding shape: {embedding.shape}")
    print(f"  found ACGT kmers: {found}")
    print(f"  missing ACGT kmers: {missing}")
    print(f"  unk strategy: {unk_strategy}")


if __name__ == "__main__":
    main()
