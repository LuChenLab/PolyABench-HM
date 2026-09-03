#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


SS_TO_ID = {
    "U": 0,
    "E": 1,
    "L": 2,
    "R": 3,
    "H": 4,
    "M": 5,
    "I": 6,
}


def read_fasta(path):
    header = None
    chunks = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)

        if header is not None:
            yield header, "".join(chunks)


def split_fasta_round_robin(input_fasta, chunk_dir, num_chunks, seq_len):
    chunk_dir.mkdir(parents=True, exist_ok=True)

    handles = []
    chunk_paths = []

    for i in range(num_chunks):
        p = chunk_dir / f"chunk_{i:03d}.fasta"
        chunk_paths.append(p)
        handles.append(open(p, "w"))

    original_headers = []
    n = 0

    try:
        for header, seq in read_fasta(input_fasta):
            seq = seq.upper()

            if len(seq) != seq_len:
                raise ValueError(f"{header}: length={len(seq)}, expected={seq_len}")

            chunk_id = n % num_chunks

            handles[chunk_id].write(f">{header}\n")
            handles[chunk_id].write(seq + "\n")

            original_headers.append(header)
            n += 1

    finally:
        for h in handles:
            h.close()

    valid_chunk_paths = [p for p in chunk_paths if p.exists() and p.stat().st_size > 0]

    return valid_chunk_paths, original_headers


def write_rna_fasta(input_fasta, output_fasta, seq_len):
    headers = []
    n = 0

    with open(output_fasta, "w") as out:
        for header, seq in read_fasta(input_fasta):
            seq = seq.upper()

            if len(seq) != seq_len:
                raise ValueError(f"{header}: length={len(seq)}, expected={seq_len}")

            rna_seq = seq.replace("T", "U")

            out.write(f">{header}\n")
            out.write(rna_seq + "\n")

            headers.append(header)
            n += 1

    return headers, n


def is_range_line(line):
    parts = line.strip().split()
    return len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit()


def is_sequence_line(line):
    s = line.strip().upper()
    return len(s) > 0 and set(s).issubset(set("ACGUN"))


def is_structure_line(line):
    s = line.strip()
    return len(s) > 0 and set(s).issubset(set(".()"))


def combine_substructure(
    raw_path,
    combined_path,
    first_start,
    first_end,
    second_start,
    second_end,
    max_structures=3,
):
    """
    Streaming version.

    Original DeepPASTA logic:
      first half structures x second half structures
      then keep top N.

    Here we directly write only the first max_structures combinations to avoid
    storing huge intermediate lists in memory.
    """
    header = None
    current_part = None
    first_structs = []
    second_structs = []

    def write_record(out):
        nonlocal header, first_structs, second_structs

        if header is None:
            return

        if len(first_structs) == 0 or len(second_structs) == 0:
            raise ValueError(
                f"{header}: missing structures. "
                f"first={len(first_structs)}, second={len(second_structs)}"
            )

        out.write(header + "\n")

        written = 0
        for first in first_structs:
            for second in second_structs:
                out.write(first + second + "\n")
                written += 1
                if written >= max_structures:
                    return

    with open(raw_path) as f, open(combined_path, "w") as out:
        for raw in f:
            line = raw.rstrip("\n")

            if not line.strip():
                continue

            if line.startswith(">"):
                write_record(out)

                header = line.strip()
                current_part = None
                first_structs = []
                second_structs = []
                continue

            if is_range_line(line):
                parts = line.strip().split()
                start = int(parts[0])
                end = int(parts[1])

                if start == first_start and end == first_end:
                    current_part = "first"
                elif start == second_start and end == second_end:
                    current_part = "second"
                else:
                    current_part = None

                continue

            if is_sequence_line(line):
                continue

            if is_structure_line(line):
                s = line.strip()

                if current_part == "first":
                    first_structs.append(s)
                elif current_part == "second":
                    second_structs.append(s)

        write_record(out)


def replace_supporting_character(final_shape):
    return ["L" if x == "<" else "R" if x == ">" else x for x in final_shape]


def update_for_i_and_m(final_shape, pattern):
    n = len(final_shape)
    first_index = -1

    for i in range(n - 1):
        if final_shape[i] == "(" and final_shape[i + 1] != "(":
            first_index = i + 1

        elif final_shape[i] in [".", ">"] and final_shape[i + 1] == ")" and first_index != -1:
            last_index = i
            segment = "".join(final_shape[first_index:last_index])

            if len(re.findall(pattern, segment)) == 1:
                mark = "I"
            elif len(re.findall(pattern, segment)) > 1:
                mark = "M"
            else:
                first_index = -1
                continue

            for l in range(first_index, last_index + 1):
                if final_shape[l] == "<":
                    final_shape[l] = "L"
                elif final_shape[l] == ">":
                    final_shape[l] = "R"
                elif final_shape[l] == ".":
                    final_shape[l] = mark

            l = first_index - 1
            m = last_index + 1

            while l >= 0 and m < n and final_shape[l] == "(" and final_shape[m] == ")":
                final_shape[l] = "<"
                final_shape[m] = ">"
                l -= 1
                m += 1

            first_index = -1

        elif final_shape[i] == "(" and final_shape[i + 1] == "(":
            first_index = -1

    return final_shape


def shape_at_each_position(dot):
    dot = dot.strip()
    n = len(dot)
    final_shape = list(dot)

    last_parenthesis_index = 0
    for i, ch in enumerate(dot):
        if ch == ")":
            last_parenthesis_index = i

    before_or_after = False
    last_bracket = ""
    first_index = 0

    for i in range(n - 1):
        if not before_or_after and dot[i] == "(":
            before_or_after = True

        if before_or_after and i > last_parenthesis_index:
            before_or_after = False

        if dot[i] == "." and not before_or_after:
            final_shape[i] = "E"

        elif dot[i] in ["(", ")"] and dot[i + 1] == ".":
            last_bracket = dot[i]
            first_index = i + 1

        elif dot[i] == "." and dot[i + 1] in [")", "("]:
            last_index = i

            if last_bracket == "(" and dot[i + 1] == ")":
                for l in range(first_index, last_index + 1):
                    final_shape[l] = "H"

                l = first_index - 1
                m = last_index + 1

                while l >= 0 and m < n and dot[l] == "(" and dot[m] == ")":
                    final_shape[l] = "<"
                    final_shape[m] = ">"
                    l -= 1
                    m += 1

            last_bracket = ""

    final_shape = update_for_i_and_m(final_shape, r"<+\w+>+")

    count = 0
    while "(" in final_shape or ")" in final_shape:
        new_shape = update_for_i_and_m(final_shape, r"<+\w*>+")

        if new_shape == final_shape:
            count += 1
            if count >= 30:
                break

        final_shape = new_shape

    final_shape = replace_supporting_character(final_shape)

    final_shape = [
        "U" if x in [".", "(", ")", "*"] else x
        for x in final_shape
    ]

    return "".join(final_shape)


def make_shape_file(combined_path, shape_path, seq_len, n_structures):
    """
    Streaming version.
    Writes shape file record-by-record.
    """
    header = None
    structs = []

    def write_record(out):
        nonlocal header, structs

        if header is None:
            return

        if len(structs) == 0:
            raise ValueError(f"{header}: no structures found")

        selected = structs[:n_structures]

        while len(selected) < n_structures:
            selected.append(selected[-1])

        out.write(header + "\n")

        for s in selected:
            if len(s) != seq_len:
                raise ValueError(f"{header}: structure length={len(s)}, expected={seq_len}")

            out.write(shape_at_each_position(s) + "\n")

    with open(combined_path) as f, open(shape_path, "w") as out:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                write_record(out)

                header = line
                structs = []
            else:
                if not set(line).issubset(set(".()")):
                    raise ValueError(f"{header}: invalid dot-bracket line")
                structs.append(line)

        write_record(out)


def read_shape_file_to_cache(shape_path, ss_path, headers_path, seq_len, n_structures):
    """
    Two-pass reading:
      pass 1: count records
      pass 2: write memmap npy

    Avoids holding Python list of all structure arrays in memory.
    """
    n = 0
    with open(shape_path) as f:
        for line in f:
            if line.startswith(">"):
                n += 1

    ss = np.lib.format.open_memmap(
        ss_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n, n_structures, seq_len),
    )

    headers = []

    header = None
    structs = []
    idx = 0

    def write_record():
        nonlocal header, structs, idx

        if header is None:
            return

        if len(structs) != n_structures:
            raise ValueError(f"{header}: got {len(structs)} structures, expected={n_structures}")

        arr = np.zeros((n_structures, seq_len), dtype=np.uint8)

        for i, ss_line in enumerate(structs):
            if len(ss_line) != seq_len:
                raise ValueError(f"{header}: structure length={len(ss_line)}, expected={seq_len}")

            bad = set(ss_line) - set(SS_TO_ID)
            if bad:
                raise ValueError(f"{header}: invalid chars: {bad}")

            arr[i] = np.asarray([SS_TO_ID[x] for x in ss_line], dtype=np.uint8)

        clean_header = header[1:] if header.startswith(">") else header
        headers.append(clean_header)
        ss[idx] = arr
        idx += 1

    with open(shape_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                write_record()
                header = line
                structs = []
            else:
                structs.append(line)

        write_record()

    headers = np.asarray(headers, dtype=object)
    np.save(headers_path, headers)

    if idx != n:
        raise ValueError(f"Written records={idx}, counted={n}")

    return headers, ss.shape


def process_one_fasta(
    fasta,
    out_prefix,
    tmp_dir,
    rnashapes_bin,
    seq_len,
    half_len,
    n_structures,
    rnashapes_c,
    rnashapes_t,
):
    out_prefix = Path(out_prefix)
    tmp_dir = Path(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    stem = out_prefix.name

    rna_fasta = tmp_dir / f"{stem}.rna.fa"
    raw_txt = tmp_dir / f"{stem}.rnashapes.raw.txt"
    combined_txt = tmp_dir / f"{stem}.combined.txt"
    shape_txt = tmp_dir / f"{stem}.shape.txt"
    log_txt = tmp_dir / f"{stem}.rnashapes.log"

    headers_from_fasta, n = write_rna_fasta(fasta, rna_fasta, seq_len)

    cmd = [
        str(rnashapes_bin),
        "-f", str(rna_fasta),
        "-s",
        "-c", str(rnashapes_c),
        "-t", str(rnashapes_t),
        "-w", str(half_len),
        "-W", str(half_len),
        "-O", r"D{%s\n}",
    ]

    with open(raw_txt, "w") as raw_out, open(log_txt, "w") as log:
        log.write("[CMD] " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            cmd,
            stdout=raw_out,
            stderr=log,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"RNAshapes failed for {fasta}. Check log: {log_txt}")

    combine_substructure(
        raw_path=raw_txt,
        combined_path=combined_txt,
        first_start=1,
        first_end=half_len,
        second_start=half_len + 1,
        second_end=seq_len,
        max_structures=n_structures,
    )

    make_shape_file(
        combined_path=combined_txt,
        shape_path=shape_txt,
        seq_len=seq_len,
        n_structures=n_structures,
    )

    ss_path = str(out_prefix) + ".ss.uint8.npy"
    ss_headers_path = str(out_prefix) + ".ss.headers.npy"
    ss_meta_path = str(out_prefix) + ".ss.meta.json"

    headers, ss_shape = read_shape_file_to_cache(
        shape_path=shape_txt,
        ss_path=ss_path,
        headers_path=ss_headers_path,
        seq_len=seq_len,
        n_structures=n_structures,
    )

    if list(headers) != headers_from_fasta:
        raise ValueError(f"Header order mismatch in {fasta}")

    meta = {
        "num_samples": int(ss_shape[0]),
        "n_structures": int(ss_shape[1]),
        "seq_len": int(ss_shape[2]),
        "structure_vocab_size": 7,
        "structure_encoding": "U=0,E=1,L=2,R=3,H=4,M=5,I=6",
        "source_fasta": str(fasta),
        "rna_fasta": str(rna_fasta),
        "rnashapes_raw": str(raw_txt),
        "combined_dotbracket": str(combined_txt),
        "shape_file": str(shape_txt),
        "rnashapes_log": str(log_txt),
        "rnashapes_bin": str(rnashapes_bin),
    }

    with open(ss_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "ss_path": ss_path,
        "headers_path": ss_headers_path,
        "meta_path": ss_meta_path,
        "num_samples": int(ss_shape[0]),
        "shape_file": str(shape_txt),
        "log": str(log_txt),
    }


def worker_process(job):
    res = process_one_fasta(
        fasta=job["fasta"],
        out_prefix=job["out_prefix"],
        tmp_dir=job["tmp_dir"],
        rnashapes_bin=job["rnashapes_bin"],
        seq_len=job["seq_len"],
        half_len=job["half_len"],
        n_structures=job["n_structures"],
        rnashapes_c=job["rnashapes_c"],
        rnashapes_t=job["rnashapes_t"],
    )

    res["chunk_id"] = job["chunk_id"]
    return res


def merge_and_reorder_results(results, original_headers, out_prefix, args, tmp_root):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if len(set(original_headers)) != len(original_headers):
        raise ValueError("Duplicate FASTA headers found. Cannot safely reorder by header.")

    total_n = len(original_headers)
    final_ss_path = str(out_prefix) + ".ss.uint8.npy"
    final_headers_path = str(out_prefix) + ".ss.headers.npy"
    final_meta_path = str(out_prefix) + ".ss.meta.json"

    header_to_chunk_pos = {}

    total_from_chunks = 0

    for res in results:
        headers = np.load(res["headers_path"], allow_pickle=True)
        total_from_chunks += len(headers)

        for i, h in enumerate(headers):
            h = str(h)
            if h in header_to_chunk_pos:
                raise ValueError(f"Duplicate header after merging: {h}")
            header_to_chunk_pos[h] = (res["ss_path"], i)

    if total_from_chunks != total_n:
        raise ValueError(f"Chunk total={total_from_chunks}, expected={total_n}")

    final_ss = np.lib.format.open_memmap(
        final_ss_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_n, args.n_structures, args.seq_len),
    )

    cache_handles = {}
    missing = []

    for out_i, h in enumerate(original_headers):
        if h not in header_to_chunk_pos:
            missing.append(h)
            continue

        ss_path, local_i = header_to_chunk_pos[h]

        if ss_path not in cache_handles:
            cache_handles[ss_path] = np.load(ss_path, mmap_mode="r")

        final_ss[out_i] = cache_handles[ss_path][local_i]

        if (out_i + 1) % 200000 == 0:
            print(f"  reordered {out_i + 1}/{total_n}", flush=True)

    if missing:
        raise ValueError(f"Missing {len(missing)} headers. Example: {missing[:3]}")

    ordered_headers = np.asarray(original_headers, dtype=object)
    np.save(final_headers_path, ordered_headers)

    meta = {
        "num_samples": int(total_n),
        "n_structures": int(args.n_structures),
        "seq_len": int(args.seq_len),
        "structure_vocab_size": 7,
        "structure_encoding": "U=0,E=1,L=2,R=3,H=4,M=5,I=6",
        "source_fasta": args.fasta,
        "rnashapes_bin": args.rnashapes_bin,
        "num_chunks": args.num_chunks,
        "num_workers": args.num_workers,
        "tmp_root": str(tmp_root),
        "order": "reordered_to_original_fasta_order",
        "first_header": str(ordered_headers[0]),
        "last_header": str(ordered_headers[-1]),
        "memory_optimized": True,
        "combined_and_shape_streaming": True,
        "final_merge_streaming": True,
    }

    with open(final_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Final ordered DeepPASTA structure cache written:")
    print(f"  ss_path: {final_ss_path}")
    print(f"  ss_headers_path: {final_headers_path}")
    print(f"  ss_meta_path: {final_meta_path}")
    print(f"  ss shape: {(total_n, args.n_structures, args.seq_len)}")
    print("  order: original FASTA order")


def run_parallel(args):
    out_prefix = Path(args.out_prefix)
    tmp_root = Path(args.tmp_dir) / f"{out_prefix.name}_parallel"

    if tmp_root.exists() and not args.resume:
        shutil.rmtree(tmp_root)

    chunk_fasta_dir = tmp_root / "chunk_fastas"
    chunk_cache_dir = tmp_root / "chunk_caches"
    chunk_tmp_dir = tmp_root / "chunk_tmp"

    chunk_fasta_dir.mkdir(parents=True, exist_ok=True)
    chunk_cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and chunk_fasta_dir.exists() and any(chunk_fasta_dir.glob("chunk_*.fasta")):
        print("Resume mode: using existing chunk FASTA files.")
        chunk_paths = sorted(chunk_fasta_dir.glob("chunk_*.fasta"))

        original_headers = []
        for header, _ in read_fasta(args.fasta):
            original_headers.append(header)

    else:
        chunk_paths, original_headers = split_fasta_round_robin(
            input_fasta=args.fasta,
            chunk_dir=chunk_fasta_dir,
            num_chunks=args.num_chunks,
            seq_len=args.seq_len,
        )

    if len(set(original_headers)) != len(original_headers):
        raise ValueError("Duplicate FASTA headers found. Cannot safely reorder by header.")

    jobs = []
    finished_results = []

    for i, chunk_fasta in enumerate(chunk_paths):
        chunk_out_prefix = chunk_cache_dir / f"chunk_{i:03d}"
        ss_path = str(chunk_out_prefix) + ".ss.uint8.npy"
        header_path = str(chunk_out_prefix) + ".ss.headers.npy"
        meta_path = str(chunk_out_prefix) + ".ss.meta.json"

        if args.resume and Path(ss_path).exists() and Path(header_path).exists() and Path(meta_path).exists():
            try:
                headers = np.load(header_path, allow_pickle=True)
                ss = np.load(ss_path, mmap_mode="r")
                if ss.shape[0] == len(headers):
                    finished_results.append({
                        "chunk_id": i,
                        "ss_path": ss_path,
                        "headers_path": header_path,
                        "meta_path": meta_path,
                        "num_samples": int(ss.shape[0]),
                    })
                    continue
            except Exception:
                pass

        jobs.append({
            "chunk_id": i,
            "fasta": str(chunk_fasta),
            "out_prefix": str(chunk_out_prefix),
            "tmp_dir": str(chunk_tmp_dir / f"chunk_{i:03d}"),
            "rnashapes_bin": args.rnashapes_bin,
            "seq_len": args.seq_len,
            "half_len": args.half_len,
            "n_structures": args.n_structures,
            "rnashapes_c": args.rnashapes_c,
            "rnashapes_t": args.rnashapes_t,
        })

    print(f"Total records: {len(original_headers)}")
    print(f"Chunks: {len(chunk_paths)}")
    print(f"Workers: {args.num_workers}")
    print(f"Finished chunks reused: {len(finished_results)}")
    print(f"Chunks to run: {len(jobs)}")

    results = list(finished_results)

    if jobs:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(worker_process, job) for job in jobs]

            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    print("ERROR: one worker failed.", flush=True)
                    print(repr(e), flush=True)
                    raise

                print(f"Finished chunk {res['chunk_id']} | n={res['num_samples']}", flush=True)
                results.append(res)

    results = sorted(results, key=lambda x: x["chunk_id"])

    if len(results) != len(chunk_paths):
        raise ValueError(f"Finished chunks={len(results)}, expected={len(chunk_paths)}")

    merge_and_reorder_results(
        results=results,
        original_headers=original_headers,
        out_prefix=out_prefix,
        args=args,
        tmp_root=tmp_root,
    )

    if not args.keep_tmp:
        shutil.rmtree(tmp_root)
        print(f"Removed tmp: {tmp_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Build DeepPASTA-style RNA structure cache with RNAshapes."
    )

    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--tmp_dir", required=True)
    parser.add_argument("--rnashapes_bin", required=True)

    parser.add_argument("--seq_len", type=int, default=240)
    parser.add_argument("--half_len", type=int, default=120)
    parser.add_argument("--n_structures", type=int, default=3)

    parser.add_argument("--rnashapes_c", type=int, default=5)
    parser.add_argument("--rnashapes_t", type=int, default=1)

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.seq_len != args.half_len * 2:
        raise ValueError(
            f"This script currently expects seq_len = 2 * half_len. "
            f"Got seq_len={args.seq_len}, half_len={args.half_len}"
        )

    if args.num_workers > 1 or args.num_chunks > 1:
        run_parallel(args)
    else:
        res = process_one_fasta(
            fasta=args.fasta,
            out_prefix=args.out_prefix,
            tmp_dir=args.tmp_dir,
            rnashapes_bin=args.rnashapes_bin,
            seq_len=args.seq_len,
            half_len=args.half_len,
            n_structures=args.n_structures,
            rnashapes_c=args.rnashapes_c,
            rnashapes_t=args.rnashapes_t,
        )

        print("DeepPASTA structure cache written:")
        print(f"  ss_path: {res['ss_path']}")
        print(f"  ss_headers_path: {res['headers_path']}")
        print(f"  ss_meta_path: {res['meta_path']}")
        print(f"  num_samples: {res['num_samples']}")


if __name__ == "__main__":
    main()
