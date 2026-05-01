"""
Download and tokenize FineWeb-Edu into GPT-2-tokenized numpy shards.

Example:
    python data/fineweb.py --output-dir data/edu_fineweb10B
"""

import argparse
import multiprocessing as mp
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/edu_fineweb10B")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-size", type=int, default=int(1e8))
    parser.add_argument("--num-procs", type=int, default=max(1, os.cpu_count() // 2))
    return parser.parse_args()


enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset, name=args.name, split=args.split)
    with mp.Pool(args.num_procs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            if token_count + len(tokens) < args.shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
                continue

            split_name = "val" if shard_index == 0 else "train"
            filename = os.path.join(output_dir, f"edufineweb_{split_name}_{shard_index:06d}.npy")
            remainder = args.shard_size - token_count
            if progress_bar is not None:
                progress_bar.update(remainder)
                progress_bar.close()
            all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

        if token_count != 0:
            split_name = "val" if shard_index == 0 else "train"
            filename = os.path.join(output_dir, f"edufineweb_{split_name}_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
