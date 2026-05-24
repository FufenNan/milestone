"""
Download and tokenize FineWeb-Edu into nanoGPT-compatible uint16 .bin shards.

Examples:
    python data/fineweb.py --data-root data/fineweb_edu
    python data/fineweb.py --data-root data/fineweb_edu --shard-size 2048 --max-shards 2 --num-proc 1
"""

import argparse
import itertools
import multiprocessing as mp
import os
import pickle

import numpy as np


enc = None
eot = None


def parse_args():
    default_data_root = os.environ.get("NANOGPT_DATA_ROOT", "data/fineweb_edu")
    default_remote_name = os.environ.get("NANOGPT_FINEWEB_REMOTE_NAME", "sample-10BT")
    default_shard_size = int(os.environ.get("NANOGPT_FINEWEB_SHARD_SIZE", int(1e8)))
    default_num_proc = max(1, (os.cpu_count() or 2) // 2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "--output-dir", dest="data_root", default=default_data_root)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--remote-name", "--name", dest="remote_name", default=default_remote_name)
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-size", type=int, default=default_shard_size)
    parser.add_argument("--num-proc", "--num-procs", dest="num_proc", type=int, default=default_num_proc)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-docs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--val-shards", type=int, default=1)
    return parser.parse_args()


def repo_path(path):
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def init_tokenizer():
    global enc, eot
    if enc is None:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token


def tokenize(doc):
    init_tokenizer()
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint32)
    assert (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    tokens_np.tofile(filename)


def write_meta(data_root):
    init_tokenizer()
    meta = {
        "vocab_size": 50257,
        "tokenizer_vocab_size": enc.n_vocab,
        "eot_token": eot,
        "dtype": "uint16",
        "format": "raw_bin",
    }
    with open(os.path.join(data_root, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


def shard_split(shard_index, val_shards):
    return "val" if shard_index < val_shards else "train"


def main():
    args = parse_args()
    from datasets import load_dataset
    from tqdm import tqdm

    data_root = repo_path(args.data_root)
    os.makedirs(data_root, exist_ok=True)
    write_meta(data_root)

    fw = load_dataset(
        args.dataset,
        name=args.remote_name,
        split=args.split,
        streaming=args.streaming,
    )
    if args.max_docs:
        fw = itertools.islice(fw, args.max_docs)

    shard_index = 0
    token_count = 0
    progress_bar = None
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)

    with mp.Pool(args.num_proc, initializer=init_tokenizer) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            pos = 0
            while pos < len(tokens):
                if progress_bar is None:
                    split = shard_split(shard_index, args.val_shards)
                    progress_bar = tqdm(
                        total=args.shard_size,
                        unit="tokens",
                        desc=f"{split} shard {shard_index}",
                    )

                available = args.shard_size - token_count
                n = min(available, len(tokens) - pos)
                all_tokens_np[token_count : token_count + n] = tokens[pos : pos + n]
                token_count += n
                pos += n
                progress_bar.update(n)

                if token_count == args.shard_size:
                    split = shard_split(shard_index, args.val_shards)
                    filename = os.path.join(data_root, f"fineweb_{split}_{shard_index:06d}.bin")
                    write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    token_count = 0
                    progress_bar.close()
                    progress_bar = None

                    if args.max_shards and shard_index >= args.max_shards:
                        return

    if token_count:
        split = shard_split(shard_index, args.val_shards)
        filename = os.path.join(data_root, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
        if progress_bar is not None:
            progress_bar.close()


if __name__ == "__main__":
    main()
