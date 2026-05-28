"""
Download and tokenize the blended training corpora into per-source uint16 shards.

Default sources:
    fineweb      HuggingFaceFW/fineweb-edu, sample-10BT, text
    wikipedia   wikimedia/wikipedia, 20231101.en, text
    arxiv        armanc/scientific_papers, arxiv, article
    pubmed       armanc/scientific_papers, pubmed, article
    books        incredible45/Gutenberg-BookCorpus-Cleaned-Data-English
    pg19         emozilla/pg19, text

Example:
    python data/blend.py --data-root data/blend --streaming --num-proc 4
    python data/blend.py --sources fineweb wikipedia --max-shards-per-source 2
"""

import argparse
import itertools
import multiprocessing as mp
import os
import pickle
import re

import numpy as np


enc = None
eot = None


DEFAULT_SOURCES = {
    "fineweb": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "output_dir": "fineweb_edu",
        "filename_prefix": "fineweb",
        "text_fields": ("text",),
    },
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "output_dir": "wikipedia",
        "filename_prefix": "wikipedia",
        "text_fields": ("text",),
    },
    "arxiv": {
        "dataset": "armanc/scientific_papers",
        "config": "arxiv",
        "split": "train",
        "output_dir": "papers_arxiv",
        "filename_prefix": "papers_arxiv",
        "text_fields": ("article", "abstract"),
    },
    "pubmed": {
        "dataset": "armanc/scientific_papers",
        "config": "pubmed",
        "split": "train",
        "output_dir": "papers_pubmed",
        "filename_prefix": "papers_pubmed",
        "text_fields": ("article", "abstract"),
    },
    "books": {
        "dataset": "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English",
        "config": None,
        "split": "train",
        "output_dir": "books",
        "filename_prefix": "books",
        "text_fields": ("context", "text", "content", "book", "body"),
    },
    "pg19": {
        "dataset": "emozilla/pg19",
        "config": None,
        "split": "train",
        "output_dir": "pg19",
        "filename_prefix": "pg19",
        "text_fields": ("text",),
    },
}


def parse_args():
    default_data_root = os.environ.get("NANOGPT_BLEND_DATA_ROOT", "data/blend")
    default_shard_size = int(os.environ.get("NANOGPT_BLEND_SHARD_SIZE", int(1e8)))
    default_num_proc = max(1, (os.cpu_count() or 2) // 2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "--output-dir", dest="data_root", default=default_data_root)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=tuple(DEFAULT_SOURCES),
        default=tuple(DEFAULT_SOURCES),
        help="Sources to prepare. Defaults to all blend sources.",
    )
    parser.add_argument("--shard-size", type=int, default=default_shard_size)
    parser.add_argument("--num-proc", "--num-procs", dest="num_proc", type=int, default=default_num_proc)
    parser.add_argument(
        "--max-shards-per-source",
        type=int,
        default=0,
        help="Maximum shards to write in this run. 0 means no limit.",
    )
    parser.add_argument(
        "--target-shards-per-source",
        type=int,
        default=0,
        help="Total train shards desired per source. With --append-existing, writes only the missing shards.",
    )
    parser.add_argument("--max-docs-per-source", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--skip-docs-per-source",
        type=int,
        default=0,
        help="Skip this many text examples before tokenization. Useful when you recorded a previous cursor.",
    )
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Start new shard numbers after existing train shards instead of overwriting from 000000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing shard files. By default existing files are protected.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle dataset order before tokenization. Helpful for appending more samples without a saved cursor.",
    )
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--wikipedia-name", default="20231101.en")
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


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(normalize_text(item) for item in value)
    return str(value)


def pick_text(doc, text_fields):
    for field in text_fields:
        text = normalize_text(doc.get(field))
        if text.strip():
            return text
    for value in doc.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def iter_texts(dataset_iter, text_fields):
    for doc in dataset_iter:
        text = pick_text(doc, text_fields)
        if text.strip():
            yield text


def tokenize_text(text):
    init_tokenizer()
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens, dtype=np.uint32)
    assert (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np, overwrite=False):
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing shard: {filename}")
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


def load_hf_dataset(spec, streaming):
    from datasets import load_dataset

    kwargs = {"split": spec["split"], "streaming": streaming}
    if spec["config"] is not None:
        kwargs["name"] = spec["config"]
    return load_dataset(spec["dataset"], **kwargs)


def existing_train_indices(output_dir, filename_prefix):
    pattern = re.compile(rf"^{re.escape(filename_prefix)}_train_(\d{{6}})\.(bin|npy)$")
    indices = []
    if not os.path.isdir(output_dir):
        return indices
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def maybe_shuffle_dataset(ds, source_name, args):
    if args.shuffle_seed is None:
        return ds
    seed = args.shuffle_seed + sum(ord(ch) for ch in source_name)
    try:
        return ds.shuffle(seed=seed, buffer_size=args.shuffle_buffer_size)
    except TypeError:
        return ds.shuffle(seed=seed)


def prepare_source(source_name, spec, args):
    from tqdm import tqdm

    output_dir = repo_path(os.path.join(args.data_root, spec["output_dir"]))
    os.makedirs(output_dir, exist_ok=True)
    write_meta(output_dir)

    existing_indices = existing_train_indices(output_dir, spec["filename_prefix"])
    if args.append_existing and existing_indices:
        shard_index = max(existing_indices) + 1
        print(
            f"Appending {source_name}: found {len(existing_indices)} existing train shards; "
            f"next shard index is {shard_index:06d}"
        )
    else:
        shard_index = 0

    if args.target_shards_per_source and args.append_existing:
        missing = args.target_shards_per_source - len(existing_indices)
        if missing <= 0:
            print(
                f"Skipping {source_name}: already has {len(existing_indices)} train shards, "
                f"target is {args.target_shards_per_source}"
            )
            return
        max_new_shards = missing
    elif args.target_shards_per_source:
        max_new_shards = args.target_shards_per_source
    else:
        max_new_shards = args.max_shards_per_source

    print(f"Preparing {source_name}: {spec['dataset']} ({spec['config'] or 'default'})")
    ds = load_hf_dataset(spec, args.streaming)
    ds = maybe_shuffle_dataset(ds, source_name, args)
    texts = iter_texts(ds, spec["text_fields"])
    if args.skip_docs_per_source:
        texts = itertools.islice(texts, args.skip_docs_per_source, None)
    if args.max_docs_per_source:
        texts = itertools.islice(texts, args.max_docs_per_source)

    shards_written = 0
    token_count = 0
    progress_bar = None
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)

    with mp.Pool(args.num_proc, initializer=init_tokenizer) as pool:
        for tokens in pool.imap(tokenize_text, texts, chunksize=16):
            pos = 0
            while pos < len(tokens):
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=args.shard_size,
                        unit="tokens",
                        desc=f"{source_name} train shard {shard_index}",
                    )

                available = args.shard_size - token_count
                n = min(available, len(tokens) - pos)
                all_tokens_np[token_count : token_count + n] = tokens[pos : pos + n]
                token_count += n
                pos += n
                progress_bar.update(n)

                if token_count == args.shard_size:
                    filename = os.path.join(
                        output_dir,
                        f"{spec['filename_prefix']}_train_{shard_index:06d}.bin",
                    )
                    write_datafile(filename, all_tokens_np, overwrite=args.overwrite)
                    shard_index += 1
                    shards_written += 1
                    token_count = 0
                    progress_bar.close()
                    progress_bar = None

                    if max_new_shards and shards_written >= max_new_shards:
                        return

    if token_count:
        filename = os.path.join(output_dir, f"{spec['filename_prefix']}_train_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count], overwrite=args.overwrite)
        if progress_bar is not None:
            progress_bar.close()


def main():
    args = parse_args()
    for source_name in args.sources:
        spec = dict(DEFAULT_SOURCES[source_name])
        if source_name == "wikipedia":
            spec["config"] = args.wikipedia_name
        prepare_source(source_name, spec, args)


if __name__ == "__main__":
    main()
