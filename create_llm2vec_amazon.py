import os
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("./dataset"))
    parser.add_argument("--model_dir", type=Path, default=Path("./models"))
    parser.add_argument("--llama_path", type=Path, default=Path("./models/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--llm2vec_path", type=Path, default=Path("./models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"))
    parser.add_argument("--llm2vec_output", type=Path, default=Path("./dataset/Amazon_Prime_Pantry_llm2vec.pt"))
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def require_path(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def main():
    args = parse_args()
    from llm2vec import LLM2Vec

    metadata_path = require_path(args.data_dir / "metadata_Prime_Pantry.csv", "Prime Pantry metadata")
    require_path(args.llama_path, "Llama model directory")
    require_path(args.llm2vec_path, "LLM2Vec PEFT directory")

    l2v = LLM2Vec.from_pretrained(
        str(args.llama_path),
        peft_model_name_or_path=str(args.llm2vec_path),
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    metadata = pd.read_csv(metadata_path)
    if "title" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain a 'title' column")

    print(f"metadata_count={len(metadata)}")
    item_word_embs = [torch.zeros(4096)]
    titles = metadata["title"].fillna("").astype(str).tolist()
    for start in tqdm(range(0, len(titles), args.batch_size)):
        batch = titles[start:start + args.batch_size]
        item_feature = l2v.encode(batch)
        item_word_embs.extend(item_feature)

    output = torch.stack(tensors=item_word_embs, dim=0)
    if output.shape != (len(metadata) + 1, 4096):
        raise ValueError(f"Expected embedding shape {(len(metadata) + 1, 4096)}, got {tuple(output.shape)}")
    args.llm2vec_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.llm2vec_output)
    print(f"embedding_shape={tuple(output.shape)}")
    print(f"saved_path={args.llm2vec_output}")


if __name__ == "__main__":
    main()
