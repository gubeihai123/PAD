import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
from llm2vec import LLM2Vec
import torch

ROOT = Path(__file__).resolve().parent


def main():
    l2v = LLM2Vec.from_pretrained(
        str(ROOT / "models" / "Meta-Llama-3-8B-Instruct"),
        peft_model_name_or_path=str(ROOT / "models" / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"),
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    d = pd.read_csv(ROOT / 'dataset' / 'metadata_Prime_Pantry.csv')

    item_word_embs = [torch.zeros(4096)]
    for i in tqdm(range(8)):
        strlist = d.iloc[1000 * i:1000 * (i + 1), 1].tolist()
        item_feature = l2v.encode(strlist)
        item_word_embs.extend(item_feature)

    strlist = d.iloc[8000:, 1].tolist()
    item_feature = l2v.encode(strlist)
    item_word_embs.extend(item_feature)

    a = torch.stack(tensors=item_word_embs, dim=0)
    torch.save(a, ROOT / 'dataset' / 'Amazon_Prime_Pantry_llm2vec.pt')


if __name__ == "__main__":
    main()
