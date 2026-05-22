from pathlib import Path

from data_utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--item_tower", type=str, default="id")
    parser.add_argument("--root_data_dir", type=str, default="../",)
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--llama_path", type=str, default="./models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--llm2vec_path", type=str, default="./models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
    parser.add_argument("--llm2vec_output", type=str, default="./dataset/Amazon_Prime_Pantry_llm2vec.pt")
    parser.add_argument("--dataset", type=str, default='MIND-small')
    parser.add_argument("--behaviors", type=str, default='behaviors_l5_tr_v.tsv')
    parser.add_argument("--cold_file", type=str, default='None')
    parser.add_argument("--new_file", type=str, default='None')
    parser.add_argument("--news", type=str, default='news_l5_tr_v.tsv')

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--target_global_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--auto_adjust_batch_size", type=str2bool, default=False)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--l2_weight", type=float, default=0)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--dnn_layers", type=int, default=0)
    parser.add_argument("--mo_dnn_layers", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--gamma2", type=float, default=0.2)
    parser.add_argument("--gamma3", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--early_stop_mode", type=str, default="greater", choices=["greater"])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_k", type=int, default=0)
    parser.add_argument("--freeze", type=int, default=1)
    parser.add_argument("--left", type=int, default=-3)
    parser.add_argument("--right", type=int, default=2)
    parser.add_argument("--n_h", type=int, default=2)
    parser.add_argument("--n_v", type=int, default=2) 
    # ============== model parameters ==============
    parser.add_argument("--bert_model_load", type=str, default='bert-base-uncased')
    parser.add_argument("--freeze_paras_before", type=int, default=165)
    parser.add_argument("--word_embedding_dim", type=int, default=768)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--min_seq_len", type=int, default=5)

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--load_ckpt_name", type=str, default='best.pt')
    parser.add_argument("--save_best_name", type=str, default='best.pt')
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--phase1_ckpt_dir", type=str, default=None)
    parser.add_argument("--phase2_ckpt_dir", type=str, default=None)
    parser.add_argument("--label_screen", type=str, default='None')
    parser.add_argument("--logging_num", type=int, default=8)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", default=-1, type=int)

    # ============== news information==============
    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--news_attributes", type=str, default='title')

    args = parser.parse_args()
    args.news_attributes = args.news_attributes.split(',')
    if args.batch_size is not None:
        args.per_device_batch_size = args.batch_size
    args.batch_size = args.per_device_batch_size

    for path_attr in (
        "data_dir",
        "model_dir",
        "output_dir",
        "checkpoint_dir",
        "llama_path",
        "llm2vec_path",
        "llm2vec_output",
        "phase1_ckpt_dir",
        "phase2_ckpt_dir",
    ):
        value = getattr(args, path_attr)
        if value is not None:
            setattr(args, path_attr, Path(value))

    return args


if __name__ == "__main__":
    args = parse_args()
