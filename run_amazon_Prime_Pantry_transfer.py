import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, \
    RobertaTokenizer, RobertaModel, RobertaConfig, \
    DebertaTokenizer, DebertaModel, DebertaConfig, \
    DistilBertTokenizer, DistilBertModel, DistilBertConfig, \
    GPT2Tokenizer, OPTModel, OPTConfig

from parameters import parse_args
import sys
if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parse_args()
    sys.exit(0)
from model.model_mmd_ada import Model2_transfer, Bert_Encoder
from data_utils import read_news, read_news_bert, get_doc_input_bert, get_id_embeddings,\
    read_behaviors, BuildTrainDataset, eval_model_amazon, eval_model, eval_model_step2, get_item_embeddings, get_item_embeddings_llm, get_item_word_embs, get_item_word_embs_llm,get_item_embeddings_llm_4
from data_utils import read_news_bert_amazon_pantry, read_behaviors_amazon_pantry
from data_utils.utils import *
from data_utils.repro import (
    append_metrics,
    configure_batch_size,
    load_checkpoint_into_model,
    pantry_paths,
    prepare_run_dirs,
    require_file,
    resolve_load_checkpoint,
    save_args_json,
    save_best_summary,
    save_checkpoint,
    seed_worker,
    set_seed,
    validate_item_embeddings,
)
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_
import gc
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(args, use_modal, local_rank):
    if use_modal:
        if 'roberta' in args.bert_model_load:
            Log_file.info('load roberta model...')
            bert_model_load = '../../pretrained_models/' + args.bert_model_load
            tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
            config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
            if 'base' in args.bert_model_load:
                args.word_embedding_dim = 768
            if 'large' in args.bert_model_load:
                args.word_embedding_dim = 1024
        elif 'opt' in args.bert_model_load:
            Log_file.info('load opt model...')
            bert_model_load = '../../pretrained_models/' + args.bert_model_load
            tokenizer = GPT2Tokenizer.from_pretrained(bert_model_load)
            config = OPTConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = OPTModel.from_pretrained(bert_model_load, config=config)
        elif 'llm' in args.bert_model_load:
            Log_file.info('load llm2vec...')
            args.word_embedding_dim = 4096

        Log_file.info('read news...')
        before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name  = read_news_bert_amazon_pantry(
            os.path.join(args.root_data_dir, args.dataset, args.news), args)
        
        Log_file.info('read behaviors...')
        item_num, item_id_to_dic, users_train, users_valid, users_test, \
        users_history_for_valid, users_history_for_test, item_name_to_id = \
            read_behaviors_amazon_pantry(pantry_paths(args)["history"], before_item_id_to_dic,
                           before_item_name_to_id, before_item_id_to_name,
                           args.max_seq_len, args.min_seq_len, Log_file)
        Log_file.info('Finish reading behaviors')


        llm2vec_path = require_file(args.llm2vec_output, "Prime Pantry LLM2Vec embedding")
        item_word_embs =torch.load(llm2vec_path)
        item_word_embs=torch.tensor(item_word_embs,dtype=torch.float32)
        validate_item_embeddings(item_word_embs, item_num, llm2vec_path)

        Log_file.info('Finish reading item embeddings')


    Log_file.info('build dataset...')

    train_dataset = BuildTrainDataset(u2seq=users_train, item_content=item_word_embs, item_num=item_num,
                                      max_seq_len=args.max_seq_len, use_modal=use_modal)
    Log_file.info('build dataset done...')
    len_users_train=len(users_train)
    del users_train
    gc.collect()
    
    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    Log_file.info('before seed')
    
    def worker_init_reset_seed(worker_id):
        seed_worker(worker_id, args.seed, dist.get_rank())

    Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, batch_size=args.per_device_batch_size, num_workers=args.num_workers,
                         worker_init_fn=worker_init_reset_seed, pin_memory=False, sampler=train_sampler)

    Log_file.info('build model...')
    model = Model2_transfer(args, item_num,  use_modal).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    checkpoint = None
    ckpt_path = resolve_load_checkpoint(args, "phase2")
    if ckpt_path is not None:
        Log_file.info('load ckpt if not None...')
        Log_file.info(f'load checkpoint path {ckpt_path}')
        checkpoint = load_checkpoint_into_model(model, ckpt_path, local_rank, strict=False)
        Log_file.info(f"Model loaded from {ckpt_path}")
        start_epoch = int(checkpoint.get("epoch", 0)) if args.resume else 0
        is_early_stop = True
        model.freeze()
    else:
        checkpoint = None  # new
        ckpt_path = None  # new
        start_epoch = 0
        is_early_stop = True

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) 
    optimizer = optim.AdamW(model.module.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    Log_file.info('\n')
    Log_file.info('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, best_ndcg, early_stop_count = -1.0, 0.0, 0
    steps_for_log, steps_for_eval = para_and_log(model, len_users_train, args.per_device_batch_size, Log_file,
                                                 logging_num=args.logging_num, testing_num=args.testing_num)
    scaler = torch.cuda.amp.GradScaler()
    if args.resume and checkpoint is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
        Log_file.info(f"scaler loaded from {ckpt_path}")

    Log_screen.info('{} train start'.format(args.label_screen))
    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        Log_file.info('\n')
        Log_file.info('epoch {} start'.format(now_epoch))
        Log_file.info('')
        loss, batch_index, need_break = 0.0, 1, False
        model.train()
        train_dl.sampler.set_epoch(now_epoch)

        for data in train_dl:
            sample_items_id, sample_items_content, log_mask = data
            sample_items_id, sample_items_content, log_mask = \
                sample_items_id.to(local_rank), sample_items_content.to(local_rank), log_mask.to(local_rank)
                 
            if use_modal:
                sample_items_content = sample_items_content.view(-1, sample_items_content.size(-1))
            sample_items_id = sample_items_id.view(-1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                bz_loss = model(sample_items_id, sample_items_content, log_mask, local_rank)
                loss += bz_loss.data.float()
            scaler.scale(bz_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(loss.data):
                need_break = True
                break

            if batch_index % steps_for_log == 0:
                Log_file.info('cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss: {:.5f}'.format(
                    batch_index, batch_index * args.per_device_batch_size, loss.data / batch_index, loss.data))
            batch_index += 1

        if not need_break and now_epoch % 1 == 0:
            Log_file.info('')
            max_eval_value, best_ndcg, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save, valid_metrics = \
                run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, best_ndcg, early_stop_count,
                         model, item_word_embs, users_history_for_valid, users_valid, args.eval_batch_size, item_num, use_modal,
                         args.mode, is_early_stop, local_rank)
            model.train()
            train_loss = float((loss / max(batch_index, 1)).detach().cpu())
            if dist.get_rank() == 0:
                save_checkpoint(now_epoch, model, model_dir, optimizer, scaler, Log_file.info)
                if need_save:
                    save_checkpoint(now_epoch, model, model_dir, optimizer, scaler, Log_file.info, args.save_best_name)
                append_metrics(args, {"phase": "phase2", "epoch": now_epoch, "train_loss": train_loss,
                                      "valid_HR@10": valid_metrics["HR@10"], "valid_nDCG@10": valid_metrics["nDCG@10"],
                                      "best": need_save})
        Log_file.info('')
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        Log_screen.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))
        if need_break:
            break
    if dist.get_rank() == 0:
        save_checkpoint(now_epoch, model, model_dir, optimizer, scaler, Log_file.info)
    Log_file.info('\n')
    Log_file.info('%' * 90)
    Log_file.info(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    Log_file.info(' early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))
    item_embeddings = get_item_embeddings_llm_4(model, item_word_embs, args.eval_batch_size, args, use_modal, local_rank)
    test_metrics = eval_model_step2(model,users_history_for_test, users_test, item_embeddings, args.eval_batch_size, args,
                             item_num, Log_file, args.mode, local_rank, return_full_metrics=True)
    if dist.get_rank() == 0:
        save_best_summary(args, {"best_epoch": max_epoch, "best_valid_HR@10": max_eval_value,
                                 "best_valid_nDCG@10": best_ndcg, "final_test_HR@10": test_metrics["HR@10"],
                                 "final_test_nDCG@10": test_metrics["nDCG@10"],
                                 "checkpoint_path": Path(model_dir) / args.save_best_name,
                                 "world_size": dist.get_world_size(),
                                 "per_device_batch_size": args.per_device_batch_size,
                                 "effective_global_batch_size": args.effective_global_batch_size,
                                 "eval_batch_size": args.eval_batch_size, "seed": args.seed})

def run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, best_ndcg, early_stop_count,
             model, item_word_embs, user_history, users_eval, batch_size, item_num, use_modal,
             mode, is_early_stop, local_rank):
    eval_start_time = time.time()
    Log_file.info('Validating...')
    item_embeddings = get_item_embeddings_llm_4(model, item_word_embs, batch_size, args, use_modal, local_rank)
    valid_metrics = eval_model_step2(model, user_history, users_eval, item_embeddings, batch_size, args,
                             item_num, Log_file, mode, local_rank, return_full_metrics=True)
    valid_Hit10 = valid_metrics["HR@10"]
    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    need_break = False
    need_save = False
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        best_ndcg = valid_metrics["nDCG@10"]
        max_epoch = now_epoch
        early_stop_count = 0
        need_save = True
    else:
        early_stop_count += 1
        if early_stop_count > args.patience:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return max_eval_value, best_ndcg, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save, valid_metrics
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", torch.distributed.get_rank()))
    torch.cuda.set_device(local_rank)
    set_seed(args.seed, dist.get_rank())
    configure_batch_size(args, dist.get_world_size(), print)
    gpus = torch.cuda.device_count()
    if 'modal' in args.item_tower:
        is_use_modal = True
        #model_load = args.bert_model_load
        model_load = 'bert-base-uncased/ft_local/'
        flag=0.0001
        if args.embedding_dim == 512:
            flag=5e-5
        dir_label = args.behaviors + f'pure_id_{model_load}_MOdnn_{args.mo_dnn_layers}'
        log_paras = f'{args.item_tower}-{model_load}-MOdnn_{args.mo_dnn_layers}-dnn_{args.dnn_layers}' \
                    f'_ed_{args.embedding_dim}_bs_pantry_id' \
                    f'_lr_{args.lr}'
    else:
        is_use_modal = False
        model_load = 'id'
        dir_label = str(args.item_tower) + ' ' + args.behaviors
        log_paras = f'{model_load}' \
                    f'_ed_{args.embedding_dim}_bs_{args.per_device_batch_size}' \
                    f'_lr_{args.lr}_L2_{args.l2_weight}'

    model_dir, output_dir = prepare_run_dirs(args, "phase2")
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run
    
    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank(), args.behaviors)
    Log_file.info(args)
    Log_file.info(f"seed {args.seed}")
    Log_file.info(f"patience {args.patience}")
    Log_file.info(f"checkpoint_dir {model_dir}")
    Log_file.info(f"output_dir {output_dir}")
    if dist.get_rank() == 0:
        save_args_json(args)

    start_time = time.time()
    if 'train' in args.mode:
        print(local_rank)
        train(args, is_use_modal, local_rank)
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
    print(args.gamma)
    Log_file.info("##### freeze: {} gamma: {}".format(args.freeze, args.gamma))
