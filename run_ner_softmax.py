import glob
import logging
import os
import json
import time
import numpy as np
import torch
from tools.common import json_to_text
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from callback.adversarial import FGM
from tools.common import seed_everything
from tools.common import init_logger, logger
from models.transformers import WEIGHTS_NAME,  AlbertConfig
from models.bert_for_ner import BertSoftmaxForNer
from models.BART_for_ner import BartSoftmaxForNer
from models.transformers_master.models.gpt2.configuration_gpt2 import GPT2Config #new config
from models.transformers_master.models.bert.configuration_bert import BertConfig #new config
from models.transformers_master.models.bart.configuration_bart import BartConfig
from models.gpt_for_ner import GPT2SoftmaxForNer_fix, BareGPT2 , GPT2GenerateForNer
from models.gpt_LE_for_ner import GPT2SoftmaxForNer_LE# , GPT2GenerateForNer

from models.gptLMHead_for_ner import GPT2LMSoftmaxForNer, BareChineseGPT2
from models.albert_for_ner import AlbertSoftmaxForNer
from processors.utils_ner import CNerTokenizer, get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore, NewSeqEntityScore
from tools.finetuning_argparse import get_argparse
from transformers import AutoTokenizer
import wandb
import pprint

MODEL_CLASSES = {
    'bert': (BertConfig, BertSoftmaxForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertSoftmaxForNer, CNerTokenizer),
    'gpt2': (GPT2Config, GPT2SoftmaxForNer_fix, CNerTokenizer),
    "chinese_pretrained_gpt2": (GPT2Config, GPT2LMSoftmaxForNer, CNerTokenizer),
    'bare_gpt2': (GPT2Config, BareGPT2, CNerTokenizer),
    'bare_chinese_gpt2':  (GPT2Config, BareChineseGPT2, CNerTokenizer),
    'label_embedding': (GPT2Config, GPT2SoftmaxForNer_LE, CNerTokenizer),
    'generate': (GPT2Config, GPT2GenerateForNer, CNerTokenizer),
     #'bart': (BartConfig, BartSoftmaxForNer, CNerTokenizer)
}

TEMPLATE_CLASSES = {
    '1': (6, 6, 0),# use the prompt + input + prompt + input module, and cut the hidden state of the later input to classify
    #'2': (6, 32, 0),# use the prompt + input + prompt module, and cut the hidden state of the later prompt to classify
    '3': (12, 12, 0),
    #'4': (24, 24, 0),# note '4 for cluener'= (12, 24, 0)  因为gpt for ner 与 gptlm for ner 不一样
    #'5': (24, 88, 0)# todo 88 = 24+64  ontonote  这个是不是写错了啊 为啥这么低！！！check！！难道不能用24+64？
    #'6': (12, 32, 0)
}
# modify the template for prompt my changing TEMPLATE_CLASSES

TRAIN_LIMIT = None
EVAL_LIMIT = None
TEST_LIMIT = None
# modify the number of examples for train, eval, test
# the default is None, meaning use all the data from files.

use_wandb = False
if use_wandb:
    sweep_config = {
        'method': 'random',# grid, random
        'metric': {
            'name': 'f1',
            'goal': 'maximize'
        },
        'parameters': {
            'weight_decay': {
                'values': [0.004, 0.005, 0.006, 0.008, 0.01, 0.012]
            },
            'learning_rate': {
                'values': [2e-4, 1e-4, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5]
            },

            'epochs': {
                'values': [3]
            }, # fix
            'train_max_seq_length': {
                'values':[64]
            },
            'eval_max_seq_length': {
                'values': [64]
            }
        }
    }
    pprint.pprint(sweep_config)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:#自动半精度训练，缩短训练时间和模型大小
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to global_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    if args.do_adv:# 对抗训练
        fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)#fast gradient method
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')

        for step, batch in enumerate(train_dataloader):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don't use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]# model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.do_adv:
                fgm.attack()
                loss_adv = model(**inputs)[0]
                if args.n_gpu>1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()

            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and epoch == args.num_train_epochs-1:
                #     # Log metrics
                #     print(" in the last epoch, do testing  ")
                #     if args.local_rank == -1:
                #         # Only evaluate when single GPU otherwise metrics may not average well
                #         predict(args, model, tokenizer)

                if args.local_rank in [-1, 0] and args.save_steps > 0 \
                        and global_step % args.save_steps == 0 and args.save_model and epoch == args.num_train_epochs-1:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    if args.model_name_or_path in ["gpt2", 'gpt2-large']:
                        checkpoint = {"model_state_dict": model.state_dict(),
                                      "optimizer_state_dict": optimizer.state_dict(),
                                      "global_step": global_step,
                                      "epoch": epoch}
                        logger.info("Saving model checkpoint to %s", output_dir)
                        torch.save(checkpoint, os.path.join(output_dir, "model.pkl"))
                    else:
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        tokenizer.save_vocabulary(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

        print(" in the last epoch, do testing  ")
        predict(args, model, tokenizer)

        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix=''):
    if args.model_type == "chinese_pretrained_gpt2":
        metric = SeqEntityScore(args.id2label, markup=args.markup)
    else:
        metric = NewSeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev', limit = EVAL_LIMIT)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    output_results = []
    labels = []
    output_submit_file = os.path.join(eval_output_dir, prefix, args.output_file_name)

    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)

        tmp_eval_loss, logits = outputs[:2]
        # convert the example into tokens and add into json_d
        example = outputs[2]
        example = example.tolist()
        example = tokenizer.decode(example)
        example = ' '.join(example)

        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        input_lens = batch[4].cpu().numpy().tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i]-1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(preds[i][j])


        # to wirte results in the output file
        labels.append(batch[3])
        if args.task_name in ['cluener','cner']:
            preds = preds[0][1:-1]# [CLS]XXXX[SEP]
        else:
            preds = preds[0]# 对于英文没有用[cls]和[sep] 因此不截取
        tags = [args.id2label[x] for x in preds]
        #label_entities = get_entities(preds, args.id2label, args.markup)
        #true_labels = batch[3].detach().cpu().numpy().tolist()[0]
        #true_label_entities = get_entities(true_labels, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        #json_d['true_tag_seq'] = " ".join(true_labels)
        json_d['pred_tag_seq'] = " ".join(tags)
        json_d['example of the gpt2 output words'] = example
        #json_d['entities'] = label_entities
        #json_d['true_entities'] = true_label_entities
        output_results.append(json_d)

        pbar(step)

    with open(output_submit_file, "w") as writer:
        for record in output_results:
            writer.write(json.dumps(record) + '\n')

    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps

    if args.model_type == "chinese_pretrained_gpt2":
        eval_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        logger.info("***** Eval results %s *****", prefix)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)
        logger.info("***** Entity results %s *****", prefix)
        for key in sorted(entity_info.keys()):
            logger.info("******* %s results ********"%key)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            logger.info(info)
    else:
        eval_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        logger.info("***** Eval results %s *****", prefix)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        logger.info(info)

    if use_wandb:
        wandb.log(results)
    return results

def predict(args, model, tokenizer, prefix = ''):
    if args.model_type == "chinese_pretrained_gpt2":
        metric = SeqEntityScore(args.id2label, markup=args.markup)
    else:
        metric = NewSeqEntityScore(args.id2label, markup=args.markup)
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test', limit = TEST_LIMIT)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    output_results = []
    output_submit_file = os.path.join(pred_output_dir, prefix, args.output_file_name)
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)

            logits = outputs[0]
            preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            input_lens = batch[4].cpu().numpy().tolist()
            out_label_ids = batch[3].cpu().numpy().tolist()

        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i]-1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(preds[i][j])


        # used for write results in output file
        if args.task_name in ['cluener', 'cner', 'ontonote4']:
            preds = preds[0][1:-1]# [CLS]XXXX[SEP]
        else:
            preds = preds[0]# 英文没有用[cls]和[sep] 因此不截取

        # tags = [args.id2label[x] for x in preds]
        if args.model_type == "chinese_pretrained_gpt2":
            label_entities = get_entities(preds, args.id2label, args.markup)
            #true_labels = batch[3].detach().cpu().numpy().tolist()[0]
            #true_label_entities = get_entities(true_labels, args.id2label, args.markup)
            json_d = {}
            json_d['id'] = step
            #f = all_tokens[step]
            #json_d['token'] = f
            #json_d['true_tag_seq'] = " ".join(true_labels)
            # json_d['tag_seq'] = " ".join(tags)
            json_d['entities'] = label_entities
            #json_d['true_entities'] = true_label_entities
            output_results.append(json_d)

        pbar(step)

    logger.info("\n")
    if args.task_name != 'cluener':
        if args.model_type == "chinese_pretrained_gpt2":
            test_info, entity_info = metric.result()
            results = {f'{key}': value for key, value in test_info.items()}
            if use_wandb:
                wandb.log(results)
            logger.info("***** Test results %s *****", prefix)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
            logger.info(info)
            logger.info("***** Test Entity results %s *****", prefix)
            for key in sorted(entity_info.keys()):
                logger.info("******* %s results ********"%key)
                info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
                logger.info(info)
        else:
            test_info = metric.result()
            results = {f'{key}': value for key, value in test_info.items()}
            logger.info("***** Test results %s *****", prefix)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
            logger.info(info)

        if use_wandb:
            wandb.log(results)
        with open(output_submit_file, "w") as writer:
            for record in output_results:
                writer.write(json.dumps(record) + '\n')


    # get the test results !
    print("get the test results")
    if args.task_name == "cluener":
        output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
        test_text = []
        with open(os.path.join(args.data_dir, "test.json"), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, output_results):
            json_d = {}
            json_d['id'] = x['id']
            json_d['label'] = {}
            entities = y['entities']
            words = list(x['text'])
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file, test_submit)


def load_and_cache_examples(args, task, tokenizer, data_type='train', limit = None):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_soft-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type=='train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.markup)
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir, limit)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir, limit)
        else:
            examples = processor.get_test_examples(args.data_dir, limit)

        if args.task_name in ['cluener', 'cner', 'ontonote4']:
            ENGLISH = False
        else:
            ENGLISH = True
        # gpt2tokenizer 没有sep_token  pad_token cls_token 因此都是None
        features, count = convert_examples_to_features(english=ENGLISH, markup=args.markup,
                                                tokenize_split_with_O=args.tokenize_split_with_O,
                                                task_name=data_type,
                                                tokenizer_name=args.tokenizer_name if args.tokenizer_name!='' else args.model_name_or_path,
                                                examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type=='train' \
                                                               else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token ="[CLS]",
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token="[SEP]",
                                                # pad on the left for xlnet
                                                pad_token=0,
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        print("number of examples whose labels cannot be aligned "+str(count))# 统计所有不能正常tokenize（label与input_id不对应）的examples

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset

def main():
    args = get_argparse().parse_args()
    if args.model_type == "chinese_pretrained_gpt2":
        assert args.task_name in ['cluener', 'cner', 'ontonote4']
        assert args.markup == 'biso'# 中文一律采用biso
    if use_wandb:
        wandb.init(config=args, project='gpt2_sweep_2_try', entity='li_xuechun')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda", args.cuda)
        # torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1 #目前只采用1个GPU训练
        # torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1),args.fp16,)
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()

    # 按照markup方式得到labels： bieso, biso, bio
    label_list = processor.get_labels(args.markup)
    if args.markup == 'bio':
        assert (len(label_list)-1) % 2 == 0
    elif args.markup == 'biso':
        assert (len(label_list)-1) % 3 == 0
    elif args.markup == 'bieso':
        assert (len(label_list)-1) % 4 == 0

    assert label_list[0] == 'O'

    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    TEMPLATE = TEMPLATE_CLASSES[args.template]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          loss_type=args.loss_type,
                                          cache_dir=args.cache_dir if args.cache_dir else None,)

    if args.model_name_or_path in ['gpt2', 'gpt2-large', 'gpt2-medium', "distilgpt2"]:
        if args.task_name in ['cluener', 'cner', 'ontonote4']:
            # 中文只采用bert-base-chinese 并且采用 Cner-tokenizer
            tokenizer_name = 'bert-base-chinese'
            tokenizer = tokenizer_class.from_pretrained(tokenizer_name,
                                                        do_lower_case=args.do_lower_case,
                                                        cache_dir=args.cache_dir if args.cache_dir else None,)
        else:
            # 英文采用与model一致的tokenizer(不采用cner tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name != '' else args.model_name_or_path, use_fast=False)#use_fast=True, add_prefix_space=True

        # model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
        #                                     config=config, device=args.device, template=TEMPLATE, model_name=args.model_name_or_path,  cache_dir=args.cache_dir if args.cache_dir else None,)
        model = model_class(config=config, device=args.device, template=TEMPLATE, model_name=args.model_name_or_path)

    else:# bert or albert
        if args.task_name in ['cluener', 'cner', 'ontonote4']:
            # 中文只采用bert-base-chinese
            tokenizer_name = 'bert-base-chinese'
            tokenizer = tokenizer_class.from_pretrained(tokenizer_name,
                                                        do_lower_case=args.do_lower_case,
                                                        cache_dir=args.cache_dir if args.cache_dir else None,)
        else:
            # 英文采用与model一致的tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name != '' else args.model_name_or_path, use_fast=False)
        # for bert or albert, load the model in the from_pretrained way!
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config, device=args.device, template=TEMPLATE, model_name=args.model_name_or_path,
                                            cache_dir=args.cache_dir if args.cache_dir else None,)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train', limit=TRAIN_LIMIT)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.save_model:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = (
        #     model.module if hasattr(model, "module") else model
        # )  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # todo tokenizer 训练了吗？？？咋区分的best perform model？？？

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        # torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
        # torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
        # logger.info("Saving optimizer and scheduler states to %s", args.output_dir)

    # Evaluation（加载保存的模型，单独evaluation）
    results = {}
    if args.do_eval_with_saved_model and args.local_rank in [-1, 0]:
        if args.task_name in ['cluener', 'cner', 'ontonote4']:
            # 中文延用tokenizer_class
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        else:
            # 英文采用AutoTokenizer todo assume containing vocabulary files named ['vocab.txt'] but couldn't find such vocabulary files at this path or url, save 的是json
            #tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name != '' else args.model_name_or_path, use_fast=False)

        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            if args.model_name_or_path in ["gpt2", 'gpt2-large']:
                checkpoint = os.path.join(checkpoint, "checkpoint-{}".format(10), "model.pkl") # todo should not use 10
                checkpoint = torch.load(checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])

            else:# bert 可以直接利用from_pretrained函数按照url/model_name中加载
                model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    # test（加载保存的模型，单独test）
    if args.do_predict_with_saved_model and args.local_rank in [-1, 0]:
        if args.task_name in ['cluener', 'cner', 'ontonote4']:
            # 中文延用tokenizer_class
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        else:
            # 英文采用AutoTokenizer todo assume containing vocabulary files named ['vocab.txt'] but couldn't find such vocabulary files at this path or url, save 的是json
            # tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name != '' else args.model_name_or_path, use_fast=False)

        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            if args.model_name_or_path in ["gpt2", 'gpt2-large']:
                checkpoint = os.path.join(checkpoint, "checkpoint-{}".format(10), "model.pkl") # todo should not use 10
                checkpoint = torch.load(checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model = model_class.from_pretrained(checkpoint, template=TEMPLATE, device=args.device)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)

if use_wandb:
    sweep_id = wandb.sweep(sweep_config,  project='gpt2_sweep_2_try', entity='li_xuechun')
    wandb.agent(sweep_id, function=main)

if __name__ == "__main__":
    main()