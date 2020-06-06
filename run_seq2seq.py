#!/usr/bin/env python3
# coding=utf-8

"""
Seq2Seq
Derived from run_ner.py, this file makes a seq2seq model using transformers and trains it.
This file handles a very large amount of arguments using argparse, which appears to be the standard way
within transformers to specify an experiment.
The training and testing data format is defined in utils_seq2seq.py.
"""

from transformers import EncoderDecoderModel, BertTokenizer, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import islice
from keras.preprocessing.sequence import pad_sequences
import copy
import argparse
from tqdm import tqdm

from os import walk, path
import os

from IPython.display import clear_output


def load_files(file):
    with open(file, encoding='UTF-8') as f:
        for line in f:
            yield line

def batch_generator(loader=None, tokenizer=None, dataset=None, batch_size=16, max_seq_length=10):
    input_batch = list(islice(loader[f'{dataset}_source'], batch_size))
    target_batch = list(islice(loader[f'{dataset}_target'], batch_size))
    while input_batch and target_batch:
        input_elem = tokenizer.batch_encode_plus(input_batch,
                                                 max_length=max_seq_length,
                                                 return_tensors="pt",
                                                 pad_to_max_length=True)
        target_elem = tokenizer.batch_encode_plus(target_batch,

                                                 max_length=max_seq_length,
                                                 return_tensors="pt",
                                                 pad_to_max_length=True)
        yield (input_elem["input_ids"],
               input_elem["attention_mask"],
               target_elem["input_ids"],
               target_elem["attention_mask"],
               target_elem["input_ids"])
        input_batch = list(islice(loader[f'{dataset}_source'], batch_size))
        target_batch = list(islice(loader[f'{dataset}_target'], batch_size))

def batch_generator_with_pad(loader=None, tokenizer=None, dataset=None, batch_size=16, max_seq_length=10):
    input_batch = list(islice(loader[f'{dataset}_source'], batch_size))
    target_batch = list(islice(loader[f'{dataset}_target'], batch_size))
    while input_batch and target_batch:
        input_batch = [tokenizer.tokenize(sent, add_special_tokens=True) for sent in input_batch]
        target_batch = [tokenizer.tokenize(sent, add_special_tokens=True) for sent in target_batch]
        lm_labels = copy.deepcopy(target_batch)
        [sent.insert(0, "[PAD]") for sent in lm_labels]
        input_batch = [tokenizer.convert_tokens_to_ids(x) for x in input_batch]
        target_batch = [tokenizer.convert_tokens_to_ids(x) for x in target_batch]
        lm_labels = [tokenizer.convert_tokens_to_ids(x) for x in lm_labels]
        input_batch = pad_sequences(input_batch,
                                  maxlen=max_seq_length,
                                  dtype="long",
                                  truncating="post",
                                  padding="post")
        target_batch = pad_sequences(target_batch,
                                  maxlen=max_seq_length,
                                  dtype="long",
                                  truncating="post",
                                  padding="post")
        lm_labels = pad_sequences(lm_labels,
                                  maxlen=max_seq_length,
                                  dtype="long",
                                  truncating="post",
                                  padding="post")

        attention_masks_encode = [[float(i>0) for i in seq] for seq in input_batch]
        attention_masks_decode = [[float(i>0) for i in seq] for seq in target_batch]
        yield (torch.tensor(input_batch, dtype=torch.long),
               torch.tensor(attention_masks_encode),
               torch.tensor(target_batch, dtype=torch.long),
               torch.tensor(attention_masks_decode),
               torch.tensor(lm_labels))
        input_batch = list(islice(loader[f'{dataset}_source'], batch_size))
        target_batch = list(islice(loader[f'{dataset}_target'], batch_size))

def batch_loader(tokenizer, data_dir, step='test', batch_size=16, start_pad=False):
    print('Reading examples : ' + step)
    input_data_loader = {}

    source_file = os.path.join(data_dir, "{}.txt".format(step + '_source'))
    target_file = os.path.join(data_dir, "{}.txt".format(step + '_target'))

    input_data_loader[f'{step}_source'] = load_files(source_file)
    input_data_loader[f'{step}_target'] = load_files(target_file)

    if start_pad:
        return batch_generator_with_pad(input_data_loader, tokenizer, step, batch_size)
    else:
        return batch_generator(input_data_loader, tokenizer, step, batch_size)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents.",
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents.",
    )
    parser.add_argument(
        "--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.",
    )
    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--optimizer", default="lamb", type=str, help="Optimizer (AdamW or lamb)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank",
    )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()


    # New example based on https://colab.research.google.com/drive/1uVP09ynQ1QUmSE2sjEysHjMfKgo4ssb7?usp=sharing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE: ' + str(device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'bert-base-cased')
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)


    model.train()
    train_loss_set = []
    train_loss = 0
    save_step = 500

    for epoch in range(int(args.num_train_epochs)):
        batches = tqdm(batch_loader(tokenizer, args.data_dir, step='train', batch_size=args.train_batch_size, start_pad=False), desc='Training')
        for step, batch in enumerate(batches):
            batch = tuple(t.to(device) for t in batch)
            input_ids_encode, attention_mask_encode, input_ids_decode, attention_mask_decode, lm_labels = batch
            optimizer.zero_grad()
            model.zero_grad()

            loss, outputs = model(input_ids=input_ids_encode,
                              decoder_input_ids=input_ids_decode,
                              attention_mask = attention_mask_encode,
                              decoder_attention_mask = attention_mask_decode,
                              lm_labels=lm_labels)[:2]

            train_loss_set.append(loss.item())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(epoch)
        clear_output(True)
        plt.plot(train_loss_set)
        plt.title(f'Training loss. Epoch {epoch}')
        plt.xlabel(f'Batch {step}')
        plt.ylabel('Loss')
        plt.show()




    print('STARTING EVALUATION')
    model.eval()

    test_batches = tqdm(batch_loader(tokenizer, args.data_dir, step='test', batch_size=1, start_pad=True), desc='Evaluating')
    for step, batch in enumerate(test_batches):
        batch = tuple(t.to(device) for t in batch)
        input_ids_encode, attention_mask_encode, input_ids_decode, attention_mask_decode, lm_labels = batch
        with torch.no_grad():
            generated = model.generate(input_ids_encode, attention_mask = attention_mask_encode, decoder_start_token_id=model.config.decoder.pad_token_id,
                                    do_sample=True,
                                    max_length=10,
                                    top_k=200,
                                    top_p=0.75,
                                    num_return_sequences=10,
                                    #num_beams=5,
                                    #no_repeat_ngram_size=2,
                                )
            for i in range(len(generated)):
                print(f'Generated {i}: {tokenizer.decode(generated[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)}')

            print('Expected: ', ' '.join([tokenizer.decode(elem, skip_special_tokens=True, clean_up_tokenization_spaces=True) for elem in input_ids_decode]))
            print('Lm Labels: ', ' '.join([tokenizer.decode(elem, skip_special_tokens=True, clean_up_tokenization_spaces=True) for elem in lm_labels]))
            print('Input: ', ' '.join([tokenizer.decode(elem, skip_special_tokens=True, clean_up_tokenization_spaces=True) for elem in input_ids_encode]))
            print()



if __name__ == "__main__":
    main()
