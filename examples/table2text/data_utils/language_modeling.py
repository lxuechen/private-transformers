import copy
import json
import os
import sys

import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineE2ETextDataset(Dataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        bos_tok: str,
        eos_tok: str,
        max_seq_len=sys.maxsize,
        max_examples=sys.maxsize,
        **_,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [
                line.split('||')
                for line in f.read().splitlines() if (
                    len(line) > 0 and not line.isspace() and len(line.split('||')) == 2
                )
            ]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        edited_sents = []
        for src, tgt in zip(src_lines, tgt_lines):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        # --- Filter out super long sentences ---
        new_src_lines, new_tgt_lines, new_edited_sents = [], [], []
        for src_line, tgt_line, edited_sent in zip(src_lines, tgt_lines, edited_sents):
            tokenized_edited_sent = tokenizer.tokenize(edited_sent)
            if len(tokenized_edited_sent) <= max_seq_len:
                new_src_lines.append(src_line)
                new_tgt_lines.append(tgt_line)
                new_edited_sents.append(edited_sent)
            del src_line, tgt_line, edited_sent
        src_lines, tgt_lines, edited_sents = new_src_lines, new_tgt_lines, new_edited_sents
        # ---------------------------------------

        # --- Truncate the dataset if necessary; this must be after the length filtering. ---
        src_lines = src_lines[:max_examples]
        tgt_lines = tgt_lines[:max_examples]
        edited_sents = edited_sents[:max_examples]
        # ---

        batch_encoding = tokenizer(
            edited_sents,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            is_split_into_words=False,
        )

        self.examples = batch_encoding["input_ids"]
        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(
            ssl_lst,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            is_split_into_words=True
        )['input_ids']

        self.src_sent = []
        self.tgt_sent = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0

        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.src_sent.append(self.examples[i][:sep_idx - 1])
            self.tgt_sent.append(self.examples[i][sep_idx - 1:])
            self.labels[i][:sep_idx] = [-100] * sep_idx  # Doesn't contribute to loss.
            temp_src_len += sep_idx - 1
            temp_tgt_len += len(elem) - (sep_idx - 1)
            temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)

        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            torch.tensor(self.src_sent[i], dtype=torch.long),
            torch.tensor(self.tgt_sent[i], dtype=torch.long),
            torch.tensor(self.src_cat[i], dtype=torch.long),
        )


class LineByLineWebNLGTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        bos_tok: str,
        eos_tok: str,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                if j > 0:
                    temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in sents:
                if sent["comment"] == 'good':
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)

        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)

        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                                 is_split_into_words=True)['input_ids']

        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0

        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx - 1])  # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx - 1:])  # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)

        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        print()
        print(self.labels[1])
        print(self.examples[1])
        print(edited_sents[1])
        print(self.src_sent[1])
        print(self.tgt_sent[1])
        print(self.src_cat[1])
        assert len(self.src_cat) == len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            torch.tensor(self.src_sent[i], dtype=torch.long),
            torch.tensor(self.tgt_sent[i], dtype=torch.long),
            torch.tensor(self.src_cat[i], dtype=torch.long),
        )


class LineByLineTriplesTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        bos_tok: str,
        eos_tok: str,
        max_seq_len=sys.maxsize,
        max_examples=sys.maxsize,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []
        for example in lines_dict:
            rela_lst = []
            temp_triples = ''
            for i, tripleset in enumerate(example['tripleset']):
                subj, rela, obj = tripleset
                rela = rela.lower()
                rela_lst.append(rela)
                if i > 0:
                    temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in example['annotations']:
                full_tgt_lst.append(sent['text'])
                full_src_lst.append(temp_triples)
                full_rela_lst.append(rela_lst)

        # Truncate the dataset if necessary.
        full_rela_lst = full_rela_lst[:max_examples]
        full_src_lst = full_src_lst[:max_examples]
        full_tgt_lst = full_tgt_lst[:max_examples]

        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)

        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = f' {src} {bos_tok} {tgt} {eos_tok} '
            edited_sents.append(sent)

        # --- Filter out super long sentences ---
        this_full_rela_lst = []
        this_full_src_lst = []
        this_full_tgt_lst = []
        this_edited_sents = []
        for rela, src, tgt, edited_sent in zip(full_rela_lst, full_src_lst, full_tgt_lst, edited_sents):
            tokenized_edited_sent = tokenizer.tokenize(edited_sent)
            if len(tokenized_edited_sent) <= max_seq_len:
                this_full_rela_lst.append(rela)
                this_full_src_lst.append(src)
                this_full_tgt_lst.append(tgt)
                this_edited_sents.append(edited_sent)
        full_rela_lst = this_full_rela_lst
        full_src_lst = this_full_src_lst
        full_tgt_lst = this_full_tgt_lst
        edited_sents = this_edited_sents
        # ---------------------------------------

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                                 is_split_into_words=True)['input_ids']

        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx - 1])  # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx - 1:])  # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx

                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)

        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            torch.tensor(self.src_sent[i], dtype=torch.long),
            torch.tensor(self.tgt_sent[i], dtype=torch.long),
            torch.tensor(self.src_cat[i], dtype=torch.long),
        )
