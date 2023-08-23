import json
import random
import torch
import numpy as np
from tqdm import tqdm

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


class Vocab:
    def __init__(self, token2id=None, id2token=None):
        if token2id is not None and id2token is not None:
            self.token2id = token2id
            self.id2token = id2token
            self.pad = "[PAD]"
            self.bos = "[BOS]"
            self.eos = "[EOS]"
            self.unk = "[UNK]"
            self.bos_id = self.token2id[self.bos]
            self.eos_id = self.token2id[self.eos]
            self.unk_id = self.token2id[self.unk]
            if "[MASK]" in self.token2id:
                self.mask_id = self.token2id["[MASK]"]


    @classmethod
    def get_vocab_from_text(cls, text_path, vocab_size, src=False, tgt=False):
        id2token, token2id = {0: "[PAD]", 1: "[BOS]", 2: "[EOS]",3:"[UNK]"}, {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2,"[UNK]":3}
        num4token = {}
        with open(text_path) as f:
            data = f.readlines()
            for item in data:
                item = json.loads(item)
                if src:
                    sentence = item["src"].strip().split()
                if tgt:
                    sentence = item["context"].strip().split()
                for token in sentence:
                    num4token[token] = num4token.get(token, 0) + 1
        num4token = sorted(num4token.items(), key=lambda x: x[1], reverse=True)
        num4token = num4token[:vocab_size]
        for id, (token, token_num) in enumerate(num4token):
            id2token[len(id2token)] = token
            token2id[token] = len(token2id)

        return cls(token2id, id2token)

    @classmethod
    def load_vocab(cls, vocab_path):
        id2token, token2id = {}, {}
        with open(vocab_path) as f:
            tokens = f.readlines()
            for token in tokens:
                token = token.strip()
                id2token[len(id2token)] = token
                token2id[token] = len(token2id)
        return cls(token2id,id2token)
    def tokenize(self, sentence: str, add_bos=False, add_eos=False):
        sentence_list = sentence.strip().split()
        sentence_id = []
        for token in sentence_list:
            if token in self.token2id:
                sentence_id +=[self.token2id[token]]
            else:
                sentence_id += [self.unk_id]

        if add_bos:
            sentence_id = [self.bos_id] + sentence_id
        if add_eos:
            sentence_id += [self.eos_id]

        return sentence_id

    def decode(self, ids: list):
        str = []
        for id in ids:
            if id in self.id2token:
                str+=[self.id2token[id]]
            else:
                str+=[self.unk]
        return " ".join(str)

    def save_vocab(self, vocab_save_path: str):
        with open(vocab_save_path, "w", encoding="utf-8") as f:
            for token in self.token2id.keys():
                f.writelines(token + "\n")

    def __len__(self):
        return len(self.token2id)


class Dataset:
    def __init__(self, data_path, src_vocab, tgt_vocab):
        self.all_train_need = self.load_data(data_path, src_vocab, tgt_vocab)

    def load_data(self, data_path, src_vocab: Vocab, tgt_vocab: Vocab,add_bos=True):
        all_train_need = []
        f = open(data_path)
        data = f.readlines()
        for item in data:
            temp = {}
            item = json.loads(item)
            temp["src"] = src_vocab.tokenize(item["src"])
            if len(temp["src"])>256:
                continue
            temp["context"] = tgt_vocab.tokenize(item["context"],add_bos=True,add_eos=True)
            temp["sigle_label"] = tgt_vocab.tokenize(item["label"])

            temp["mask_index"] =temp["context"].index(tgt_vocab.mask_id)
            temp["labels"] = [-100] * len(temp["context"])
            temp["labels"][temp["mask_index"]] = temp["sigle_label"][0]
            all_train_need.append(temp)

        return all_train_need

    def __len__(self):
        return len(self.all_train_need)

    def __getitem__(self, idx):
        return self.all_train_need[idx]


def pad(seq: list,maxlen, pad_id=0):
    paded_seq = []

    for s in seq:
        paded_seq.append(s + [pad_id] * (maxlen - len(s)))
        # mask_list.append([s] + [pad_id] * (maxlen - len(s)))
    return paded_seq


def prepare_batch_inputs_and_labels(batch: list,src_vocab,tgt_vocab):
    enc_inputs = []
    dec_inputs = []
    dec_labels = []
    single_labels = []
    mask_indexs = []
    src_max_len = 0
    tgt_max_len = 0
    label_max_len = 0
    for item in batch:
        enc_inputs.append(item["src"])
        src_max_len = max(src_max_len, len(item["src"]))
        dec_inputs.append(item["context"])
        tgt_max_len = max(tgt_max_len, len(item["context"]))
        dec_labels.append(item["labels"])
        label_max_len = max(label_max_len, len(item["labels"]))
        mask_indexs.append(item["mask_index"])
        single_labels+=item["sigle_label"]

    # print(src_vocab.decode(enc_inputs[0]))
    # print(tgt_vocab.decode(dec_inputs[0]))
    # print(tgt_vocab.decode(dec_labels[0]))
    # print(tgt_vocab.decode(single_labels[0]))
    #
    # exit()
    enc_inputs = pad(enc_inputs, src_max_len)
    dec_inputs = pad(dec_inputs, tgt_max_len)
    dec_labels = pad(dec_labels, label_max_len,-100)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_labels),torch.LongTensor(single_labels),mask_indexs



