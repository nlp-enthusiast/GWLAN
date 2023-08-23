import os
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from utils import Dataset, Vocab, set_seed, prepare_batch_inputs_and_labels
from model import Transformer

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, default="zh")
    parser.add_argument("--tgt_lang", type=str, default="en")

    parser.add_argument("--train_data_path", type=str, default="data/train.json")
    parser.add_argument("--dev_data_path", type=str, default="data/dev.json")
    parser.add_argument("--test_data_path", type=str, default="data/test.json")
    parser.add_argument("--vocab_path", type=str, default="data/vocab")
    parser.add_argument("--save_path", type=str, default="model/")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--update_feq", type=int, default=15)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device(args.device)

    print("-----Build Vocab-----")
    if args.vocab_path == "":
        src_vocab = Vocab.get_vocab_from_text(args.train_data_path, vocab_size=50000, src=True)
        src_vocab.save_vocab(f"data/vocab.{args.src_lang}")
        tgt_vocab = Vocab.get_vocab_from_text(args.train_data_path, vocab_size=50000, tgt=True)
        tgt_vocab.save_vocab(f"data/vocab.{args.tgt_lang}")
       
    else:
        src_vocab = Vocab.load_vocab(args.vocab_path + f".{args.src_lang}")
        tgt_vocab = Vocab.load_vocab(args.vocab_path + f".{args.tgt_lang}")
    print(f"src vocab size:{len(src_vocab)} tgt vocab size:{len(tgt_vocab)}")
    print("-----Build Dataset-----")

    test_dataset = Dataset(args.dev_data_path, src_vocab, tgt_vocab)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    print(f"test data num:{len(test_dataset)}")
    model = Transformer(len(src_vocab), len(tgt_vocab), d_model=512, batch_first=True, norm_first=True).to(device)

    state = torch.load("model/model.0.pt",map_location="cpu")
    model.load_state_dict(state["model_parameters"])
    model.eval()
    num = 0
    ncorrect=0
    for batch in tqdm(test_dataloader):
        enc_inputs, dec_inputs, dec_labels, signal_labels, mask_indexs = prepare_batch_inputs_and_labels(batch, src_vocab,tgt_vocab)
        logits = model(enc_inputs.to(device), dec_inputs.to(device))
        for mask_index, logit, signal_label in zip(mask_indexs, logits, signal_labels):
            num += 1
            print(torch.softmax(logit[mask_index, :], dim=-1))
            pre = torch.argmax(torch.softmax(logit[mask_index, :], dim=-1))
            print(pre)
            print(signal_label[0])
            input()
            if signal_label[0] == pre:
                ncorrect += 1
    acc = ncorrect / num
    print(acc)




if __name__ == '__main__':
    set_seed(2023)
    args = set_args()
    train(args)
