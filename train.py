import os
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast,GradScaler

import torch.distributed as dist
from torch.utils import data
import torch.multiprocessing as mp

from myutils import Dataset, Vocab, set_seed, prepare_batch_inputs_and_labels
from model import Transformer
# from tf import Transformer


import os





def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, default="zh")
    parser.add_argument("--tgt_lang", type=str, default="en")

    parser.add_argument("--train_data_path", type=str, default="data/train.json")
    parser.add_argument("--dev_data_path", type=str, default="data/dev.json")
    parser.add_argument("--test_data_path", type=str, default="data/test.json")
    parser.add_argument("--vocab_path", type=str, default="data/vocab")
    parser.add_argument("--save_path", type=str, default="models/")

    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--dl", type=str, default="2", help="指定gpu编号")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--update_feq", type=int, default=1)
    args = parser.parse_args()
    return args

def eval(dev_dataloader,model,early_stop,src_vocab,tgt_vocab,device,best_acc,optimizer):
    num, ncorrect = 0, 0
    pre_words = []
    target_words = []
    for batch in dev_dataloader:
        enc_inputs, dec_inputs, dec_labels, singnal_labels, mask_indexs = prepare_batch_inputs_and_labels(batch,
                                                                                                          src_vocab,
                                                                                                          tgt_vocab)
        logits = model(enc_inputs.to(device), dec_inputs.to(device))
        for mask_index, logit, signal_label in zip(mask_indexs, logits, singnal_labels):
            num += 1
            pre = torch.argmax(torch.softmax(logit[mask_index, :], dim=-1))
            pre_words.append(int(pre))
            target_words.append(int(signal_label))
            if signal_label == pre:
                ncorrect += 1
    pre_words = tgt_vocab.decode(pre_words).split()
    target_words = tgt_vocab.decode(target_words).split()
    print(pre_words[:10])
    print(target_words[:10])
    acc = ncorrect / num

    early_stop += 1
    if best_acc <= acc:
        best_acc = acc
        early_stop = 0
        # torch.save({"model_parameters": model.module.state_dict(),
        #             "optimizer_parameters": optimizer.state_dict()},
        #            os.path.join(args.save_path, f'model.best.pt'))
    print(f"ACC:{acc:.4f} BEST-ACC:{best_acc:.4f}")

    return acc,early_stop,best_acc


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
    train_dataset = Dataset(args.dev_data_path, src_vocab, tgt_vocab)
    train_sampler = data.distributed.DistributedSampler(train_dataset,shuffle=True,)
    train_dataloader = DataLoader(train_dataset,sampler=train_sampler, batch_size=args.batch_size,  collate_fn=lambda x: x)

    dev_dataset = Dataset(args.dev_data_path, src_vocab, tgt_vocab)
    dev_sampler = data.distributed.DistributedSampler(dev_dataset, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset,sampler=dev_sampler, batch_size=args.batch_size, collate_fn=lambda x: x)

    test_dataset = Dataset(args.test_data_path, src_vocab, tgt_vocab)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    print(f"train data num:{len(train_dataset)} dev data num:{len(dev_dataset)} test data num:{len(test_dataset)}")
    model = Transformer(len(src_vocab), len(tgt_vocab), d_model=512, batch_first=True,norm_first=True).to(args.local_rank)
    # state = torch.load(args.save_path+"model.best.pt")
    # model.load_state_dict(state_dict=state["model_parameters"])
    # model = Transformer(len(src_vocab), len(tgt_vocab), d_model=512).to(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scaler = GradScaler()

    train_steps = int(len(train_dataset) * args.epochs / args.batch_size)
    # eval_step = int(1*len(train_dataset) / args.batch_size / args.gpu_num)
    eval_step = 3000

    step = 0
    best_acc = 0
    early_stop = 0
    loss_cache = []
    print_loss = 200
    print("-----Start Training----")
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        print(f"This is epoch {epoch}")
        for batch in tqdm(train_dataloader):
            step += 1
            enc_inputs, dec_inputs, dec_labels, single_labels, mask_indexs = prepare_batch_inputs_and_labels(batch,src_vocab,tgt_vocab)
            # label_logits = torch.zeros(0,50004).to(device)
            # print(src_vocab.decode(enc_inputs[0].tolist()))
            # print(tgt_vocab.decode(dec_inputs[0].tolist()))
            # print(tgt_vocab.decode([dec_inputs[0].tolist()[int(mask_indexs[0])]]))
            # print(tgt_vocab.decode([dec_labels[0].tolist()[int(mask_indexs[0])]]))
            # input()
            optimizer.zero_grad()
            with autocast():
                logits = model(enc_inputs.to(device), dec_inputs.to(device))

                # for logit,index in zip(logits,mask_indexs):
                #     label_logits = torch.cat((label_logits,logit[index].unsqueeze(dim=0)),dim=0)
                # loss = loss_fn(label_logits.view(-1, len(tgt_vocab)), single_labels.to(device))

                loss = loss_fn(logits.view(-1, len(tgt_vocab)), dec_labels.view(-1).to(device))

            loss_cache.append(loss.item())

            if (step/args.update_feq) % print_loss == 0:
                # print(loss.item())
                loss_cache = loss_cache[-250:]
                print(f"epoch:{epoch} step:{step//args.update_feq} loss:{sum(loss_cache) / len(loss_cache):.4f}")

            scaler.scale(loss).backward()
            if step%args.update_feq==0:
                scaler.step(optimizer)
                scaler.update()

            if step%eval_step==0:
                acc,early_stop,best_acc = eval(dev_dataloader,model,early_stop,src_vocab,tgt_vocab,device,best_acc,optimizer)
                if early_stop == args.patience:
                    break


        acc, early_stop,best_acc = eval(dev_dataloader, model, early_stop,src_vocab,tgt_vocab,device,best_acc,optimizer)
        if early_stop == args.patience:
            break

        torch.save({"model_parameters": model.module.state_dict(),
                    "optimizer_parameters": optimizer.state_dict()},
                   os.path.join(args.save_path, f'model.{epoch}.pt'))



if __name__ == '__main__':
    set_seed(2023)
    args = set_args()
    device_list = [int(item) for item in args.dl.split(",")]
    args.gpu_num = len(device_list)
    torch.distributed.init_process_group(backend="nccl")
    args.local_rank = device_list[args.local_rank]
    torch.cuda.set_device(args.local_rank)
    train(args)
#     python -m torch.distributed.launch --nproc_per_node 2 train.py --dl 2,3

