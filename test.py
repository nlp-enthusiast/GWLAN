import argparse
import torch
from torch.utils import data


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, input):
        ans = torch.ones(0).cuda()
        for item in input:
            output = self.linear(item.unsqueeze(dim=0))
            ans = torch.cat((ans, output), dim=0).cuda()
        return ans


class Dataset(data.Dataset):
    def __init__(self):
        self.data = [[i, i * 2 + 1] for i in range(1, 37*10000)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def collate_fn(example):
    input, ans = [], []
    for item in example:
        input.append(item[0])
        ans.append(item[1])
    return torch.tensor(input, dtype=torch.float32), torch.tensor(ans, dtype=torch.float32)


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--dl", type=str, default="0",help="指定gpu编号")
args = parser.parse_args()

torch.distributed.init_process_group(backend="nccl")
device_list = [int(item) for item in args.dl.split(",")]
args.local_rank = device_list[args.local_rank]
torch.cuda.set_device(args.local_rank)
model = Model()
model.to(args.local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
dataset = Dataset()
sampler = data.distributed.DistributedSampler(dataset)
data_loader = data.DataLoader(dataset, batch_size=2,  sampler=sampler, collate_fn=collate_fn)

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    sampler.set_epoch(epoch)
    for i, (input, target) in enumerate(data_loader):
        output = model(input.cuda())
        optimizer.zero_grad()
        loss = loss_func(output, target.cuda())
        loss.backward()
        optimizer.step()
        print(f"epoch:{epoch + 1} batch:{i + 1}/{len(data_loader)} loss:{loss.item()}")

for i, (input, target) in enumerate(data_loader):
    output = model(input)
    print(f"input:{input.tolist()} output:{output.tolist()} target:{target.tolist()}" % ())
