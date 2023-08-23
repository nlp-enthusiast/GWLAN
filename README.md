# GWLAN

This repo contains the codes for **G**eneral **W**ord-**L**evel **A**utocompletio**N** (GWLAN) task.
Since there is no open source code for **GWLAN**, I reproduced the codes.

You can get the GWLAN paper data from [here](https://github.com/ghrua/gwlan)

Next you need use the [scripts](https://github.com/lemaoliu/WLAC) to deal the data.

You can start training with the following command --dl is the number of gpus.
```
python -m torch.distributed.launch --nproc_per_node 2 train.py --dl 2,3
```
