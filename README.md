# GWLAN

This repo contains the codes for **G**eneral **W**ord-**L**evel **A**utocompletio**N** (GWLAN) task.

Since there is no open source code for **GWLAN**, I reproduced the codes.

The GWLAN paper can be read in [here](https://arxiv.org/abs/2105.14913).

You can get the GWLAN paper data from [here](https://github.com/ghrua/gwlan) and you also can use yourself data.

Next you need use the [scripts](https://github.com/lemaoliu/WLAC) to deal the data.


The data samples is as follows:
```json
{
    "src":"The Security Council ,",
    "context_type":"zero_context",
    "left_context":"",
    "right_context":"",
    "typed_seq":"a",
    "target":"安全"
}
{
    "src":"安全 理事会 ，",
    "context_type":"prefix",
    "left_context":"The Security",
    "right_context":"",
    "typed_seq":"Coun",
    "target":"Council"
}
```

You can start training with the following command --dl is the number of gpus.
```
python -m torch.distributed.launch --nproc_per_node 2 train.py --dl 2,3
```
