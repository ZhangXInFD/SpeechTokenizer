
CONFIG="config/spt_base_cfg.json"


# NPROC_PER_NODE=4
# CUDA_VISIBLE_DEVICES=1,2,6,7 torchrun \
#     --nnode 1 \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master_port 50025  \
# train_example.py \
#     --config ${CONFIG} \

CUDA_VISIBLE_DEVICES=1,2,6,7 accelerate launch scripts/train_example.py\
    --config ${CONFIG}\
    --continue_train