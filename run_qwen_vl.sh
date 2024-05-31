#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
hostfile=./hostfile
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_qwen_vl
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# export DS_ACCELERATOR=musa
export NCCL_PROTOS=2
# export MUSA_KERNEL_TIMEOUT=1800
# export DS_ENV_FILE=./ds_env

deepspeed pretrain_llama.py \
        --data_path ./data/test_data.json \
        --data_split 2,4,4 \
        --model_name_or_path ../Qwen-VL-Chat \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --max_seq_len 4096 \
        --learning_rate 3.4e-5 \
        --weight_decay 0. \
        --num_train_epochs 1000  \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 100 \
        --seed 1234 \
        --gradient_checkpointing \
        --deepspeed \
        --zero_stage $ZERO_STAGE \
        --print_loss \
        --output_dir $OUTPUT \
        &> $OUTPUT/training_12.log

#--offload \   --force_multi \--hostfile $hostfile \        
        
