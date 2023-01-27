#!/bin/bash
lp=5.0
batch=128
lr_array=(3e-5 4e-5 5e-5 6e-5 1e-4)
min_len=30
temperature=1.0
num_beams=5

for lr in "${lr_array[@]}"
do
    python train.py \
    --output_dir ./models/kobart_${batch}_${lr}_datav2_min${min_len}_lp${lp}_temperature${temperature} \
    --overwrite_output_dir \
    --model_name_or_path gogamza/kobart-base-v2 \
    --per_device_train_batch_size ${batch} \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --learning_rate ${lr} \
    --eval_steps 5000 \
    --logging_steps 100 \
    --save_steps=10000 \
    --save_total_limit=1 \
    --predict_with_generate=True \
    --do_sample \
    --num_beams ${num_beams} \
    --temperature ${temperature} \
    --top_k 50 \
    --top_p 0.92 \
    --min_target_length ${min_len} \
    --max_seq_length 512 \
    --max_target_length 128 \
    --length_penalty ${lp} \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --wandb_entity sajo-tuna \
    --wandb_project generation_test \
    --wandb_name kobart_${batch}_${lr}_datav2_min${min_len}_lp${lp}_temperature${temperature} \
    --push_to_hub True \
    --push_to_hub_model_id kobart_${batch}_${lr}_datav2_min${min_len}_lp${lp}_temperature${temperature} \
    --push_to_hub_organization nlp04

done

