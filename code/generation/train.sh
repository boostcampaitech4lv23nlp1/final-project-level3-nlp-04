python train.py \
--output_dir ./models/train_dataset \
--fp16 \
--overwrite_output_dir \
--train_data gamsung_comment.csv \
--eval_data comment_100.csv \
--model_name_or_path gogamza/kobart-base-v2 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 128 \
--num_train_epochs 3 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 5.6e-5 \
--eval_steps 1000 \
--logging_steps 100 \
--save_steps=1000 \
--save_total_limit=3 \
--predict_with_generate=True \
--load_best_model_at_end \
--evaluation_strategy steps \
--wandb_entity sajo-tuna \
--wandb_project generation_test \
--wandb_name generation_test
# push_to_hub 관련 인자
# --push_to_hub True \
# --push_to_hub_model_id 'korean_sentiment_analysis_dataset3_best' \
# --push_to_hub_organization 'nlp04'
# --max_input_length 512 \
# --max_target_length 128 \
# --max_generate_length 64 \