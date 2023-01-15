python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--fp16 \
--overwrite_output_dir \
--train_data "./emotion_all_20_train.csv" \
--eval_data "./emotion_all_20_eval.csv" \
--model_name_or_path "klue/roberta-large" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_train_epochs 1 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 2e-5 \
--eval_steps 500 \
--logging_steps 500 \
--load_best_model_at_end \
--evaluation_strategy "steps" \
--wandb_entity \
--wandb_project "test" \
--wandb_name "test"
# push_to_hub 관련 인자
# --push_to_hub True \
# --push_to_hub_model_id 'korean_sentiment_analysis_dataset3_best' \
# --push_to_hub_organization 'nlp04'