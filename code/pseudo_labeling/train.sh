python train.py \
--output_dir ./models/train_dataset \
--fp16 \
--overwrite_output_dir \
--train_data ./data/emotion_all_20_train.csv \
--eval_data ./data/emotion_all_20_eval.csv \
--model_name_or_path klue/roberta-large \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 4 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 2e-5 \
--eval_steps 100 \
--logging_steps 100 \
--save_steps=1000 \
--load_best_model_at_end \
--evaluation_strategy steps \
--wandb_entity \
--wandb_project test \
--wandb_name test
# push_to_hub 관련 인자
# --push_to_hub True \
# --push_to_hub_model_id 'korean_sentiment_analysis_dataset3_best' \
# --push_to_hub_organization 'nlp04'
