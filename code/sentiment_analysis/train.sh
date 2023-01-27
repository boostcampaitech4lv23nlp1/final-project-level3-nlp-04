python train.py \
--output_dir ./models/train_dataset \
--fp16 \
--train_data './data/emotion_ver1_train.csv' \
--eval_data './data/emotion_ver1_test.csv' \
--model_name_or_path klue/roberta-large \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 4 \
--num_train_epochs 5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 1e-5 \
--eval_steps 100 \
--logging_steps 100 \
--save_steps=1000 \
--load_best_model_at_end \
--evaluation_strategy steps \
--wandb_entity sajo-tuna \
--wandb_project roberta_large_emotion \
--wandb_name exp1 \
--push_to_hub True \
--push_to_hub_model_id 'kote_sentiment_roberta_large' \
--push_to_hub_organization 'nlp04'