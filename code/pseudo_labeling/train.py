import os
import sys
import pandas as pd

import torch
import wandb
import logging

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    HfArgumentParser,
    set_seed
)

from transformers.optimization import AdamW, get_cosine_schedule_with_warmup


from arguments import Arguments
from utils import *
from dataset import Dataset

def train():
    parser = HfArgumentParser(
        (Arguments, TrainingArguments)
    )
    logger = logging.getLogger(__name__)

    args, training_args = parser.parse_args_into_dataclasses()
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity, name=args.wandb_name)

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    # config, tokenizer, model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    config.label2id = label2id
    config.id2label = id2label
    config.num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, max_length=args.max_seq_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config)

    # 데이터셋
    train_df = pd.read_csv(args.train_data)
    eval_df = pd.read_csv(args.eval_data)
    
    train_dataset = Dataset(train_df, tokenizer, config)
    eval_dataset = Dataset(eval_df, tokenizer, config)
    

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,               
        train_dataset=train_dataset,        
        eval_dataset=eval_dataset,          
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # last checkpoint 찾기.
    last_checkpoint = get_last_checkpoint(args, training_args)

    # Training
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(args.model_name_or_path):
        checkpoint = args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    output_train_file = os.path.join(
        training_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # State 저장
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )

if __name__ == "__main__":
    train()