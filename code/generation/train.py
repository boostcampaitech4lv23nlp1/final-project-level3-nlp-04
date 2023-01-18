import os
import sys
import pandas as pd

import torch
import wandb
import logging
import datasets
from functools import partial

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    HfArgumentParser,
    set_seed
)

from transformers.optimization import AdamW, get_cosine_schedule_with_warmup


from arguments import *
from utils import *
from preprocessing import tokenize_func
from get_model import get_model_func


def train():
    parser = HfArgumentParser(
        (Arguments, ConfigArguments, Seq2SeqTrainingArguments)
    )
    logger = logging.getLogger(__name__)

    args, config_args, training_args = parser.parse_args_into_dataclasses()
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, max_length=args.max_seq_length)

    model = get_model_func(config, args, config_args, tokenizer)
    print("####### config: ", config)

    # 데이터셋
    train_dataset = datasets.load_dataset('csv', data_files=args.train_data, split='train')
    eval_dataset = datasets.load_dataset('csv', data_files=args.eval_data, split='train')
    train_dataset.shuffle(training_args.seed)

    # 데이터셋을 전처리합니다.
    prepro_fn = partial(tokenize_func, tokenizer=tokenizer, max_input_length=512, max_target_length=128) # TODO: max_len args화
    tokenized_train_dataset = train_dataset.map(prepro_fn,
                                                batched=True,
                                                )
    tokenized_eval_dataset = eval_dataset.map(prepro_fn,
                                                batched=True,
                                                )
    # collator의 입력 형식에 맞게 wrangling합니다.
    tokenized_train_dataset.remove_columns(train_dataset.column_names)
    tokenized_eval_dataset.remove_columns(eval_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None, model = model
    )

    metric_fn = partial(compute_metrics, tokenizer=tokenizer)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,               
        train_dataset=tokenized_train_dataset,        
        eval_dataset=tokenized_eval_dataset,          
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_fn,
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

    if training_args.push_to_hub:
        trainer.push_to_hub()

    # evaluation
    trainer.evaluate(
        eval_dataset=tokenized_eval_dataset,
        max_length=64,
        num_beams=4
    )

if __name__ == "__main__":
    train()