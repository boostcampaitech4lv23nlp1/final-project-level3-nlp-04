import os
import torch
import sys
import wandb
import logging
import datasets

from transformers import (
    AutoConfig,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint

from arguments import *
from utils import *
from dataset import KoGPTDataset


def train():
    parser = HfArgumentParser(
        (Arguments, ConfigArguments, TrainingArguments)
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

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name_or_path,
                                                        eos_token='</s>',
                                                        pad_token='<pad>',
                                                        unk_token='<unk>',
                                                        sep_token='<sep>',
                                                        )

    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        unk_token_id=tokenizer.unk_token_id,
                                        sep_token_id=tokenizer.sep_token_id,
                                        )

    model = get_model_func(config, args, config_args)
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 데이터셋
    dataset = datasets.load_dataset("nlp04/preprocessed_diary_dataset")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    train_dataset.shuffle(training_args.seed)
    eval_dataset.shuffle(training_args.seed)

    tokenized_train_dataset = KoGPTDataset(train_dataset, tokenizer, max_len=args.max_seq_length)
    tokenized_eval_dataset = KoGPTDataset(eval_dataset, tokenizer, max_len=args.max_seq_length)

    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # last checkpoint 찾기
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Training
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(args.model_name_or_path):
        checkpoint = args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  

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
            logger.info(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")

    # State 저장
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )

    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == '__main__':
    train()