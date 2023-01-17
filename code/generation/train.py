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


from arguments import Arguments
from utils import *
from preprocessing import tokenize_func


def train():
    parser = HfArgumentParser(
        (Arguments, Seq2SeqTrainingArguments)
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, max_length=args.max_seq_length)
    # config.label2id = label2id
    # config.id2label = id2label
    # config.num_labels = len(label2id)
    # # setting config for decoder
    # TODO: arg 인자로 넣기.
    # config.decoder_start_token_id = tokenizer.cls_token_id
    print("####### cls token: ", tokenizer.cls_token_id)
    print("####### eos token: ", tokenizer.eos_token_id)
    config.eos_token_id = tokenizer.sep_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.forced_eos_token_id = tokenizer.eos_token_id
    config.min_length = 1
    config.max_length = 128 # TODO: args화
    config.no_repeat_ngram_size = 2
    config.early_stopping = True
    config.length_penalty = 0.0
    config.num_labels=1
    config.num_beams = 4

    print("####### config: ", config)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path, config=config)
    ### TODO: rouge score가 0점으로 나오고 있음. compute_metrics를 BLUE로 바꾸거나 rouge score 고치기.
    rouge = datasets.load_metric("rouge")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
            pred_ids = pred_ids.argmax(-1)
        print('pred_ids: ', pred_ids[:1])
        print('###pred_len: ', len(pred_ids[0]), len(pred_ids[1]))
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        print('pred_str: ', pred_str[:10])
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        print("label_str: ", label_str[:1])
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
        print('### rouge_output: ', rouge_output)
        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    # 데이터셋
    train_dataset = datasets.load_dataset('csv', data_files=args.train_data, split='train')
    eval_dataset = datasets.load_dataset('csv', data_files=args.eval_data, split='train')

    # 필요없는 칼럼을 삭제합니다.
    train_dataset.remove_columns(['diary'])
    eval_dataset.remove_columns(['diary'])

    # 데이터셋을 전처리합니다.
    prepro_fn = partial(tokenize_func, tokenizer=tokenizer, max_input_length=512, max_target_length=128) # TODO: max_len args화
    tokenized_train_dataset = train_dataset.map(prepro_fn,
                                                batched=True,
                                                )
    tokenized_eval_dataset = eval_dataset.map(prepro_fn,
                                                batched=True,
                                                )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None, model = model
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,               
        train_dataset=tokenized_train_dataset,        
        eval_dataset=tokenized_eval_dataset,          
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
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

if __name__ == "__main__":
    train()