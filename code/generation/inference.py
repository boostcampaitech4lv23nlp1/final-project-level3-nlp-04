import os
import sys
import torch
import logging
import datasets
from functools import partial
import pandas as pd

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    set_seed
)

from arguments import *
from utils import *
from preprocessing import tokenize_func

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (Arguments, ConfigArguments, Seq2SeqTrainingArguments)
    )    

    args, config_args, training_args = parser.parse_args_into_dataclasses()

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다
    set_seed(training_args.seed)

    dataset = datasets.load_dataset("nlp04/diary_dataset")
    test_dataset = dataset["test"]
    column_names = test_dataset.column_names
 
    # config, tokenizer, model
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        max_length=args.max_seq_length,
    )

    model = get_model_func(config, args, config_args, tokenizer)

    # 데이터셋 전처리
    prepro_fn = partial(tokenize_func, tokenizer=tokenizer, max_input_length=args.max_seq_length, max_target_length=config_args.max_target_length)

    tokenized_test_dataset = test_dataset.map(prepro_fn, batched=True, remove_columns=column_names) 

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None, model = model
    )

    metric_fn = partial(compute_metrics, tokenizer=tokenizer)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,               
        train_dataset=None,        
        eval_dataset=tokenized_test_dataset,          
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_fn,
    )

    if training_args.do_predict:
        logger.info("***** Predict *****")
        
        predict_results = trainer.predict(
            tokenized_test_dataset,
            metric_key_prefix="predict", 
            max_length=config_args.max_target_length, 
            num_beams=config_args.num_beams
        )

        if 'label_ids' in predict_results._fields:
            if predict_results._fields[1] == 'label_ids':
                labels = predict_results[1]
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics = predict_results.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]

                if decoded_labels:
                    # decoded_labels(origin), predictions 비교를 위한 csv 생성
                    result_df = pd.DataFrame(zip(decoded_labels, predictions), columns=['labels', 'preds'])
                    result_df.to_csv(os.path.join(training_args.output_dir, "generated_predictions.csv"), encoding='utf-8-sig')

if __name__ == "__main__":
    main()