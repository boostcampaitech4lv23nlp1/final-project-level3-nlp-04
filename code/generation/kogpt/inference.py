import os
import sys
import torch
import logging
import datasets
import pandas as pd
from tqdm import tqdm
import json

from transformers import (
    AutoConfig,
    PreTrainedTokenizerFast,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

from arguments import *
from utils import *


def inference():
    parser = HfArgumentParser(
        (Arguments, ConfigArguments, TrainingArguments)
    )
    logger = logging.getLogger(__name__)

    args, config_args, training_args = parser.parse_args_into_dataclasses()

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)
    
    set_seed(training_args.seed)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = get_model_func(config, args, config_args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 데이터셋
    dataset = datasets.load_dataset("nlp04/preprocessed_diary_dataset")

    #train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    eval_dataset.shuffle(training_args.seed)

    outputs = []
    for i in tqdm(range(len(eval_dataset))):
        ids = tokenizer.encode(eval_dataset['diary'][i] + tokenizer.sep_token, return_tensors='pt').to(device)
        output = model.generate(
            ids,
            do_sample=True,
            min_length=config.min_length+len(ids[0]),
            max_length=args.max_seq_length+len(ids[0]),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            early_stopping=True,
            bad_words_ids=[[tokenizer.unk_token_id]]
        )
        output = tokenizer.decode(output[0]).split(tokenizer.sep_token)[-1]
        outputs.append(output.replace('</s>', ''))

    result_df = pd.DataFrame(zip(eval_dataset['diary'], eval_dataset['comment'], outputs), columns=['diary', 'labels', 'preds'])

    metrics = compute_metrics(result_df['diary'].tolist(), result_df['preds'].tolist(), result_df['labels'].tolist(), tokenizer)
    print('***** Metrics *****')
    print(metrics)

    prefix = str(config_args.length_penalty) + '_' + str(config_args.min_length) + '_' +  str(args.temperature) + '_' + str(config_args.num_beams)
    # comment 생성 결과 저장
    result_df.to_csv(os.path.join(training_args.output_dir, prefix + "_generated_predictions.csv"), encoding='utf-8-sig')

    # metrics 저장
    metrics = {k:float(v) for k, v in metrics.items()}
    with open(os.path.join(training_args.output_dir, prefix + '_predict_results.json'), 'w') as f:
         json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    inference()