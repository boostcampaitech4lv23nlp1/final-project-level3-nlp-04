import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
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

from arguments import Arguments
from utils import *
from dataset import Dataset

def inference():
    parser = HfArgumentParser(
        (Arguments, TrainingArguments)
    )
    logger = logging.getLogger(__name__)

    args, training_args = parser.parse_args_into_dataclasses()

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
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config)
    

    # 데이터셋
    eval_df = pd.read_csv(args.eval_data)
    if 'long_diary_split' in eval_df.columns:
        eval_df.rename(columns = {'long_diary_split' : 'text'}, inplace = True)
    
    eval_dataset = Dataset(eval_df, tokenizer, config)
    

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,               
        train_dataset=None,        
        eval_dataset=eval_dataset,          
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    predictions = trainer.predict(test_dataset=eval_dataset)
    
    probs = F.softmax(torch.FloatTensor(predictions.predictions), dim=1)
    preds = probs.argmax(1)
    
    preds_labels = [config.id2label[i] for i in probs.argmax(1).tolist()]  # 예측한 labels
    logits = torch.gather(probs, 1, preds.unsqueeze(1)).squeeze() # 예측한 labels에 대한 확률
    
    predict_df = eval_df.copy()
    
    predict_df['predict'] = preds_labels
    predict_df['logits'] = logits
    # predict_df['probs'] = probs.tolist()
    
    all_score_list = []
    for i in probs.cpu().detach().numpy():
        score = i
        all_score = {model.config.id2label[label]: score[label]
                     for label in model.config.id2label.keys()}
        all_score = dict(
            sorted(all_score.items(), key=lambda all_score: -all_score[1]))
        all_score_list.append(all_score)
    predict_df['all_score'] = all_score_list  
    
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    
    predict_df.to_csv(training_args.output_dir + '/predict.csv', index=False)

if __name__ == "__main__":
    inference()