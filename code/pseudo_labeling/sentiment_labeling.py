from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class SentimentAnalysis():
    def __init__(self, ckpt_name):
        self.ckpt_name = ckpt_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.ckpt_name)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def sentiment_analysis(self, text, return_prob=False, return_all=False):
        """text에 대해 감정 레이블과 확률값을 반환하는 함수

        Args:
            text (str): 감정을 판별하고자하는 text
            return_prob (bool, optional): 확률값을 반환할지. Defaults to False.
            return_all (bool, optional): 모든 클래스에 대해 확률값을 반환할지. Defaults to False.

        Returns:
            default: label(str)
            return_prob: label(str), score(float)
            return_all: Dict(label: score)
        """
        
        # model.config.id2label 에 dict 저장. 
        
        model_outputs = self.model(**self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length').to(self.device))
        outputs = model_outputs["logits"][0]
        outputs = outputs.cpu().detach().numpy()
        
        if self.model.num_labels == 2: # binary
            scores = sigmoid(outputs)
        else: # multiclass
            scores = softmax(outputs)
        
        if self.ckpt_name == 'matthewburke/korean_sentiment':
            self.model.config.id2label = {0: '부정', 1: '긍정'}
        
        label = self.model.config.id2label[scores.argmax().item()]
        score = scores.max().item()
        
        if return_all:
            return {self.model.config.id2label[label]: scores[label] for label in self.model.config.id2label.keys()}
        elif return_prob:
            return label, score
        else:
            return label



if __name__ == '__main__':
    
    # huggingface repo or local path
    sa = SentimentAnalysis("nlp04/korean_sentiment_analysis_dataset3_best")
    
    tqdm.pandas()
    
    # ../../data/ 에 있는 파일들에게 pseudo_labeling 적용해서 ../../data/pseudo_labeled 폴더에 저장
    for filename in ['비출판물_정제.csv']:
        df = pd.read_csv('../../data/'+filename)
        
        
        df['pseudo_label'] = df['diary'].progress_apply(sa.sentiment_analysis)
        
        
        df.to_csv('../../data/pseudo_labeled/'+filename)