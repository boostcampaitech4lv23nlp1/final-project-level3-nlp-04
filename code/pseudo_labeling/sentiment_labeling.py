import torch
import numpy as np
import torch.nn.functional as F
import streamlit as st

from transformers import AutoModelForSequenceClassification, AutoTokenizer


@st.cache(allow_output_mutation=True)
class SentimentAnalysis():
    def __init__(self, ckpt_name):
        self.ckpt_name = ckpt_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.ckpt_name)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def sentiment_analysis(self, text):
        """text에 대해 감정 레이블과 확률값을 반환하는 함수

        Args:
            text (str): 감정을 판별하고자하는 text

        Returns:
            default: label(str)
            return_prob: label(str), score(float)
            return_all: Dict(label: score)
        """

        model_outputs = self.model(
            **self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length').to(self.device))

        outputs = model_outputs["logits"][0]
        scores = F.softmax(outputs, dim=0).cpu().detach().numpy()

        label = self.model.config.id2label[scores.argmax().item()]
        score = scores.max().item()
        all_score = {self.model.config.id2label[label]: scores[label]
                     for label in self.model.config.id2label.keys()}
        all_score = dict(
            sorted(all_score.items(), key=lambda all_score: -all_score[1]))

        return label, score, all_score


if __name__ == '__main__':
    preds = SentimentAnalysis("nlp04/korean_sentiment_analysis_dataset3")
    sentence = input('문장 입력 : ')
    while (sentence != ''):
        label, score, all_score = preds.sentiment_analysis(sentence)
        for k in all_score.keys():

            print(f"{k}: {all_score[k]:.3f}")
        print('---'*10, '\n'*3)

        sentence = input('문장 입력 : ')
