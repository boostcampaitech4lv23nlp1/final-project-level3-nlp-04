import torch
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from hanspell import spell_checker
from soynlp.normalizer import emoticon_normalize
from quickspacer import Spacer

spacer = Spacer()

# 맞춤법을 교정합니다.

@st.cache(allow_output_mutation=True)
class CommentGeneration():
    def __init__(self, ckpt_name):
        self.ckpt_name = ckpt_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_name)
        self.max_length = 512
        self.max_target_length = 128
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.ckpt_name, max_length=self.max_length)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def comment_generation(self, text):
        """text에 대해 감정 레이블과 확률값을 반환하는 함수

        Args:
            text (str): input 일기 데이터

        Returns:
            comment : 생성된 코맨트
        """
        # 데이터셋 전처리
        inputs = self.spell_check_and_spacing([text])
        inputs = self.tokenizer(inputs, max_length=self.max_length, return_tensors='pt', truncation=True).to(self.device)
        input_ids = inputs["input_ids"]
        model_outputs = self.model.generate(input_ids, eos_token_id=self.tokenizer.eos_token_id, max_length=self.max_target_length, min_length = 60, num_beams=5,temperature=0.7)
        comment = self.tokenizer.decode(model_outputs[0],skip_special_tokens=True)
        #print(comment)
        return comment
    
    def spell_check_and_spacing(self, array):
        # 띄어쓰기를 교정합니다.
        array = spacer.space(array, batch_size=64)
        
        # 맞춤법을 교정합니다.
        array = self.spell_check(array)
        # 문장별로 # '맞앜ㅋㅋ' -> '맞아 ㅋㅋㅋ' 와 같이 정규화합니다. 
        result = [emoticon_normalize(x, num_repeats=2) for x in array]
        return result
    
    def spell_check(self, array):
        try:
            spell_checked = spell_checker.check(array).as_dict()['checked']
        except:
            spell_checked = array

        return spell_checked

if __name__ == '__main__':
    preds = CommentGeneration("./models/total_comment_emotion")
    sentence = input('문장 입력 : ')
    while (sentence != ''):
        preds.comment_generation(sentence)
        sentence = input('문장 입력 : ')
