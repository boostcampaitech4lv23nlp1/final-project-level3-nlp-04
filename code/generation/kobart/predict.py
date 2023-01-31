import torch
# import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from hanspell import spell_checker
from soynlp.normalizer import emoticon_normalize
from quickspacer import Spacer
import re
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from kss import split_sentences
import numpy as np
import warnings
spacer = Spacer()

# 맞춤법을 교정합니다.

# @st.cache(allow_output_mutation=True)
class CommentGeneration():
    def __init__(self, ckpt_name):
        self.ckpt_name = ckpt_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_name)
        self.max_length = 512
        self.max_target_length = 128
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.ckpt_name, max_length=self.max_length)
        warnings.filterwarnings(action='ignore')
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
        err_char = re.compile('[^\u0000-\u0100\0x1100-0x11FF\u3131-\u3163\uac00-\ud7a3]+')
        inputs = self.spell_check_and_spacing([text])
        inputs = self.tokenizer(inputs, max_length=self.max_length, return_tensors='pt', truncation=True).to(self.device)
        input_ids = inputs["input_ids"]
        model_outputs = self.model.generate(input_ids, eos_token_id=self.tokenizer.eos_token_id, 
                                            max_length=self.max_target_length, 
                                            min_length = 30, 
                                            num_beams=10, 
                                            do_sample=True,
                                            temperature=1.0,
                                            top_k=50,
                                            top_p=0.92,
                                            length_penalty=0.5,
                                            )
        comment = self.tokenizer.decode(model_outputs[0],skip_special_tokens=True)

        # comment에 대해 맞춤법, 띄어쓰기 교정을 수행합니다.
        comment = self.spell_check_and_spacing([comment])[0]
        #print(comment)
        
        while err_char.search(comment):
            model_outputs = self.model.generate(input_ids, eos_token_id=self.tokenizer.eos_token_id, 
                                            max_length=self.max_target_length, 
                                            min_length = 30, 
                                            num_beams=10, 
                                            do_sample=True,
                                            temperature=1.0,
                                            top_k=50,
                                            top_p=0.92,
                                            length_penalty=0.5,
                                            )
            comment = self.tokenizer.decode(model_outputs[0],skip_special_tokens=True)
            comment = self.spell_check_and_spacing([comment])
        return self.post_processing(comment)
    
    def spell_check_and_spacing(self, array):
        # 띄어쓰기를 교정합니다.
        array = spacer.space(array)
        
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

    def post_processing(self, comment):
        sentences = split_sentences(comment)
        tokenizer = Mecab()
        def tokenized(x):
            return [i for i in tokenizer.pos(x) if i[1] not in ["JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC","EP","EF","EC","ETN","ETM"]]
        vectorizer = TfidfVectorizer(tokenizer=tokenized)
        sentences =  sorted(set(sentences), key=lambda x: sentences.index(x)) # 중복된 문장 제거
        del_index = []
        for i,s_i in enumerate(sentences):
            query = s_i
            corpus = [s_j for j,s_j in enumerate(sentences) if i!=j]
            sp_matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([query])
            result = query_vec * sp_matrix.T
            sorted_result = np.argsort(-result.data)
            doc_scores = result.data[sorted_result]
            doc_ids = result.indices[sorted_result]
            for score,ids in zip(doc_scores,doc_ids):
                if score>0.7:
                    del_index.append(ids if i>ids else ids+1)
        
        del_index =  sorted(set(del_index))
        for i in del_index:
            sentences[i]=""
        comment = " ".join(sentences)
        return comment

if __name__ == '__main__':
    preds = CommentGeneration("nlp04/final_bart")
    sentence = input('문장 입력 : ')
    while (sentence != ''):
        preds.comment_generation(sentence)
        sentence = input('문장 입력 : ')
