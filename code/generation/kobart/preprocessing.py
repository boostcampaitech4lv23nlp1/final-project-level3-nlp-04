import torch
from torch.utils.data import Dataset
import numpy as np
import re
from hanspell import spell_checker
from soynlp.normalizer import emoticon_normalize
from kss import split_sentences
from tqdm import tqdm

# 맞춤법을 교정합니다.
def spell_check_(array):
    def spell_check_and_correct(sentences):
        result = []
        
        for i in sentences:
            try:
                result.append(spell_checker.check(i).checked)
            except:
                result.append(i)

        return " ".join(result)
    return [spell_check_and_correct(split_sentences(i)) for i in tqdm(array)]

def spacing_spell_check_and_correct(array):
    array = [emoticon_normalize(x, num_repeats=2) for x in array] # nomalize 먼저
    result = spell_check_(array) # 맞춤법 교정
    return result    

# tokenizing
def tokenize_func(df, tokenizer, max_input_length, max_target_length):
    # 띄어쓰기 및 맞춤법 교정을 수행합니다.
    inputs = [diary for diary in df['diary'] if diary != None]
    inputs = spacing_spell_check_and_correct(inputs)
    model_inputs = tokenizer(inputs, max_length = max_input_length, truncation=True)
    # Temporarily sets the tokenizer for encoding the targets. 
    # Useful for tokenizer associated to sequence-to-sequence models that need a slightly different processing for the labels.
    with tokenizer.as_target_tokenizer():
        labels = [str(comment).replace('\n', ' ').strip() for comment in df['comment'] if comment != None]
        # 띄어쓰기 및 맞춤법 교정을 수행합니다.
        labels = spacing_spell_check_and_correct(labels)
        labels = tokenizer(labels, max_length = max_target_length, truncation=True)
        labels["input_ids"] = [x + [tokenizer.eos_token_id] for x in labels["input_ids"]]
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
