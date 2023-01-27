import torch
from torch.utils.data import Dataset
import numpy as np
import re
from hanspell import spell_checker
from soynlp.normalizer import emoticon_normalize
from quickspacer import Spacer

spacer = Spacer()

# 맞춤법을 교정합니다.
def spell_check(array):
    try:
        spell_checked = spell_checker.check(array).as_dict()['checked']
    except:
        spell_checked = array

    return spell_checked

def spell_check_and_spacing(array):
    # 띄어쓰기를 교정합니다.
    array = spacer.space(array, batch_size=128)
    
    # 맞춤법을 교정합니다.
    array = spell_check(array)
    # 문장별로 # '맞앜ㅋㅋ' -> '맞아 ㅋㅋㅋ' 와 같이 정규화합니다. 
    result = [emoticon_normalize(x, num_repeats=2) for x in array]
    return result


# tokenizing
def tokenize_func(df, tokenizer, max_input_length, max_target_length):
    # 띄어쓰기 및 맞춤법 교정을 수행합니다.
    inputs = [diary for diary in df['diary']]
    # inputs = spell_check_and_spacing(inputs)
    model_inputs = tokenizer(inputs, max_length = max_input_length, truncation=True)
    # Temporarily sets the tokenizer for encoding the targets. 
    # Useful for tokenizer associated to sequence-to-sequence models that need a slightly different processing for the labels.
    with tokenizer.as_target_tokenizer():
        labels = [str(comment).replace('\n', ' ').strip() for comment in df['comment']]
        # 띄어쓰기 및 맞춤법 교정을 수행합니다.
        # labels = spell_check_and_spacing(labels)
        labels = tokenizer(labels, max_length = max_target_length-1, truncation=True)
        labels["input_ids"] = [x + [tokenizer.eos_token_id] for x in labels["input_ids"]]
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
