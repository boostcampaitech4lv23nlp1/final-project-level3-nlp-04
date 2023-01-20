import torch
from torch.utils.data import Dataset
import numpy as np


def tokenize_func(df, tokenizer, max_input_length, max_target_length):
    inputs = [diary.strip() for diary in df['long_diary_split']]

    model_inputs = tokenizer(inputs, max_length = max_input_length, truncation=True)
    # Temporarily sets the tokenizer for encoding the targets. 
    # Useful for tokenizer associated to sequence-to-sequence models that need a slightly different processing for the labels.
    with tokenizer.as_target_tokenizer():
        labels = [str(comment).strip() for comment in df['comment']]
        labels = tokenizer(labels, max_length=max_target_length-1, truncation=True)
        labels["input_ids"] = [x.append(tokenizer.eos_token_id) for x in labels["input_ids"]]
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
