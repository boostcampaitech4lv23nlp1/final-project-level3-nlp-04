import torch
from torch.utils.data import Dataset
import numpy as np

def tokenize_func(df, tokenizer, max_input_length, max_target_length):
    inputs = [diary.strip() for diary in df['diary'] if diary != None]

    model_inputs = tokenizer(inputs, max_length = max_input_length, truncation=True)
    # Temporarily sets the tokenizer for encoding the targets. 
    # Useful for tokenizer associated to sequence-to-sequence models that need a slightly different processing for the labels.
    with tokenizer.as_target_tokenizer():
        labels = [str(comment).replace('\n', ' ').strip() for comment in df['comment'] if comment != None]
        labels = tokenizer(labels, max_length = max_target_length, truncation=True)
        labels["input_ids"] = [x + [tokenizer.eos_token_id] for x in labels["input_ids"]]
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
