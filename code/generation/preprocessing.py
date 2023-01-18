import torch
from torch.utils.data import Dataset
import numpy as np

# class Dataset(Dataset):
#     def __init__(self, df, tokenizer, config):
#         self.df = df
#         self.tokenizer = tokenizer
#         self.config = config
#         self.tokenized_sentence = None
#         self.tokenize()
#         self.max_input_length = config.max_input_length
#         self.max_target_length = config.max_target_length
#         self.bos_token = self.tokenizer.bos_token
#         self.eos_token = self.tokenizer.eos_token

def tokenize_func(df, tokenizer, max_input_length, max_target_length):
    inputs = [tokenizer.bos_token + ' ' + diary.strip() + ' ' + tokenizer.eos_token for diary in df['long_diary_split']]
    # inputs = ["summarize_summary:" + " " + diary.strip() + " " + tokenizer.eos_token for diary in df['long_diary_split']] # for T5

    model_inputs = tokenizer(inputs, max_length = max_input_length, truncation=True)
    # Temporarily sets the tokenizer for encoding the targets. 
    # Useful for tokenizer associated to sequence-to-sequence models that need a slightly different processing for the labels.
    with tokenizer.as_target_tokenizer():
        labels = [tokenizer.bos_token + ' ' + str(comment).strip() + ' ' + tokenizer.eos_token for comment in df['comment']]
        labels = tokenizer(labels, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
    
    # def __getitem__(self, index):

    #     line = self.df.iloc[index]
    #     # 전처리: '\n', 및 좌우 공백 삭제.
    #     diary, comment = line['long_diary_split'].replace('\n', '').strip(), line['comment'].replace('\n', '').strip()

    #     diary_tokens = [self.bos_token] + self.tokenizer.tokenize(diary) + [self.eos_token]
    #     comment_tokens = [self.bos_token] + self.tokenizer.tokenize(comment) + [self.eos_token]
    #     encoder_input_ids, encoder_attention_mask = self.tokens_to_ids(diary_tokens)
    #     decoder_input_ids, decoder_attention_mask = self.tokens_to_ids(comment_tokens)

    #     labels = self.tokenizer.convert_tokens_to_ids(comment_tokens[1:self.max_seq_len + 1])
    #     if len(labels) < self.max_seq_len:
    #         labels += [-100] * (self.max_seq_len - len(labels))

    #     return {'input_ids': np.array(encoder_input_ids, dtype=np.int_),
    #                     'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
    #                     'decoder_input_ids': np.array(decoder_input_ids, dtype=np.int_),
    #                     'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
    #                     'labels': np.array(labels, dtype=np.int_)}
    # def __len__(self):
    #     return len(self.df)
