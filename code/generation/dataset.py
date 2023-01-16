import torch
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.tokenized_sentence = None
        self.tokenize()
        self.max_seq_len = 512
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token

    def tokenize(self):
        self.tokenized_sentence = self.tokenizer(
            list(self.df['long_diary_split']),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.tokenized_comment = self.tokenizer(
            list(self.df['comment']),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    def tokens_to_ids(self, tokens):
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        if len(input_ids) < self.max_seq_len:
            input_ids += [self.tokenizer.pad_token_id]*(self.max_seq_len - len(input_ids))
            attention_mask += [0] * (self.max_seq_len - len(attention_mask))
        else:
            input_ids = input_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
            
        return input_ids, attention_mask
    
    def __getitem__(self, index):
        # inputs = dict()
        # inputs['input_ids'] = self.tokenized_sentence['input_ids'][index]

        # labels = self.tokenized_comment

        # labels["input_ids"] = [
        #         [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]
        # inputs["labels"] = labels['input_ids'][index]

        # return inputs
        record = self.df.iloc[index]
        diary, comment = record['long_diary_split'], record['comment']

        diary_tokens = [self.bos_token] + self.tokenizer.tokenize(diary) + [self.eos_token]
        comment_tokens = [self.bos_token] + self.tokenizer.tokenize(comment) + [self.eos_token]
        encoder_input_ids, encoder_attention_mask = self.tokens_to_ids(diary_tokens)
        decoder_input_ids, decoder_attention_mask = self.tokens_to_ids(comment_tokens)

        labels = self.tokenizer.convert_tokens_to_ids(comment_tokens[1:self.max_seq_len + 1])
        if len(labels) < self.max_seq_len:
            labels += [-100] * (self.max_seq_len - len(labels))

        return {'input_ids': np.array(encoder_input_ids, dtype=np.int_),
                        'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                        'decoder_input_ids': np.array(decoder_input_ids, dtype=np.int_),
                        'decoder_attention_mask': np.array(decoder_attention_mask, dtype=np.float_),
                        'labels': np.array(labels, dtype=np.int_)}
    def __len__(self):
        return len(self.df)
