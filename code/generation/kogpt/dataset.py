import numpy as np
import pandas as pd
from torch.utils.data import Dataset

MASK = '<mask>'
COMMENT = '<unused1>'
BOS = '</s>'
EOS = '</s>'
UNK = '<unk>'
PAD = '<pad>'

class KoGPTDataset(Dataset):
    def __init__(self, file, tok, max_len,
                 bos_token=BOS, eos_token=EOS,
                 pad_token=PAD, mask_token=MASK,
                 comment_token = COMMENT,
                 ignore_index = -100,
                 prompt_length = 0
                ):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.docs = pd.read_csv(file, encoding='utf-8')
        self.len = self.docs.shape[0]
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.comment_token = comment_token
        self.ignore_index = ignore_index
        self.prompt_length = prompt_length

    def add_padding_data(self, inputs, pad_index):
        if len(inputs) < self.max_len:
            pad = [pad_index] *(self.max_len - len(inputs))
            inputs = inputs + pad
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        
        diary = self.tok.encode(self.bos_token) + self.tok.encode(instance['diary'])
        len_diary = len(diary)
        
        comment = self.tok.encode(self.comment_token) + self.tok.encode(instance['comment']) + self.tok.encode(self.eos_token)
        len_comment = len(comment)
        context = diary + comment

        if len(context) > self.max_len:
            additional_len = len(context) - self.max_len
            diary = diary[:-additional_len]
            len_diary = len(diary)
            context = diary + comment

        labels = [-100] * len_diary + comment[1:]
        mask = [0] * len_diary + [1] * len_comment + [0] * (self.max_len - len_diary - len_comment)

        if len(context) < self.max_len:
            context = self.add_padding_data(context, self.tok.pad_token_id)

        if len(labels) < self.max_len:
            labels = self.add_padding_data(labels, -100)

        return {'input_ids': np.array(context, dtype=np.int_),
                'attention_mask': np.array(mask, dtype=np.int_),
                'labels': np.array(labels, dtype=np.int_)}

    def __len__(self):
        return self.len