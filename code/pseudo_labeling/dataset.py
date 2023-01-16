import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.tokenized_sentence = None
        self.tokenize()

    def tokenize(self):
        self.tokenized_sentence = self.tokenizer(
            list(self.df['text']),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, index):
        inputs = dict()
        inputs['input_ids'] = self.tokenized_sentence['input_ids'][index]
        inputs['attention_mask'] = self.tokenized_sentence['attention_mask'][index]
        inputs['token_type_ids'] = self.tokenized_sentence['token_type_ids'][index]

        if "emotion" in self.df.columns:
            labels = self.config.label2id[self.df["emotion"][index]]
            inputs["labels"] = torch.tensor(labels)

        return inputs

    def __len__(self):
        return len(self.df)
