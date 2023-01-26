import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


class KoGPTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        super().__init__()
        self.tok = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file_path, encoding='utf-8')
        self.len = self.docs.shape[0]
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        context = instance['diary'] + self.tok.eos_token + instance['comment'] + self.tok.eos_token

        batch_encoding = self.tok(
            context,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        inputs = {k: batch_encoding[k] for k in batch_encoding}
        labels = batch_encoding["input_ids"]
        inputs['labels'] = labels

        return inputs

    def __len__(self):
        return self.len