from torch.utils.data import Dataset


class KoGPTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.diary = dataset['diary']
        self.comment = dataset['comment']
        self.len = len(dataset)
    

    def __getitem__(self, idx):
        encoded_diary = self.tokenizer.encode(self.diary[idx])
        encoded_comment = self.tokenizer.encode(self.comment[idx])

        context = encoded_diary  + [self.tokenizer.sep_token_id] + encoded_comment + [self.tokenizer.eos_token_id]

        if len(context) > self.max_len:
            encoded_diary = encoded_diary[:self.max_len-2-len(encoded_comment)]
            context = encoded_diary  + [self.tokenizer.sep_token_id] + encoded_comment + [self.tokenizer.eos_token_id]
        else:
            context += [self.tokenizer.pad_token_id] * (self.max_len - len(context))

        mask = [1] * len(context) + [0] * (self.max_len - len(context))
        
        input = {}
        input['input_ids'] = context
        input['attention_mask'] = mask
        input['labels'] = context
        return input


    def __len__(self):
        return self.len