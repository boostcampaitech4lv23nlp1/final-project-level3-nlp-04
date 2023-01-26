from torch.utils.data import Dataset
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
    array = spacer.space(array, batch_size=64)
    
    # 맞춤법을 교정합니다.
    array = spell_check(array)
    # 문장별로 # '맞앜ㅋㅋ' -> '맞아 ㅋㅋㅋ' 와 같이 정규화합니다. 
    result = [emoticon_normalize(x, num_repeats=2) for x in array]
    return result


class KoGPTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        super().__init__()
        self.tok = tokenizer
        self.max_len = max_len
        self.diary = spell_check_and_spacing(dataset['diary'])
        self.comment = spell_check_and_spacing(dataset['comment'])
        self.len = len(dataset)
    
    def __getitem__(self, idx):
        context = self.diary[idx] + self.tok.eos_token + self.comment[idx] + self.tok.eos_token

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