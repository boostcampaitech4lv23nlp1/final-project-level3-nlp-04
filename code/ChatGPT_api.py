import openai
from tqdm import tqdm
import pandas as pd

# put your api key
openai.api_key = ""
model_engine = "text-davinci-003"
# put your data path
path = './hb_diary_data.csv'

def chatgpt(prompt):
    # Generate a response
    completion = openai.Completion.create(
        engine = model_engine,
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature = 0.5
    )

    response = completion.choices[0].text

    return response

## main
data = pd.read_csv(path)
diary = data['long_diary_split']
response_list = []

for i in tqdm(range(len(diary))):
    text = diary[i]
    response = chatgpt(f'이 일기에 코멘트를 달아줘.{text}')
    response_list.append(response)

data['comment'] = response_list

# export to csv file
data.to_csv('./comment_data.csv')
