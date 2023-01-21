import re
import os
import pandas as pd
import numpy as np
import json

root_path = '../../data/'

corpus_name = "aihub"
corpus_name_train = "aihub_Training"
corpus_name_val = "aihub_Validation"

with open(f'{root_path+corpus_name_train}.json', 'r', encoding='utf8') as f:
    train_dataset = json.load(f)
with open(f'{root_path+corpus_name_val}.json', 'r', encoding='utf8') as f:
    validation_dataset = json.load(f)

total_dataset = train_dataset + validation_dataset


def sentence_split(sentences):
    sentences = list(
        filter(lambda x: x, [i.strip() for i in re.split(r'([.?!])', sentences)]))
    respon = [
        "응.", "맞아.", "고마워.", "그래.", "아니.", "응 맞아.", "응 많이.", "아니야.", "그런가.", "그런가?", "그렇긴 하지.", "그런 것 같아.", "그러네.",
        "아니 그건 아니야.", "아니 없을 거야.", "아.", "그렇군요.", "그러시군요.", "그랬군요.", "네 그러시군요.", "네.", "그러셨군요.", "아하.", "그러네요.",
        "그러니까 말이야.", "그러게나 말이야.", "그러게 말이야.", "그랬군요.", "모르겠어.", "네 그러셨군요.", "그런가 봐.", "응 고마워.","어.","정말.","그렇지.","그럼요.","그치.", "그래요.","몰라.",
        "좋아요.","그러려나.","웅.","글쎄.","없어.","좋아.", "맞다.", "아니 없어.", "좋지.", "그럼."
    ]
    is_question = False
    result = []
    if sentences:
        is_question = sentences[-1] == '?'
        i = 0
        punct = ['.', '?', '!']

        temp = ""
        while (i < len(sentences)):
            if sentences[i] not in punct:
                temp += sentences[i]
            else:
                temp += sentences[i]

                if i+1 < len(sentences) and sentences[i+1] not in punct:
                    if temp.strip() not in respon:
                        result.append(temp)
                    temp = ""
            i += 1
        if temp.strip() not in respon:
            result.append(temp)
    return result, is_question


def return_text_comment(dataset):
    text = ""
    comment = ""

    HS01, _ = sentence_split(dataset['talk']['content']['HS01'])
    SS01, SS01_q = sentence_split(dataset['talk']['content']['SS01'])
    HS02, _ = sentence_split(dataset['talk']['content']['HS02'])
    SS02, SS02_q = sentence_split(dataset['talk']['content']['SS02'])
    HS03, _ = sentence_split(dataset['talk']['content']['HS03'])
    SS03, _ = sentence_split(dataset['talk']['content']['SS03'])

    text += ' '.join(HS01) + ' '
    if SS01_q:
        SS01 = SS01[0:-1]
    comment += ' '.join(SS01) + ' '

    text += ' '.join(HS02) + ' '
    if SS02_q:
        SS02 = SS02[0:-1]
    comment += ' '.join(SS02) + ' '

    text += ' '.join(HS03) + ' '
    comment += ' '.join(SS03) + ' '

    text = text.strip()
    comment = comment.strip()

    return text, comment


id = []
diary = []
comment = []
sentiment = []

emotion_dict = {
    "E10": "분노", "E11": "분노/툴툴대는", "E12": "분노/좌절한", "E13": "분노/짜증나는", "E14": "분노/방어적인 ", "E15": "분노/악의적인 ", "E16": "분노/안달하는", "E17": "분노/구역질 나는", "E18": "분노/노여워하는", "E19": "분노/성가신",
    "E20": "슬픔", "E21": "슬픔/실망한", "E22": "슬픔/비통한", "E23": "슬픔/후회되는", "E24": "슬픔/우울한 ", "E25": "슬픔/마비된 ", "E26": "슬픔/염세적인", "E27": "슬픔/눈물이 나는", "E28": "슬픔/낙담한", "E29": "슬픔/환멸을 느끼는",
    "E30": "불안", "E31": "불안/두려운", "E32": "불안/스트레스 받는", "E33": "불안/취약한", "E34": "불안/혼란스러운 ", "E35": "불안/당혹스러운 ", "E36": "불안/회의적인", "E37": "불안/걱정스러운", "E38": "불안/조심스러운", "E39": "불안/초조한",
    "E40": "상처", "E41": "상처/질투하는", "E42": "상처/배신당한", "E43": "상처/고립된", "E44": "상처/충격 받은 ", "E45": "상처/불우한 ", "E46": "상처/희생된", "E47": "상처/억울한", "E48": "상처/괴로워하는", "E49": "상처/버려진",
    "E50": "당황", "E51": "당황/고립된", "E52": "당황/남의 시선 의식하는", "E53": "당황/외로운", "E54": "당황/열등감 ", "E55": "당황/죄책감 ", "E56": "당황/부끄러운", "E57": "당황/혐오스러운", "E58": "당황/한심한", "E59": "당황/혼란스러운",
    "E60": "기쁨", "E61": "기쁨/감사하는", "E62": "기쁨/사랑하는", "E63": "기쁨/편안한", "E64": "기쁨/만족스러운 ", "E65": "기쁨/흥분되는 ", "E66": "기쁨/느긋한", "E67": "기쁨/안도하는", "E68": "기쁨/신이 난", "E69": "기쁨/자신하는",
}


for i in total_dataset:
    id.append(i['talk']['id']['talk-id'])
    t, c = return_text_comment(i)
    diary.append(t)
    comment.append(c)
    sentiment.append(emotion_dict[i['profile']['emotion']['type']])

diary_source = ["ai_hub_018"] * len(id)

df = pd.DataFrame({'id': id, 'diary': diary, 'comment': comment,
                  'emotion': sentiment, 'source': diary_source})
df = df.dropna(axis=0, how='any')
df = df.sort_values(by=['id'])

df.to_csv(f'{root_path}aihub_감성대화_수정2.csv', encoding="utf8", index=False)
