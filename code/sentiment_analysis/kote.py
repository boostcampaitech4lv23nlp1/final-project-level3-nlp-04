from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

MAP = {'불평/불만': '짜증남',
 '환영/호의': '즐거운(신나는)',
 '감동/감탄': '놀란',
 '지긋지긋': '짜증남',
 '고마움': '고마운',
 '슬픔': '슬픔(우울한)',
 '화남/분노': '짜증남',
 '존경': '놀란',
 '기대감': '설레는(기대하는)',
 '우쭐댐/무시함': '기타(우쭐댐/무시함)',
 '안타까움/실망': '슬픔(우울한)',
 '비장함': '기타(비장함)',
 '의심/불신': '기타(의심/불신)',
 '뿌듯함': '뿌듯한',
 '편안/쾌적': '일상적인',
 '신기함/관심': '놀란',
 '아껴주는': '사랑하는',
 '부끄러움': '후회하는',
 '공포/무서움': '걱정스러운(불안한)',
 '절망': '슬픔(우울한)',
 '한심함': '짜증남',
 '역겨움/징그러움': '짜증남',
 '짜증': '짜증남',
 '어이없음': '짜증남',
 '없음': '일상적인',
 '패배/자기혐오': '슬픔(우울한)',
 '귀찮음': '짜증남',
 '힘듦/지침': '힘듦(지침)',
 '즐거움/신남': '즐거운(신나는)',
 '깨달음': '다짐하는',
 '죄책감': '후회하는',
 '증오/혐오': '짜증남',
 '흐뭇함(귀여움/예쁨)': '사랑하는',
 '당황/난처': '놀란',
 '경악': '짜증남',
 '부담/안_내킴': '짜증남',
 '서러움': '슬픔(우울한)',
 '재미없음': '짜증남',
 '불쌍함/연민': '슬픔(우울한)',
 '놀람': '놀란',
 '행복': '기쁨(행복한)',
 '불안/걱정': '걱정스러운(불안한)',
 '기쁨': '기쁨(행복한)',
 '안심/신뢰': '일상적인'}

model_name = "searle-j/kote_for_easygoing_people"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation='max_length', padding='max_length')

pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0, # gpu number, -1 if cpu used
        return_all_scores=True,
        function_to_apply='sigmoid'
    )

def top_k_emotions(text, threshold:float = 0.3, k:int = 1):
    if not 0 <= threshold <=1:
        raise ValueError("theshold must be a float b/w 0 ~ 1.")

    cur_result = {}
    for out in pipe(text, padding=True, truncation=True)[0]:
        if out["score"] > threshold:
            cur_result[out["label"]] = round(out["score"], 2)
    cur_result = sorted(cur_result.items(), key=lambda x: x[1], reverse=True)
    
    return pd.Series([MAP[cur_result[0][0]], cur_result[0][1]])



def sentiment_analysis(file_path):
    
    df = pd.read_csv(file_path)
    
    df[['predict', 'score']] = df['text'].progress_apply(top_k_emotions)
    
    df.to_csv('./outputs/kote_predict.csv')


if __name__ == "__main__":
    sentiment_analysis('./data/emotion_all.csv')