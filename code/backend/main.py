from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI

# import config

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


diaries = []


class Diary(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    diary_content: str
    emotions: List[str] = Field(default_factory=list)
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)



app = FastAPI()


@app.on_event('startup')
def load_models():
    
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
    from sentiment_analysis.predict import SentimentAnalysis
    from generation.predict import CommentGeneration
    
    global clf_model, cmt_model
    
    print("start loading model")
    clf_model = SentimentAnalysis('JunHyung1206/kote_sentiment_roberta_large')
    cmt_model = CommentGeneration('lcw99/t5-base-korean-paraphrase')
    
    print("successfully loaded!")
    


@app.get('/')
def read_root():
    return {"message": "FastAPI"}

@app.post("/diary", description="일기작성")
def make_diary(diary_content: str):
    
    _, _, all_scores = clf_model.sentiment_analysis(diary_content)
    
    all_scores_list = [(emotion, all_scores[emotion]) for emotion in all_scores.keys()]
    all_scores_list.sort(key=lambda x: x[1], reverse=True)
    
    
    top1 = all_scores_list[0][0]
    top1_logit = all_scores_list[0][1]
    top2 = all_scores_list[1][0]
    top2_logit = all_scores_list[1][1]
    
    emotions = [top1]
    
    if top1_logit * 0.9 <= top2_logit:
        emotions.append(top2)
    
    comment = cmt_model.comment_generation(diary_content)
    
    
    diary = Diary(diary_content=diary_content,
                  emotions=emotions,
                  comment=comment)
    
    
    diaries.append(diary)
    
    
    return {'diary': diary_content,
            'emotion': ",".join(emotions),
            'comment': comment,
            'created_at': diary.created_at.strftime('%Y-%m-%d-%H-%M-%S-%f')}


@app.on_event("shutdown")
def save_diaries():
    import pandas as pd
    
    save_df = pd.DataFrame(columns=['id', 'diary', 'emotion', 'comment', 'created_at'])
    
    for diary in diaries:
        save_df = save_df.append({'id': diary.id,
                        'diary': diary.diary_content,
                        'emotion': ", ".join(diary.emotions),
                        'comment': diary.comment,
                        'created_at': diary.created_at.strftime('%Y-%m-%d-%H-%M-%S-%f')}, ignore_index=True)
    
    save_df.to_csv('./log/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.csv', index=False)
       
    

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080)
    

