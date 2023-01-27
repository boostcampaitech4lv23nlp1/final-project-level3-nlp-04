from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI


# import config

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


diaries = dict()


class DiaryIn(BaseModel):
    id: Optional[UUID] = Field(default_factory=uuid4)
    diary_content: str

class DiaryOut(DiaryIn):
    emotions: List[str] = Field(default_factory=list)
    comment: Optional[str] = None
    logits: Optional[Dict] = None
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
    cmt_model = CommentGeneration('nlp04/kobart_8_5.6e-5_min30_lp4_sample')
    
    print("successfully loaded!")
    


@app.get('/')
def read_root():
    return {"message": "FastAPI"}

@app.post("/diary", description="일기작성")
def make_diary(diary_in: DiaryIn):
    
    _, _, all_scores = clf_model.sentiment_analysis(diary_in.diary_content)
    
    all_scores_list = [(emotion, all_scores[emotion]) for emotion in all_scores.keys()]
    all_scores_list.sort(key=lambda x: x[1], reverse=True)
    
    
    top1 = all_scores_list[0][0]
    top1_logit = all_scores_list[0][1]
    top2 = all_scores_list[1][0]
    top2_logit = all_scores_list[1][1]
    
    emotions = [top1]
    
    if top1_logit * 0.9 <= top2_logit:
        emotions.append(top2)
    
    comment = cmt_model.comment_generation(diary_in.diary_content)
    
    
    diary_out = DiaryOut(
                  diary_content=diary_in.diary_content,
                  emotions=emotions,
                  comment=comment,
                  logits = all_scores_list)
    
    diaries[str(diary_out.id)] = diary_out.dict()
    
    return diary_out.dict()


@app.on_event("shutdown")
def save_diaries():
    import json
    
    file_path = './log/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.json'
    
    with open(file_path, 'w') as outfile:
        json.dump(diaries, outfile)
       
    

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080)
    

