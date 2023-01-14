import zipfile
import os
import glob

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

root_path = '../../data/'

def unzip(source_file, dest_path):
    success = 0
    fail = 0

    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            try:
                success += 1
                member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')
                zf.extract(member, dest_path)
            except:
                fail += 1
                raise Exception('what?!')
                
    return success, fail

def text_preprocessing(content):

    content = re.sub('<p>', '', content)
    content = re.sub('</p>', '\n', content)
    content = re.sub('<s>', '', content)
    content = re.sub('</s>', '\n', content)
    content = re.sub('&lt;', '<', content)
    content = re.sub('&lt', '<', content)
    content = re.sub('&gt;', '>', content)
    content = re.sub('\s+', ' ', content).strip()

    return content

def find_content(tag, raw_corpus):
    start_tag = "<"+tag+".*?>"
    end_tag = "</"+tag+">"

    start_pos = re.search(start_tag, raw_corpus).end()
    end_pos = re.search(end_tag, raw_corpus).start()
    content = raw_corpus[start_pos:end_pos]

    return text_preprocessing(content)



zip_file = root_path + 'NIKL_NP_v1.2(비출판물말뭉치).zip'
file = root_path + '비출판물말뭉치'

success, fail = unzip(zip_file, file) # 압축파일명, 생성할 폴더명
print(f'success : {success}\t fail : {fail}') # success : 10757	 fail : 0


pdf_file_path =file +'/NIKL_NP_v1.2/국립국어원 비출판물 말뭉치(버전 1.2)/국립국어원 비출판물 말뭉치(버전 1.1).pdf'

if os.path.exists(pdf_file_path):
    os.remove(pdf_file_path)
    
file_list = glob.glob(file+'/NIKL_NP_v1.2/국립국어원 비출판물 말뭉치(버전 1.2)/*') # 폴더안에 pdf 파일은 삭제한 상태

corpus = []
for file_path in tqdm(file_list):
    with open(file_path, "r",encoding="utf8") as f:
        xml = f.read()
        
    soup = BeautifulSoup(xml, "lxml")
    category_tag = soup.find("category")
    
    # 일기 카테고리만 텍스트 가져오기
    content_tag = soup.find("text")
    corpus.append(content_tag.get_text())
    
category = []
for i,file in enumerate(file_list):
        with open(file, 'r', encoding='utf-8') as f:
            raw_corpus = f.read()
            category.append(find_content("category", raw_corpus).strip())  # category

emotion = ['']*len(corpus)
comment = ['']*len(corpus)

df = pd.DataFrame({'diary':corpus, 'comment':comment, 'emotion':emotion, 'source':category})
df = df[df["source"].apply(lambda x: x.count('일기')>0)]
df["id"]=list(range(len(df)))
df = df[['id','diary','comment','emotion','source']]

df.to_csv(root_path+'국립국어원_비출판물.csv',index=False)