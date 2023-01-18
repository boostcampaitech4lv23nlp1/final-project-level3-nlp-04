import pandas as pd
import re
import hgtk
from collections import defaultdict
from changer import Changer
import time
from hanspell import spell_checker

def informal_to_honorific(sentence, honorific_model):
    """반말 문장을 존댓말 문장으로 바꿔주는 함수
        
    Args: 
        sentence (str): 문장
        honorific_model: 반말 문장을 존댓말 문장(습니다체)으로 바꿔주는 함수 (reference: https://github.com/kosohae/AIpjt-1)
    
    Returns:
        result (str): 존댓말 변환이 적용된 문장
    """
        
    honorific = honorific_model.changer(sentence)
    
    # 존댓말 변형 오류 교정
    decomposed = hgtk.text.decompose(honorific)
    # 하었습니다, 가었습니다 등 -> 핬습니다, 갔습니다
    sub = re.sub(r'ㅏᴥㅇㅓㅆ', r'ㅏㅆ', decomposed)
    honorific = hgtk.text.compose(sub)
    #핬습니다 -> 했습니다
    sub = re.sub('핬', '했', honorific)
    honorific = hgtk.text.compose(sub)
    # 막었습니다, 받었습니다 -> 막았습니다. 받았습니다
    decomposed = hgtk.text.decompose(honorific)
    honorific = re.sub(r'(ㅏ[ㄱ-ㅎ]ᴥ)ㅇㅓㅆ', r'\1았', decomposed)
    honorific = hgtk.text.compose(sub)
    
    # 부른다 -> 부릅니다., 치르다 -> 치릅니다., 올린다 -> 올립니다
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'([ㅏ-ㅣ][ㄱ-ㅎ]?ᴥ)([ㄱ-ㅎ]ᴥ?[ㅏ-ㅣ])ㄴ?ᴥㄷㅏᴥ', r'\1\2ㅂ니다', decomposed)
    honorific = hgtk.text.compose(sub)
    
    # 기타 오류 수정
    honorific = honorific.replace('하어', '해')
    honorific = honorific.replace('했다.', '했습니다.')
    honorific = honorific.replace('이다.', '입니다.')
    honorific = honorific.replace('었다.', '었습니다.')
    honorific = honorific.replace('있다.', '있습니다.')
    honorific = honorific.replace('갔다.', '갔습니다.')
    honorific = honorific.replace('입닙니다.', '입니다.')
    honorific = honorific.replace('습닙니다.', '습니다.')
    honorific = honorific.replace('()', '')
    honorific = honorific.replace('  ', ' ')
    
    # 습니다체로 변경이 안 된 경우 마지막 부분에 '-요' 첨가
    if '니다' not in honorific:
        honorific = honorific[:-1] + '요.'
    result = honorific

    # 맞춤법 검사
    result = spell_checker.check(result).as_dict()['checked'].strip()
    return result

model = Changer()
result = informal_to_honorific("제주도를 가게 되어 너무나 기쁘게 생각하고 있구나!", model)
print(result)
result = informal_to_honorific("이런 기회는 없는 것 같아", model)
print(result)
result = informal_to_honorific("마음껏 즐기고 사진도 많이 찍어서 추억을 남기는 것이 좋을 것 같아!", model)
print(result)
result = informal_to_honorific("그리고 마스크를 꼭 챙겨서 안전하게 여행하는 것을 잊지 말고, 바로 자는 것도 잊지 말고 즐거운 여행이 되길 바래!", model)
print(result)