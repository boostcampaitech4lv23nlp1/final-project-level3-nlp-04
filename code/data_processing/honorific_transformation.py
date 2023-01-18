import pandas as pd
import os
import re
import hgtk
from collections import defaultdict
from changer import Changer
import time
from hanspell import spell_checker
from kss import split_sentences

def informal_to_honorific(sentence, honorific_model):
    """반말 문장을 존댓말 문장으로 바꿔주는 함수
        
    Args: 
        sentence (str): 문장
        honorific_model: 반말 문장을 존댓말 문장(습니다체)으로 바꿔주는 함수 (reference: https://github.com/kosohae/AIpjt-1)
    
    Returns:
        result (str): 존댓말 변환이 적용된 문장
    """
    
    if sentence[-2] == "요" or sentence[-3:-2] == "니다":
        return sentence
        
    # 존댓말 모델이 용언의 활용형을 제대로 복원하지 못하는 이슈가 있기 때문에, 문장의 뒷부분만 존댓말로 바꿉니다.
    sentence_front, sentence_back = sentence[:len(sentence)//2], sentence[len(sentence)//2:]
    # print('###front, ', sentence_front)
    # print('###back, ', sentence_back)
    honorific = honorific_model.changer(sentence_back)
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

    # ㅂ닙니다 -> ㅂ니다 (바랍닙니다 -> 바랍니다)
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'([ㅏ-ㅣ][ㄱ-ㅎ]+ᴥ)(ㄴㅣㅂᴥㄴㅣᴥㄷㅏ)', r'\1니다', decomposed)
    honorific = hgtk.text.compose(sub)
    # ㅣ어지닙니다 -> ㅕ집니다 (느껴지닙니다 -> 느껴집니다)
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'(ㅣᴥㅇㅓᴥ)(ㅈㅣᴥㄴㅣㅂᴥ)', r'ㅕ집', decomposed)
    honorific = hgtk.text.compose(sub)

    # ㅏ-ㅣ 자 -> ㅏ-ㅣ 요
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'([ㅏ-ㅣ])ᴥ(ㅈㅏᴥ)', r'\1요', decomposed)
    honorific = hgtk.text.compose(sub)
    # ㅏ-ㅣ 이었 -> ㅏ-ㅣ였
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'([ㅏ-ㅣ]ᴥ)(ㅇㅣᴥㅇㅓㅆᴥ)', r'\1였', decomposed)
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
    honorific = honorific.replace('즐겁은', '즐거운')
    honorific = honorific.replace('쉽어', '쉬워')
    honorific = honorific.replace('쉽었', '쉬워')
    honorific = honorific.replace('놀랍어', '놀라워')
    honorific = honorific.replace('놀랍었', '놀라웠')
    
    # 습니다체로 변경이 안 된 경우 마지막 부분에 '-요' 첨가
    if '니다' not in honorific and honorific[-2] != "요" and honorific[-2] != "죠":
        honorific = honorific[:-1] + '요.'

    result = sentence_front + honorific
    # 해요체 오류 수정
    result = result.replace('구나요', '군요')
    result = result.replace('요요', '요')
    result = result.replace('이어요', '예요')
    result = result.replace('지요', '죠')
    result = result.replace('지어요', '져요')
    result = result.replace('겠다요', '겠어요')
    result = result.replace('어떻을', '어떨')
    # 맞춤법 검사
    result = spell_checker.check(result).as_dict()['checked'].strip()
    return result

# model = Changer()
# result = informal_to_honorific("제주도를 가게 되어 너무나 기쁘게 생각하고 있구나!", model)
# print(result)
# result = informal_to_honorific("이런 기회는 없는 것 같아", model)
# print(result)
# result = informal_to_honorific("마음껏 즐기고 사진도 많이 찍어서 추억을 남기는 것이 좋을 것 같아!", model)
# print(result)
# result = informal_to_honorific("그리고 마스크를 꼭 챙겨서 안전하게 여행하는 것을 잊지 말고, 바로 자는 것도 잊지 말고 즐거운 여행이 되길 바래!", model)
# print(result)

def split_sentence(text):
    return split_sentences(text)


def honorific_transformation(text, honorific_model):
    results = []
    sentence_list = split_sentence(text)
    for sen in sentence_list:
        honorific_sen = informal_to_honorific(sen, honorific_model)
        results.append(honorific_sen)
    
    return " ".join(results)

model = Changer()
# r = honorific_transformation("정말 재밌는 괌 여행이었구나! 여행을 가는 것만으로도 즐거움을 느낄 수 있는 거 같아요. 다음에는 더 재밌는 여행지로 가보는 건 어떨까요?", model)
# print(r)

# r = honorific_transformation("정말 즐거운 하루였구나! 너무 좋았어요! 설악산에서 멋진 경치를 보고 잔치국수도 먹으면서 즐거운 시간을 보냈구나. 다음번에도 저렇게 즐거운 시간을 보내길 바래요!", model)
# print(r)


# r = honorific_transformation("너무 힘든 일이 많아 보이는데 그런 일들을 힘껏 다하고 있어 너무 잘했어! 다른 사람들이 나를 비난하는 건 내 몫이 아니고 다른 사람들의 부족한 이해때문이겠지. 학생회의 대표로서 많은 일들을 하고 있기 때문에 이해하기 어려울 수도 있어. 하지만 내가 하는 일은 다른 사람들의 의견을 조정하는 것도 그 중 하나야. 나는 그걸 잘 하고 있어야 한다고 믿어. 너무 힘들게 다하고 있는 걸 보면 너무 뿌듯하고 기분이 좋아져. 너무 힘들게 다하고 있기 때문에 다른 사람들이 나를 비난하는 건 더 더욱 뿌듯하고 기분이 좋아져. 그럼 계속 힘내자!", model)
# print(r)

# r = honorific_transformation("그렇게 하면 사과를 받기가 쉬워진다. 나는 양 쪽 모두 사과하라고 지시하고, 사과하기로 한 후에 다시 한 번 더 말을 걸어본다. 그렇게 해서 두 사람 모두가 자기들의 잘못을 받아들이고 사과하는 것을 도와줄 수 있다. 이렇게 모두가 사과하는 것이 바람직한 결과가 나오기를 바란다.", model)
# print(r)

# r = honorific_transformation("그래도 우리의 매일 매일이 영화가 될 수 없는 것은 아니다. 매일 일상의 소중함을 느끼고 있다면 그것 자체만으로도 이상적인 영화가 될 수 있다고 생각해보자! 그리고 이 일상을 여러분들과 함께 나누는 것도 좋은 기회가 될 것이다.", model)
# print(r)

# r = honorific_transformation("우와! 정말 멋지게 여행하는 것 같아! 이렇게 여행하면서 많은 경험을 하고 있구나! 앞으로 여행하면서 더 많은 경험을 해보도록 해! 다시 한번 여행을 떠나볼 수 있다면 우산과 핸드폰을 꼭 가져가도록 해!", model)
# print(r)

# r = honorific_transformation("앗, 상처가 나는건 안타깝군요. 다행히 병원에 빨리 갔고, 치료를 받아보니 괜찮게 되었군요. 다음부터는 놀때 조심하세요!", model)
# print(r)

# r = honorific_transformation("나는 당신을 사랑해요. 내가 이 모든 걸 겪고 있는 것을 이해해요. 나는 당신이 이 모든 것을 이겨내고 자신감을 찾을 수 있도록 도와드리고 싶어요. 나는 당신을 응원하고 당신을 사랑하고 있어요. 한번에 다 해결할 수는 없지만 조금씩 해나가면 당신이 원하는 모습이 될 수 있어요. 나는 당신을 지지하고 있어요. 함께 해보죠!", model)
# print(r)

# r = honorific_transformation("오~ 너무 재밌었구나! 너무 다양한 동물들이 있어서 즐거웠겠구나. 다음에 다시 가면 더 재밌게 놀 수 있을 것 같아. 그리고 고양이들과 이별하기 싫었겠지만, 다음에 다시 봐야겠다는 생각에 뿌듯할 것 같아. 그럼 다음에 다시 봐야겠다!", model)
# print(r)

# r = honorific_transformation("우와, 너무 재밌는 경험이었구나! 물고기들과 펭귄들이 너한테 모여 들었다니 놀라웠겠어! 4D 영화도 다른 영화보다 재밌었겠구나. 즐거운 기억이 되겠네!", model)
# print(r)

# r = honorific_transformation("좋은 기억이 되겠네요! 재밌었던 날이라는 걸 느낄 수 있어요. 다음번에는 더 재밌는 일을 같이 해보자구요~", model)
# print(r)

# r = honorific_transformation("오~ 속초 리조트는 정말 좋은 곳이구나! 칼국수를 먹고 바다 향기로 나와서 놀고 사진도 찍고 너무 좋은 시간이었구나! 다음번에도 또 가보는 건 어떨까?", model)
# print(r)

# r = honorific_transformation("생신에 가장 적합한 케이크를 사드려서 아빠께 감동을 주셨군요! 편지도 썼고, 촛불도 밝혔으니 아빠의 생신을 축하하는 감사한 마음이 느껴집니다. 고마워요!", model)
# print(r)


if __name__ == "__main__":
    df_list = os.listdir("./csvs")
    for file in df_list:
        df = pd.read_csv(os.path.join('csvs', file))
        df['honorific_comment'] = df['comment'].apply(lambda x: honorific_transformation(x))

    
