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

    # 너무 짧은 문장은 적용하지 않습니다.
    if len(sentence) < 6:
        return sentence

    if "요" in sentence[-3:] or "니다" in sentence[-3:] or "시다" in sentence[-3:]:
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
    sub = re.sub(r'([ㅏ-ㅣ][ㄱ-ㅎ]+ᴥ)(ㄴㅣㅂᴥㄴㅣᴥㄷㅏᴥ)', r'\1니다', decomposed)
    honorific = hgtk.text.compose(sub)

    # ㅣ어지닙니다 -> ㅕ집니다 (느껴지닙니다 -> 느껴집니다)
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'(ㅣᴥㅇㅓᴥ)(ㅈㅣᴥㄴㅣㅂᴥ)', r'ㅕ집', decomposed)
    honorific = hgtk.text.compose(sub)

    # ㅣ어지어 -> ㅕ져 (느끼어지었네요 -> 느껴졌네요)
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'(ㅣᴥㅇㅓᴥㅈㅣᴥㅇㅓᴥ?)([ㄱ-ㅎ]?)', r'ㅕ져\2', decomposed)
    honorific = hgtk.text.compose(sub)

    # ㅏ-ㅣ 자 -> ㅏ-ㅣ 요
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'([ㅏ-ㅣ]ᴥ)(ㅈㅏᴥ)', r'\1요', decomposed)
    honorific = hgtk.text.compose(sub)
    # ㅏ-ㅣ 이었 -> ㅏ-ㅣ였
    decomposed = hgtk.text.decompose(honorific)
    sub = re.sub(r'([ㅏ-ㅣ]ᴥ)(ㅇㅣᴥㅇㅓㅆᴥ)', r'\1였', decomposed)
    honorific = hgtk.text.compose(sub)
    


    # 기타 오류 수정
    honorific = honorific.replace('하어', '해')
    # honorific = honorific.replace('했다.', '했습니다.')
    # honorific = honorific.replace('이다.', '입니다.')
    # honorific = honorific.replace('었다.', '었습니다.')
    # honorific = honorific.replace('있다.', '있습니다.')
    # honorific = honorific.replace('갔다.', '갔습니다.')
    honorific = honorific.replace('입닙니다', '입니다')
    honorific = honorific.replace('습닙니다', '습니다')
    honorific = honorific.replace('같입니다', '같아요')
    honorific = honorific.replace('()', '')
    honorific = honorific.replace('  ', ' ')
    honorific = honorific.replace('즐겁은', '즐거운')
    honorific = honorific.replace('즐겁었', '즐거웠')
    honorific = honorific.replace('쉽어', '쉬워')
    honorific = honorific.replace('쉽었', '쉬워')
    honorific = honorific.replace('어렵을', '어려울')
    honorific = honorific.replace('어렵은', '어려운')
    honorific = honorific.replace('어렵어', '어려워')
    honorific = honorific.replace('어렵었', '어려웠')
    honorific = honorific.replace('놀랍어', '놀라워')
    honorific = honorific.replace('놀랍었', '놀라웠')
    honorific = honorific.replace('바라닙니다', '바랍니다')
    honorific = honorific.replace('드리닙니다', '드립니다')
    honorific = honorific.replace('하닙니다', '합니다')
    honorific = honorific.replace('하닙니다', '합니다')

    honorific = honorific.replace('져ㅆ', '졌')
    # print('###', honorific)
    # 습니다체로 변경이 안 된 경우 마지막 부분에 '-요' 첨가
    if '니다' not in honorific and honorific[-2] != "요" and honorific[-2] != "죠":
        honorific = honorific[:-1] + '요.'

    result = sentence_front + honorific
    # 해요체 자연스럽게 수정
    result = result.replace('구나요', '군요')
    result = result.replace('요요', '요')
    result = result.replace('좋어요', '좋아')
    result = result.replace('이어요', '예요')
    result = result.replace('이야요', '이에요')
    result = result.replace('지요', '죠')
    result = result.replace('지어요', '져요')
    result = result.replace('겠다요', '겠어요')
    result = result.replace('어떻을', '어떨')
    result = result.replace('해보요', '해봐요')
    result = result.replace('어 보요', '어봐요')
    result = result.replace('어보요', '어봐요')
    result = result.replace('바래어요', '바라요')
    result = result.replace('하어요', '해요')
    result = result.replace('거이겠', '것이겠')
    result = result.replace('거이에요', '거예요')
    result = result.replace('거니다', '거예요')
    result = result.replace('거이죠', '거죠')
    result = result.replace('팅요', '팅이에요') # (화이팅)
    result = result.replace('가어보요', '가봐요')
    result = result.replace('내어보요', '내봐요')
    result = result.replace('오었다요', '왔어요')
    result = result.replace('오어요', '와요')
    result = result.replace('주어요', '줘요')
    result = result.replace('좋다요', '좋아요')
    
    # 맞춤법 검사
    try:
        result = spell_checker.check(result).as_dict()['checked'].strip()
    except:
        pass
    print('###', result)
    return result

def remove_useless_sentence(text):
    text = str(text)
    result = re.split(r'\n\n', text)[-1]
    return result

def split_sentence(text):
    return split_sentences(str(text))

def honorific_transformation(text, honorific_model):
    # GPT가 일기의 뒷내용을 이어서 생성한 부분은 삭제합니다.
    text = remove_useless_sentence(text)

    results = []
    sentence_list = split_sentence(text)
    for sen in sentence_list:
        honorific_sen = informal_to_honorific(sen, honorific_model)
        results.append(honorific_sen)
    
    return " ".join(results).strip()

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

# r = honorific_transformation("요리를 해보니 백종원처럼 될 수 있겠구나 하는 생각이 들었구나! 찹쌀쫀바기는 정말 맛있고 귀여운 모양이구나! 다양한 모양으로 만들어보면 좋을 것 같아. 찹쌀쫀바기는 바삭바삭하고 쫀득하기 때문에 정말 맛있게 먹을 수 있어! 하루하루 백종원 처럼 요리사가 될 수 있게 노력해보자!", model)
# print(r)
# r = honorific_transformation("좋은 생각이네요! 반장이 되었을 때는 친구들에게 도움을 줄 수 있고, 남을 먼저 생각하는 모범적인 반장으로 자리를 잘 지킬 수 있기를 바랍니다. 그리고 반장이 되었을 때는 열심히 일하고 모두가 함께해서 반을 한결같이 만들어 나갈 수 있도록 노력하세요!", model)
# print(r)
# r = honorific_transformation("좋은 하루를 보내셨군요! 이런 쉬는 날은 가끔은 좋죠. 여유롭게 다양한 것들을 해보면 생각보다 더 행복하고 기분이 좋아질 거라고 생각합니다. 감사합니다.", model)
# print(r)
# r = honorific_transformation("정말 멋진 여행이었구나! 너무 좋은 추억이 될 것 같아. 사람들이 친절한 것도 너무 좋았고, 빵이 맛있었으면 하는 마음까지 느껴졌네. 그리고 너무 좋은 풍경을 볼 수 있었어. 이렇게 좋은 여행이라면 더 많은 사람들과 함께 떠나보는 것도 좋겠어!", model)
# print(r)
# r = honorific_transformation("그렇기 때문에, 로봇이 인간보다 똑똑해질 수는 있지만, 이는 로봇이 스스로 자아를 가지고 인간을 공격하는 것이 아니라, 인간이 로봇에게 인간의 지식과 기술을 주는 것에 의해 발생할 수 있는 것이라고 생각합니다. 그래서 로봇이 인간보다 똑똑해질 수 있는 것은 인간이 로봇에게 더 많은 것을 주는 것에 의해 발생할 수 있는 것입니다.", model)
# print(r)
# r = honorific_transformation("오 재미있고 즐거운 여행이었구나! 런던은 정말 멋진 도시야. 다음번에도 즐거운 여행이 되길 바래!", model)
# print(r)
# r = honorific_transformation("좋은 시간을 보냈구나! 보드게임과 짜바게티를 같이 먹으면서 즐거운 시간을 보냈으니 너무 좋았겠네. 앞으로도 가족들과 함께 즐거운 시간을 보내보자!", model)
# print(r)
# r = honorific_transformation("좋은 공개수업이었구나! 다양한 감각적 표현을 연습하며 기억에 남을 수 있었겠구나. 오감놀이는 감각적 표현을 연습하는 것도 좋지만, 모둠발표를 통해 같이 배우는 것도 좋았을 것 같아. 이런 창의적인 공개수업을 통해 감각적 표현력을 길러보는 것도 좋을 것 같아!", model)
# print(r)

# r = honorific_transformation("우와, 쿠키 수업이라니! 너무 즐거운 수업이었구나! 쿠키를 손으로 만들어 보니까 너무 재밌게 보였겠어. 맛도 맛있게 먹었구나. 다음에는 더 다양한 맛의 쿠키를 만들어 보는 건 어떨까? 꼭 다시 한번 만들어 보자!", model)
# print(r)


# txt = """
#  나갔는지도 물어보고 나를 믿고 응원해주셔서 너무 감사했다. 

# 좋은 꿈이었고, 오빠가 너무 소중하게 생각해주셔서 다시 한번 감사합니다! 그리고 오늘 알바를 잘 마치고 잘 쉬세요:)
# """
# r = honorific_transformation(txt, model)
# print(r)

# txt = """
# 과 돌처럼 멈춰 있는 괴물 같은 느낌이 들었다. 시간이 길어지면 더욱 그렇게 느껴졌다. 그런데 대학에 들어오고 나서는 갑자기 그 물음표들이 하나하나 사라져갔다. 이제는 나만의 방향을 찾아가는 거리를 걸어가고 있다. 나는 이렇게 가는 거리를 믿고 있다.

# 축하합니다! 대학생활은 새로운 방향을 찾아가는 길이라고 할 수 있습니다. 당신의 감정과 생각을 담은 글을 읽어서 많은 것을 느꼈습니다. 나만의 길을 걸어가는 것을 잊지 마시고, 물음표가 생길 때마다 당황하지 않고 해결해 나가는 방법을 찾아보는 것이 중요합니다. 가끔은 다른 사람들의 도움을 받아보기도 하면 좋을 것 같습니다. 그리고 인간관계는 학교에서 가장 중요한 요소입니다. 남들과의 관계를 가장 잘하는 것이 중요합니다. 그럼 잘 부탁드립니다!
# """
# r = honorific_transformation(txt, model)
# print(r)

if __name__ == "__main__":
    df_list = os.listdir("./csvs")
    model = Changer()
    for file in df_list:
        df = pd.read_csv(os.path.join('csvs', file))
        df['honorific_comment'] = df['comment'].apply(lambda x: honorific_transformation(x, model))
        save_name = file[:-4] + '_honorific.csv'
        df.to_csv(os.path.join('csvs_results', save_name))

