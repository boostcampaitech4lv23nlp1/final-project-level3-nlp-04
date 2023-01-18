__author__ = 'Sanhee Park'
__email__ = 'carpediembackup20@gmail.com'
__version__ = '1.0.1'
__refer__ = 'Chanwoo Yoon'

import re

import hgtk
from tqdm.auto import tqdm

from kiwipiepy import Kiwi
# from kdictionary import informaldic, formaldic, abnormaldic
# from utils import Utils

def informaldic():
    return {
        ('예요', 'EF') : ('야', 'EF'),
        ('에요', 'EF') : ('야', 'EF'),
        ('어요', 'EF') : ('어', 'EF'),
        ('시', 'EP') : ('', ''),
        ('죠', 'EF') : ('지', 'EF'),
        ('ᆸ니까', 'EF') :('하나', 'NNP'),
        ('습니까', 'EF') : ('나', 'EC'),
        ('습니까', 'EC') : ('나', 'EC'),
        #
        ('안녕하세요', 'NNP') : ('안녕', 'IC'),       
        # exception : NNG + EF
        ('이예', 'NNP') : ('야', 'EF'),
        (('에', 'JKB'), ('요', 'JX')) : ('야', 'JKV'),
    }

def abnormaldic():
    """
    Change morphological anomally words
    """
    return {
                    ('ᆯ', 'ETM') : ('ㄹ', 'SW'),
                    ('ᆸ', 'NNG') : ('ㅂ', 'SW'), 
                    ('ᆫ', 'ETM') : ('ㄴ' ,'SW'), 
    }


def formaldic():
    """
    Change informal words to formal words.
    :return: Dictionary structure data set
    """

    return {
                    # 어, 아, 야, 지 
                    # ("어", "EF"): ("세요", "EP+EF"),
                    # ("어", "EF"): ("ㅓ요", "EF"),
                    # ("어", "EC"): ("어요", "EC"),
                    # ('어', "EF") : ('습니다', 'EF'),
                    ("어", "EF"): ("ㅂ니다", "EF"),
                    ('어요', 'EF'): ('습니다', 'EF'),
                    ('요', 'JX') : ('ㅂ니다', 'EF'),
                    # ('어', 'EF'): ('ㅂ니까', 'EF'),
                    # ("어", "EF"): ("십시오", "EP+EF"),
                    # ("아", "EC"): ("아요", "EC"),
                    # ("아", "EF"): ("아요", "EF"),
                    # ("지", "EF"): ("지요", "EF"),
                    ("지", "EF"): ("죠", "EF"),
                    # ("자", "EF"): ("ㅂ시다", "EF"),

                    ("나", "NP"): ("저", "NP"),
                    ("야", "EF"): ("에요", "EF"),
                    ("군", "EF"): ("군요", "EF"),
                    ("네", "EF"): ("네요", "EF"),
                    ("더군", "EF"): ("더군요", "EF"),
                    ("", ""): ("요", "EF"),
                    ('야', 'JKV'): ("ㅂ니다", "EF"),

                    # ("은", "ETM"): ("으신", "EP+ETM"),
                    # ("는지", "EF"): ("는지요", "EF"),
                    # ("는데", "EF"): ("는데요", "EF"),
                    # ("거든", "EF"): ("거든요", "EF"),
                    # ("다지", "EF"): ("다지요", "EF"),



                    ("내", "NP"): ("제", "NP"),
                    ("너", "NP"): ("당신", "NP"),
                    # ("우리", "NP"): ("저희", "NP"),
                    # ("함", "XSV+ETN"): ("ㅁ", "ETN"),


                    # ('네', 'EC'): ('네요', 'EC'),
                    # ('군', 'EC'): ('군요', 'EC'),
                    # ('자', 'EC'): ('ㅂ시다', 'EC'),

                    ('는가', 'EC'): ('는가요', 'EF'),
                    # ('는다', 'EC'): ('습니다', 'EC'),
                    ('을게', 'EF'): ('을게요', 'EF'),

                    # ('ㅁ', 'ETN'): ('ㅂ니다', 'EC'),
                    # ('가', 'XSN'): ('ㄴ가요', 'EC'),

                    # ('나', 'EC'): ('나요', 'EC'),
                    # ('나', 'JX') : ('나요', 'EC'),
                    # ('구나', 'EC'): ('군요', 'EC'),
                    ('냐', 'EF'): ('나요', 'EF'),
                    # ('어라', 'EC'): ('어요', 'EC'),
                    ('으렴', 'EF'): ('으세요', 'EF'),
                    ('렴', 'EF'): ('세요', 'EF'),
                    # ('어야지', 'EC'): ('어야지요', 'EF'),
                    # ('지', 'EC'): ('지요', 'EC'),
                    # ('마', 'EC'): ('ㄹ게요', 'EC'),
                    ('을걸', 'EF'): ('을걸요', 'EF'),

                    # ('랴', 'EF'): ('ㄹ까요', 'EF'),
                    ('대', 'EF'): ('대요', 'EF'),
                    # ('아라', 'EC'): ('세요', 'EF'),
                    # ('너라', 'EC'): ('세요', 'EF'),
                    ('려무나', 'EF'): ('세요', 'EF'),
                    ('라고', 'EF'): ('라고요', 'EF'),
                    # ('아야지', 'EC'): ('아야지요', 'EC'),
                    ('네', 'XSN'): ('네요', 'XSN'),
                    # ('냐', 'EC'): ('나요', 'EC'),
                    ('니', 'EF'): ('ㅂ니까', 'EF'),

                    #요
                    ('갈래', 'NNG'): ('갈래요', 'NNG'),
                    ('셈', 'NNB'): ('세요', 'NNB'),
                    # ('라', 'EC'): ('요', 'EC'),
                    # ('냐고', 'EC'): ('냐고요', 'EC'),
                    # ('니', 'EC'): ('ㅂ니까', 'EC'),
                    ('잖아', 'EF'): ('잖아요', 'EF'),
                    ('나', 'EF'): ('나요', 'EF'),
                    ('해', 'NNG'): ('해요', 'NNG'),
                    # ('라', 'EC'): ('아요', 'EC'),
                    # ('냐', 'EF'): ('ㄹ까요', 'EC'),
                    # ('려나', 'EC'): ('려나요', 'EC'),
                    ('마', 'NNG'): ('마요', 'NNG'),
                    
                    # 다
                    # ("이야", "JX"): ("입니다", "EF"),
                    # ('함', 'NNG'): ('합니다', "FF"),
                    # ('하자', 'NNG'): ('합시다', 'NNG'),
                    # ('련다', 'EF'): ('렵니다', 'EF'),                    
                    # ('자', 'XSN'): ('ㅂ시다', 'EC'),
                    # ('보자', 'NNG'): ('봅시다', 'NNG'),
                    # ('ㅁ', 'NNG'): ('ㅂ니다', 'EC'),
                    # ('은데', 'EC') : ('습니다', 'EF'),
                    # ('야', 'EF') : ('ㅂ니다', 'EF'),
                    # ('가', 'JKS') : (('가', 'JKS'), ("ㅂ니다", "EC")),

                    # ㅂ                
                    # ("임", "NNG"): (('이', 'VCP'), ("ㅂ니다", "EC")),
                    # ('다', 'EC'): ("ㅂ니다", "EC"),
                    # ('다', 'JKB'): ('ㅂ니다', 'JKB'),
                    # ('는다', 'EF'): ('습니다', 'EC'),
                    # ('다', 'NNG'): ('ㅂ니다', 'NNG'),
                    # ('다', 'NNG'): ('ㅂ니다', 'NNG'),
                    # ('다면서', 'EC'): ('다면서요', 'EF'),
                    # ('이다', 'JC'): ("입니다", "EF"),
                    # ("다", "EF"): ("ㅂ니다", "EC"),
                    # ('다', 'JX'): ("입니다", "EC"),
                    # ("단다", "EF"): ("답니다", "EF"),
                    # ("ᆫ단다", "EF"): ("ㄴ답니다", "EF"),
                    # ("는단다", "EF"): ("는답니다", "EF"),
                    # ("이다", "EF"): ("입니다", "EF"),

                    # ㄹ
                    # ('ᆯ까', 'EF'): ('ㄹ까요', 'EC'),
                    # ('ᆯ래', 'EC'): ('ㄹ래요', 'EF'),
                    ('ᆯ래', 'EF'): ('ㄹ래요', 'EF'),
                    # ('ᆯ게', 'EC'): ('ㄹ게요', 'EC'),
                    # ('ᆯ걸', 'EC'): ('ㄹ걸요', 'EC'),
                    # ('ᆯ라고', 'EC'): ('ㄹ라고요', 'EC'),
                    ("ᆯ까", "EF"): ("ㄹ까요", "EF"),
                    ('ᆯ', 'JKO') : ("ㄹ" , "SW"),

                    # ㄴ                    
                    # ("ᆫ다", "EC"): ("ㅂ니다", "EC"),
                    ("ᆫ대", "EF"): ("ㄴ대요", "EF"),
                    # ("ᆫ대", "EC"): ("ㄴ대요", "EC"),
                    # ("ᆫ다", "EF"): ("ㅂ니다", "EF"),
                    ('ᆫ다고', 'EF'): ('ㄴ다고요', 'EF'),
                    # ('ᆫ가', 'EC'): ('ㄴ가요', 'EC'),
                    ('건가', 'NNP') : ('건가요', 'NNP'),

                    # exception
                    ('같아', 'NNP') : (('같', 'VA'), ('습니다', 'EC')),
                    ('없어', 'NNP') : (('없', 'VA'), ('습니다', 'EF')),
                    ('세', 'EC') : (('세', 'EC'), ('요', 'JX')),

                    }
            

class Changer(object):
    def __init__(self):    
        try:
            self.kiwi = Kiwi()
            self.kiwi.prepare()
        except:
            print("[INFO] please install kiwipiepy   ")
            
        self.replace = formaldic()
        # self.utils = Utils()        

    def _makePretty(self, line):
        """
        Convert the jaso orderings which wasn't properly covered by
        Jaso restructuring process of function Mal_Gillge_Haeraing
        :param line: jaso orderings which wasn't properly covered
        :return: Converted jaso ordering
        """
        test = line
        test = test.replace("ᴥㅎㅏᴥㅇㅏᴥ", "ᴥㅎㅐᴥ")
        test = test.replace("ㅎㅏᴥㅇㅏᴥㅇㅛᴥ", "ᴥㅎㅐᴥ")
        test = test.replace("ㅎㅏᴥㄴㅣᴥㄷㅏᴥ", "ㅎㅏㅂᴥㄴㅣᴥㄷㅏᴥ")
        test = test.replace("ㅎㅏᴥㅇㅏㅆᴥ", "ᴥㅎㅐㅆᴥ")
        test = test.replace("ㅎㅏᴥㅇㅓㅆᴥ", "ᴥㅎㅐㅆᴥ")
        test = test.replace("ㄴㅏᴥㅇㅏㅆᴥ", "ᴥㅎㅐㅆᴥ")
        test = test.replace("ㄴㅏᴥㅇㅓㅆᴥ", "ᴥㅎㅐㅆᴥ")
        test = test.replace("ㄱㅏᴥㅇㅏㅆᴥ", "ᴥㄱㅏㅆᴥ")
        test = test.replace("ㄱㅏᴥㅇㅓㅆᴥ", "ᴥㄱㅏㅆᴥ")
        test = test.replace("ㅇㅣᴥㄴㅣᴥ", "ᴥㄴㅣᴥ")
        test = test.replace("ㄴㅓㄹㄴᴥ","ㄴㅓㄴᴥ")
        test = test.replace("ㄱㅡᴥㄹㅓㅎᴥㅇㅓᴥ","ㄱㅡᴥㄹㅐᴥ")
        test = test.replace("ㅡᴥㅇㅏᴥ","ㅏᴥ")
        test = test.replace("ㄱㅓㄹᴥㄴㅏᴥㅇㅛᴥ", "ㄱㅓㄴᴥㄱㅏᴥㅇㅛᴥ")
        
        return test

    def changer(self, text):
        """
        change informal speech to formal speech
        Args : str
        """
        tokens = self.kiwi.analyze(text.replace(" ","|"))
        key = formaldic().keys()
        key2 = abnormaldic().keys()
        lk = list(key)
        ak = list(key2)
        num = len(tokens[0][0])
        result = []
       
        for idx, token in enumerate(tokens[0][0]):
            if idx > int(num*0.8):        
                if (token[0], token[1]) in lk:
                    #key로 value
                    token = formaldic().get((token[0], token[1]))
                    result.append(token)
                else:
                    if (token[0], token[1]) in ak:
                        token = abnormaldic().get((token[0], token[1]))
                        result.append(token)
                    else:
                        result.append((token[0], token[1]))
            else:
                if (token[0], token[1]) in ak:
                    token = abnormaldic().get((token[0], token[1]))
                    result.append(token)
                else:
                    result.append((token[0], token[1]))
                
        # change tuple to text
        changed = ''
        for t in result:
            if isinstance(t[0], tuple):
                for i in range(len(t[0])):
                    changed += hgtk.text.decompose(t[i][0])
            else:
                changed += hgtk.text.decompose(t[0])

        # Restructuring sentence from jaso ordering.
        one_char = re.compile('ᴥ[ㅂㄴㄹ]ᴥ')
        if one_char.search(changed):
            words = changed.split('ᴥ')
            for idx in range(1,len(words)):
                # 앞 글자가 종성이 없음
                if len(words[idx]) == 1 and len(words[idx-1].replace('|',"")) == 2:
                    #앞 글자에 합침
                    words[idx - 1] = words[idx-1]+words[idx]
                    words[idx] = ""
                # 있음
                elif len(words[idx]) == 1 and len(words[idx-1].replace('|',"")) == 3:
                    shp = ['ㅆ','ㅍ','ㄱ','ㅄ','ㄶ']
                    ep = ['ㄹ']
                    if words[idx] == 'ㅂ' and len(words[idx - 1].replace('|', "")) == 3 :
                        if words[idx - 1][-1] in shp :
                            if words[idx].count("|") > 0:
                                words[idx] = "|습"
                            else:
                                words[idx ] = "습"
                            continue
                        else :
                            if words[idx].count("|") > 0:
                                words[idx] = "|입"
                            else:
                                words[idx] = "입"
                            # words[idx] = ""
                    elif words[idx] =='ㄴ' and len(words[idx-1].replace('|',"")) == 3 and words[idx - 1].endswith('ㄹ'):
                        if words[idx-1].count("|") >0 :
                            words[idx - 1] = "|" + words[idx - 1].replace("|","")[:2] + words[idx]
                        else :
                            words[idx - 1] = words[idx - 1][:2] + words[idx]
                        # 지움
                        words[idx] = ""
                    elif words[idx] =='ㄹ':
                        if words[idx].count("|") > 0:
                            words[idx] = "|일"
                        else:
                            words[idx] = "일"

            changed = "ᴥ".join([x for x in words if x != ""])+"ᴥ"
        # For cases which wasn't covered,
        changed = self._makePretty(changed)
        changed = hgtk.text.compose(changed).replace("|"," ")
        return changed
    

# changer = Changer()
# result = changer.changer("밥 먹으러 가고 있다.")
# print('####################', result)