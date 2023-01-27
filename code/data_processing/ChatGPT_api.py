import openai
from tqdm import tqdm
import pandas as pd

# put your api key
openai.api_key = "***REMOVED***"
model_engine = "text-davinci-003"
# put your data path
path = '../dataset/extra_diary.csv'

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
    if 'comment' in data.columns:
        error = data['comment'][i] == 'error'
    else:
        error = True
    
    if error:
        try: 
            response = chatgpt(f'이 일기에 코멘트를 달아줘.{text}')
        except: 
            response = 'error'
    else:
        response = data['comment'][i]
    
    response_list.append(response)

data['comment'] = response_list

# export to csv file
data.to_csv('../dataset/extra_comment_data.csv')

# text = """
# 2018년 8월 26일
# 나는 여러 일로 인내의 한계를 느껴 본 것 같다.(사촌 동생 &name2& 이와 &name1& 이 때문에) 어제 예상했던 것처럼 나는 오늘 동생 3명을 혼자 돌보았다. 처음으로 동생들을 엄마 아빠없이 돌보았는데 아무리 노력해도 마음처럼 쉽게 잘 되지 않았다. 사촌동생들은 단독주택에 살고 우리는 아파트 3층에 산다. 게다가 아랫 집에는 며칠전에 태어난 갓난 아이도 살고 있다. 나는 사촌동생들에게 아랫 집에 갓난 아이가 살고 있으니 뛰면 안되고 조용히 놀아야 한다고 이야기 해주었고 사촌동생들도 ""알았어'라고 대답했다. 하지만 시간이 지날수록 사촌동생들은 나를 무시하듯 쿵쾅거리며 뛰어 다녔고 내가 &name1& 이에게 뛰면 안된다고 이야기 하면 웃으며 ""알았어""라고 대답할 뿐 계속 뛰어 다니며 심지어는 내 바이올린 케이스 위에서도  뛰었다. 또 &name2& 이는 방학 숙제를 해야 한다고 했는데 방학 숙제는 거들떠 보지도 않고 뛰어 다니며 놀기 바빴다. 가장 심한 건 내 동생 &name3& 이었다. 평소엔 얌전했는데 사촌 동생들이 오니 뛰는 것을 말리지는 못할 망정 함께 뛰어 다니며 놀기 바빴다... 다음부터는 &name2&이와 &name1&이가 우리 집에 오지 않았으면 좋겠고 오더라도 삼촌이나 숙모가 함께 있었으면 좋겠다.
# """

# response = chatgpt(f'이 일기에 코멘트를 달아줘.{text}')
# print(response)