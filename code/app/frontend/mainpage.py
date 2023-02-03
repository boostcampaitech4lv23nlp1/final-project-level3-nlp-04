import streamlit as st
import base64
import requests
from configs import emotion_day

# 서버에서의 streamlit 실행 방법
# streamlit run mainpage.py --server.fileWatcherType none --server.port=30001


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css?family=Gowun+Dodum&display=swap');

@font-face {
    font-family: 'Cafe24Shiningstar';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_twelve@1.1/Cafe24Shiningstar.woff') format('woff');
    font-weight: normal;
    font-style: normal;
}

.title{
    font-family: 'Cafe24Shiningstar';
    text-align: center;
    font-size: 75px;
    color: #282828;
}


.subtitle{
    font-family: 'Gowun Dodum', serif;
    text-align: center;
    font-size: 20px;
    margin-bottom: 0px;
    color: #8c8c8c; 
}

.diary_box{
    box-sizing: border-box;
    margin-bottom: 13px;
    border-style: solid;
    border-color: grey;
    border-width: 0px;
    border-radius: 5px 5px;
    box-shadow: 2px 2px 2px 2px #CCCCCC;
    padding: 25px;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
}

.fade-out-title {
    font-family: 'Cafe24Shiningstar';
    text-align: center;
    font-size: 75px;
    color: #282828;
    animation: fadeout 3s;
    -webkit-animation: fadeout 3s; /* Safari and Chrome */
    animation-fill-mode: forwards;
}

@keyframes fadeout {
    from {
        opacity: 1;
    }
    to {
        opacity: 0;
    }
}

@-webkit-keyframes fadeout { /* Safari and Chrome */
    from {
        opacity: 1;
    }
    to {
        opacity: 0;
    }
}

@keyframes fadein {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@-webkit-keyframes fadein { /* Safari and Chrome */
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}


.container{
    display: flex;
    flex-direction: row;
    justify-content: center;
    margin: 5px;
    flex-wrap: nowrap;
}

.text{
    font-family: 'Gowun Dodum', serif;
    text-align: center;
    font-size: 15px;
    margin-bottom: 0px;
    color: #8c8c8c; 
    animation: fadein 1s;
    -webkit-animation: fadein 1s; /* Safari and Chrome */
    animation-fill-mode: forwards;
}

.emotion_box{
    font-family: 'Gowun Dodum', serif;
    justify-content: center;
    box-sizing: border-box;
    margin: 15px;
    width: auto;
    align-items: center;
    border-stype: solid;
    border-width: 250px;
    background-color: #fff8dc;
    color: #5a5a5a;
    border-radius: 5px 5px;
    padding: 10px;
    white-space: pre;
    display: flex;
}

.comment_box{
    font-family: 'Gowun Dodum', serif;
    background-color: #fff5ee;
    align-items: center;
    color: #5a5a5a;
    box-sizing: border-box;
    margin: 15px;
    border-stype: solid;
    border-width: 250px;
    border-radius: 5px 5px;
    padding: 10px;
    display: flex;
}


</style>""", unsafe_allow_html=True)


## setting background img 
def get_base64_of_bin_file(bin_file):
    """
    function to read jpeg file 
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(jpeg_file, header_file):
    bin_str = get_base64_of_bin_file(jpeg_file)
    bin_str2 = get_base64_of_bin_file(header_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    header_bg_img = '''
    <style>
    [data-testid="stHeader"] {
          background: url(data:image/jpeg;base64,%s);
      }
    </style>
    ''' % bin_str2
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown(header_bg_img, unsafe_allow_html=True)
    return
    

set_png_as_page_bg('./background.jpeg', './moon.jpeg')


## setting title of project
st.markdown('''
<div class="container">
    <div class="title"> 너 </div>
    <div class="fade-out-title"> 의 </div> 
    <div class="title">&nbsp;&nbsp;하</div>
    <div class="fade-out-title"> 루 </div>
    <div class="fade-out-title"> 가 </div> 
    <div class="title">&nbsp;궁</div>
    <div class="fade-out-title"> 금 </div>
    <div class="fade-out-title"> 해 </div> 
    <div class="title"> &#127769; </div> 
</div>
''', unsafe_allow_html=True)


st.markdown('<p class="subtitle">당신의 하루를 이야기해주세요.</p>', unsafe_allow_html=True)

## write diary
diary = st.text_area(label='Please put your diary', placeholder = '일기를 최소 20자 이상 작성해주세요!', key='diary_key', height=270, label_visibility="hidden")
_, col, _ = st.columns([2.2]*2+[1.18])
writting_btn = col.button("나의 하루 보내기")


user_diary = {'diary_content': diary}


if writting_btn:
    st.markdown(f'''
    <br>
    <div class="container">
    <div class="text"> 당신의 하루에서 느껴지는<div class="emotion_box"; style="display:inline; color:#969696; margin:10px;">감정 해시태그</div>와 당신에게 들려주고 싶은 <div class="comment_box"; style="display:inline; color:#969696; margin:10px;"> 제 마음 </div>을 전할게요. </div>
    </div>
    <hr>
    ''', unsafe_allow_html=True)

    response = requests.post("http://115.85.181.5:30002/diary", json=user_diary)
    
    emotions = response.json()["emotions"]
    comment = response.json()['comment']

    ## 감정은 최대 top 2까지 출력되므로
    if len(emotions) == 2:
        str_emotions = emotion_day[emotions[0]]+'\n'+emotion_day[emotions[1]]
    else:
        str_emotions = emotion_day[emotions[0]]


    st.markdown(f"""
    <div class="container">
        <div class="emotion_box"> {str_emotions} </div>
        <div class="comment_box"> {comment} </div>
    </div>
    """, unsafe_allow_html=True)