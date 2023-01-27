import streamlit as st
import base64
import requests

# 서버에서의 streamlit 실행 방법
# streamlit run mainpage.py --server.fileWatcherType none --server.port=30001


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css?family=Jua&display=swap');
@import url('https://fonts.googleapis.com/css?family=Gowun+Dodum&display=swap');

@font-face {
    font-family: 'BMDoHyeon';
    font-weight: normal; 
    font-style: normal; 
    src: url(https://cdn.jsdelivr.net/gh/webfontworld/woowahan/BMDoHyeon.woff2) format('woff2');
    font-display: swap;
}

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
    color: #969696; 
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

.box_content{
    font-size: 16px;
    margin: 5px;
}

.text{
    font-size: 18px;
}

.fade-in-title {
    padding: 10px;
    font-family: 'Cafe24Shiningstar';
    text-align: center;
    font-size: 75px;
    color: #282828;
    animation: fadein 3s;
    -webkit-animation: fadein 3s; /* Safari and Chrome */
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

.emotion_box{
    font-family: 'Gowun Dodum', serif;
    justify-content: center;
    box-sizing: border-box;
    margin: 15px;
    width: 150px;
    border-stype: solid;
    border-width: 250px;
    background-color: #fff8dc;
    color: #5a5a5a;
    border-radius: 5px 5px;
    padding: 10px;
    display: flex;
}

.comment_box{
    font-family: 'Gowun Dodum', serif;
    background-color: #fff5ee;
    color: #5a5a5a;
    box-sizing: border-box;
    margin: 15px;
    border-stype: solid;
    border-width: 250px;
    border-radius: 5px 5px;
    padding: 10px;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
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
st.markdown('<p class="fade-in-title">오늘 하루는 어땠어요?</p>', unsafe_allow_html =True)
st.markdown('<p class="subtitle">당신의 하루를 이야기해주세요.</p>', unsafe_allow_html=True)

## write diary
diary = st.text_area(label='Please put your diary', key='diary_key', height=270, max_chars=700, label_visibility="hidden")
_, col, _ = st.columns([2.2]*2+[1.18])
writting_btn = col.button("나의 하루 보내기")


user_diary = {'diary_content': diary}


if writting_btn:
    
    response = requests.post("http://localhost:8080/diary", json=user_diary)
    
    emotions = response.json()["emotions"]
    comment = response.json()['comment']
    
    st.markdown(f'''
    <div class="container">
        <div class="emotion_box"> {emotions} </div>
        <div class="comment_box"> {comment} </div>
    </div>
    ''', unsafe_allow_html=True)

    # st.markdown(f'''
    # <div class="container">
    #     <div class="emotion_box"> #좋아요 <br> #행복 </div>
    #     <div class="comment_box"> 그동안 고생을 많이 했군요! 앞으로 당신에게 펼쳐질 미래를 응원해요. 다음에는 이렇게 해보는건 어떨까요? </div>
    # </div>
    # ''', unsafe_allow_html=True)


